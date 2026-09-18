"""FastAPI application factory.

Localhost-only by default, no CORS unless explicitly configured, no telemetry,
no outbound requests. Domain errors are translated into their declared HTTP
status with a stable machine-readable ``code``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api import deps
from app.api.routers import (
    compatibility,
    garments,
    jobs,
    master,
    motion,
    system,
    templates,
)
from app.core.config import AppConfig, load_config
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, new_correlation_id
from app.version import APP_NAME, APP_VERSION

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None, **context_kwargs: Any) -> FastAPI:
    resolved = config or load_config()
    configure_logging(
        resolved.runtime.log_level,
        log_dir=(resolved.data_root().resolve("logs") if resolved.runtime.log_to_file else None),
    )

    # Logging is already configured above; callers may still override the rest.
    context_options: dict[str, Any] = {"configure_logs": False, **context_kwargs}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        deps.configure(resolved, **context_options)
        try:
            yield
        finally:
            deps.close()

    application = FastAPI(
        title=f"{APP_NAME} (local)",
        version=APP_VERSION,
        description=(
            "Local, model-agnostic video garment replacement. The master human "
            "performance is immutable; only the garment region of the reveal "
            "segment is re-rendered. No cloud, no telemetry, localhost only."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    if resolved.api.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.api.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    @application.middleware("http")
    async def correlate(request: Request, call_next: Any) -> Any:
        correlation_id = new_correlation_id()
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["x-correlation-id"] = correlation_id
        return response

    @application.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "api_error",
            extra={"event": "api_error", "code": exc.code, "error_message": exc.message},
        )
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    application.include_router(system.router)
    application.include_router(templates.router)
    application.include_router(garments.router)
    application.include_router(compatibility.router)
    application.include_router(jobs.router)
    application.include_router(motion.router)
    application.include_router(master.router)

    if resolved.api.enable_web_ui:
        application.get("/", response_class=HTMLResponse, include_in_schema=False)(_index)

    return application


async def _index() -> str:
    """A deliberately minimal operator landing page.

    The pipeline, API, CLI, data integrity and tests are the product; this page
    exists so an operator can confirm the service is up and jump to the docs.
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME}</title>
<style>
  :root {{ color-scheme: light dark; --fg: #16181d; --bg: #fbfbfd; --muted: #5b6270;
           --line: #d8dbe2; --accent: #2f6feb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e8eaf0; --bg: #13151a; --muted: #9aa2b1; --line: #2a2e38;
             --accent: #6f9bff; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--fg);
          font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 46rem; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }}
  p.sub {{ color: var(--muted); margin: 0 0 2rem; }}
  section {{ border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.25rem;
             margin-bottom: 1rem; }}
  h2 {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;
        color: var(--muted); margin: 0 0 .75rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em;
          background: color-mix(in oklab, var(--fg) 8%, transparent);
          padding: .1em .35em; border-radius: 4px; }}
  ul {{ margin: 0; padding-left: 1.1rem; }}
  li {{ margin: .3rem 0; }}
  a {{ color: var(--accent); }}
  .note {{ color: var(--muted); font-size: .9em; }}
</style>
</head>
<body>
<main>
  <h1>{APP_NAME}</h1>
  <p class="sub">Version {APP_VERSION} &middot; local only &middot; no cloud, no telemetry</p>

  <section>
    <h2>Endpoints</h2>
    <ul>
      <li><a href="/docs">/docs</a> &mdash; interactive API reference</li>
      <li><a href="/health">/health</a> &mdash; service and schema status</li>
      <li><a href="/offline/verify">/offline/verify</a> &mdash; offline guarantees</li>
      <li><a href="/adapters">/adapters</a> &mdash; preprocessing adapter status</li>
      <li><a href="/jobs/backends">/jobs/backends</a> &mdash; renderer capabilities</li>
    </ul>
  </section>

  <section>
    <h2>Operator workflow</h2>
    <ul>
      <li><code>app doctor</code></li>
      <li><code>app template ingest &lt;video&gt; --name "..."</code></li>
      <li><code>app template import-masks &lt;id&gt; --kind garment --from &lt;dir&gt;</code></li>
      <li><code>app garment ingest --image front=&lt;png&gt; ...</code></li>
      <li><code>app compatibility check --template &lt;id&gt; --garment &lt;id&gt;</code></li>
      <li><code>app job create ... &amp;&amp; app job render &lt;id&gt; --backend mock</code></li>
      <li><code>app job compose &lt;id&gt; &amp;&amp; app job qc &lt;id&gt;</code></li>
    </ul>
  </section>

  <section>
    <h2>Model status</h2>
    <p class="note">
      No AI model weights are installed, referenced or downloaded. The
      <code>mock</code> backend renders deterministic placeholder garments so the
      whole pipeline can be exercised; the <code>comfyui</code> backend submits a
      workflow you author locally. See <code>docs/model-selection-checklist.md</code>.
    </p>
  </section>
</main>
</body>
</html>
"""


__all__ = ["create_app"]
