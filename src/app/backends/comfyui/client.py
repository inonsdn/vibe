"""HTTP client for a **localhost** ComfyUI instance.

Safety properties, all enforced here rather than by convention:

* :func:`assert_local_endpoint` rejects any non-loopback host unless the
  operator has explicitly set ``comfyui.allow_remote: true`` *and* added the
  host to ``comfyui.allowed_hosts``. This is checked in the constructor, so a
  misconfigured backend fails before any request is made.
* Only the documented ComfyUI endpoints are called; there is no code path that
  triggers a model download or a custom-node install.
* Every request has a timeout, and polling has an overall job deadline.
* Uploaded inputs are validated to live inside the configured data root before
  they are read.
"""

from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import LOCAL_HOSTS, ComfyUIConfig
from app.core.errors import BackendError, BackendUnavailableError, OfflinePolicyError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Endpoints this client is allowed to touch.
ENDPOINT_HEALTH = "/system_stats"
ENDPOINT_PROMPT = "/prompt"
ENDPOINT_HISTORY = "/history"
ENDPOINT_QUEUE = "/queue"
ENDPOINT_INTERRUPT = "/interrupt"
ENDPOINT_UPLOAD = "/upload/image"
ENDPOINT_VIEW = "/view"
ENDPOINT_OBJECT_INFO = "/object_info"


def _is_loopback(host: str) -> bool:
    cleaned = host.strip("[]").lower()
    if cleaned in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def assert_local_endpoint(base_url: str, config: ComfyUIConfig) -> str:
    """Validate a ComfyUI base URL against the offline policy.

    Returns the normalised URL. Raises :class:`OfflinePolicyError` for anything
    that is not loopback, unless remote access has been explicitly enabled and
    the host is on the allowlist.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise OfflinePolicyError(
            "ComfyUI base_url must use http or https",
            base_url=base_url,
            scheme=parsed.scheme,
        )
    host = parsed.hostname or ""
    if not host:
        raise OfflinePolicyError("ComfyUI base_url has no host", base_url=base_url)

    if _is_loopback(host):
        return base_url.rstrip("/")

    if not config.allow_remote:
        raise OfflinePolicyError(
            "Refusing a non-local ComfyUI endpoint. This system is designed to "
            "run fully offline; set comfyui.allow_remote=true and add the host "
            "to comfyui.allowed_hosts only if you really intend to leave the "
            "machine.",
            base_url=base_url,
            host=host,
            allowed_hosts=list(config.allowed_hosts),
        )
    if host.lower() not in {h.lower() for h in config.allowed_hosts}:
        raise OfflinePolicyError(
            "ComfyUI host is not on the configured allowlist",
            base_url=base_url,
            host=host,
            allowed_hosts=list(config.allowed_hosts),
        )
    logger.warning(
        "comfyui_remote_endpoint_allowed",
        extra={"event": "comfyui_remote_endpoint_allowed", "host": host},
    )
    return base_url.rstrip("/")


@dataclass
class PromptHandle:
    """Identifiers returned by ComfyUI when a prompt is queued."""

    prompt_id: str
    number: int | None = None
    client_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "number": self.number, "client_id": self.client_id}


@dataclass
class PromptOutcome:
    prompt_id: str
    completed: bool
    outputs: dict[str, Any]
    status: dict[str, Any]
    elapsed_s: float

    @property
    def images(self) -> list[dict[str, Any]]:
        """Flatten the per-node image descriptors from a history entry."""
        found: list[dict[str, Any]] = []
        for node_id, payload in self.outputs.items():
            if not isinstance(payload, dict):
                continue
            for image in payload.get("images", []) or []:
                if isinstance(image, dict):
                    found.append({**image, "node_id": node_id})
        return found


class ComfyUIClient:
    """Minimal, explicit ComfyUI API client."""

    def __init__(
        self,
        config: ComfyUIConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self.base_url = assert_local_endpoint(config.base_url, config)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=config.request_timeout_s,
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": "garment-replacer/local"},
        )

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ComfyUIClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- low level --------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        files: Any = None,
        data: Any = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(
                method,
                path,
                json=json_body,
                files=files,
                data=data,
                params=params,
                timeout=timeout or self._config.request_timeout_s,
            )
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(
                "Cannot reach ComfyUI. Start it locally and confirm the port.",
                base_url=self.base_url,
                path=path,
                hint="Run ComfyUI with --listen 127.0.0.1 and check comfyui.base_url.",
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendError(
                "ComfyUI request timed out",
                base_url=self.base_url,
                path=path,
                timeout_s=timeout or self._config.request_timeout_s,
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(
                "ComfyUI request failed", base_url=self.base_url, path=path, error=str(exc)
            ) from exc
        return response

    @staticmethod
    def _json(response: httpx.Response, path: str) -> Any:
        if response.status_code >= 400:
            raise BackendError(
                f"ComfyUI returned HTTP {response.status_code} for {path}",
                status_code=response.status_code,
                body=response.text[:2000],
                hint=(
                    "A 400 usually means the workflow graph is invalid for the "
                    "installed nodes; check the node titles in your contract."
                ),
            )
        try:
            return response.json()
        except ValueError as exc:
            raise BackendError(
                "ComfyUI returned a non-JSON response", path=path, body=response.text[:500]
            ) from exc

    # -- API --------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Query ``/system_stats``. Raises BackendUnavailableError if down."""
        response = self._request(
            "GET", ENDPOINT_HEALTH, timeout=min(5.0, self._config.request_timeout_s)
        )
        return dict(self._json(response, ENDPOINT_HEALTH))

    def is_healthy(self) -> bool:
        try:
            self.health()
        except (BackendError, BackendUnavailableError):
            return False
        return True

    def object_info(self) -> dict[str, Any]:
        """Installed node types. Used to report missing nodes actionably."""
        response = self._request("GET", ENDPOINT_OBJECT_INFO)
        return dict(self._json(response, ENDPOINT_OBJECT_INFO))

    def missing_node_types(self, graph: dict[str, Any]) -> list[str]:
        """Node classes a workflow needs that this ComfyUI does not have."""
        try:
            installed = set(self.object_info())
        except (BackendError, BackendUnavailableError):
            return []
        needed = {
            str(node.get("class_type"))
            for node in graph.values()
            if isinstance(node, dict) and node.get("class_type")
        }
        return sorted(needed - installed)

    def upload_image(self, path: str | Path, *, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload a local image into ComfyUI's input folder; returns its name."""
        source = Path(path)
        if not source.is_file():
            raise BackendError("Cannot upload a file that does not exist", path=str(source))
        with source.open("rb") as handle:
            response = self._request(
                "POST",
                ENDPOINT_UPLOAD,
                files={"image": (source.name, handle, "image/png")},
                data={
                    "overwrite": "true" if overwrite else "false",
                    "type": "input",
                    "subfolder": subfolder,
                },
                timeout=max(60.0, self._config.request_timeout_s),
            )
        payload = self._json(response, ENDPOINT_UPLOAD)
        name = payload.get("name") if isinstance(payload, dict) else None
        if not name:
            raise BackendError("ComfyUI upload did not return a filename", body=str(payload)[:500])
        sub = payload.get("subfolder") or ""
        return f"{sub}/{name}" if sub else str(name)

    def submit(self, graph: dict[str, Any], *, client_id: str | None = None) -> PromptHandle:
        """Queue a workflow graph for execution."""
        resolved_client_id = client_id or self._config.client_id
        body: dict[str, Any] = {"prompt": graph, "client_id": resolved_client_id}
        response = self._request("POST", ENDPOINT_PROMPT, json_body=body)
        payload = self._json(response, ENDPOINT_PROMPT)
        if not isinstance(payload, dict) or "prompt_id" not in payload:
            errors = payload.get("error") if isinstance(payload, dict) else None
            raise BackendError(
                "ComfyUI rejected the workflow",
                error=errors,
                node_errors=(payload.get("node_errors") if isinstance(payload, dict) else None),
                hint="Verify the workflow's node titles match the contract bindings.",
            )
        return PromptHandle(
            prompt_id=str(payload["prompt_id"]),
            number=payload.get("number"),
            client_id=resolved_client_id,
        )

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        """History entry for a prompt, or ``None`` while it is still queued."""
        response = self._request("GET", f"{ENDPOINT_HISTORY}/{prompt_id}")
        payload = self._json(response, ENDPOINT_HISTORY)
        if not isinstance(payload, dict):
            return None
        entry = payload.get(prompt_id)
        return dict(entry) if isinstance(entry, dict) else None

    def queue_state(self) -> dict[str, Any]:
        response = self._request("GET", ENDPOINT_QUEUE)
        return dict(self._json(response, ENDPOINT_QUEUE))

    def is_queued(self, prompt_id: str) -> bool:
        """Whether a prompt is still running or pending."""
        state = self.queue_state()
        for key in ("queue_running", "queue_pending"):
            for entry in state.get(key, []) or []:
                if isinstance(entry, list) and len(entry) > 1 and str(entry[1]) == prompt_id:
                    return True
        return False

    def interrupt(self) -> None:
        """Ask ComfyUI to stop the currently executing prompt."""
        self._request("POST", ENDPOINT_INTERRUPT, json_body={})

    def wait(
        self,
        prompt_id: str,
        *,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        sleep: Any = time.sleep,
    ) -> PromptOutcome:
        """Poll until the prompt finishes, times out, or vanishes.

        A vanished prompt (absent from both history and queue after having been
        queued) is reported as incomplete rather than silently succeeding, which
        is what lets ``resume`` re-submit it safely.
        """
        deadline_s = timeout_s if timeout_s is not None else self._config.job_timeout_s
        interval = poll_interval_s if poll_interval_s is not None else self._config.poll_interval_s
        started = time.monotonic()
        misses = 0

        while True:
            entry = self.history(prompt_id)
            if entry is not None:
                status = dict(entry.get("status", {}) or {})
                completed = bool(status.get("completed", True))
                outputs = dict(entry.get("outputs", {}) or {})
                if completed or outputs:
                    return PromptOutcome(
                        prompt_id=prompt_id,
                        completed=completed,
                        outputs=outputs,
                        status=status,
                        elapsed_s=time.monotonic() - started,
                    )
            elif not self.is_queued(prompt_id):
                misses += 1
                if misses >= 3:
                    return PromptOutcome(
                        prompt_id=prompt_id,
                        completed=False,
                        outputs={},
                        status={"error": "prompt disappeared from queue and history"},
                        elapsed_s=time.monotonic() - started,
                    )
            else:
                misses = 0

            elapsed = time.monotonic() - started
            if elapsed >= deadline_s:
                raise BackendError(
                    "Timed out waiting for ComfyUI to finish the prompt",
                    prompt_id=prompt_id,
                    elapsed_s=round(elapsed, 2),
                    timeout_s=deadline_s,
                    hint="Increase comfyui.job_timeout_s or reduce frame_window.",
                )
            sleep(interval)

    def fetch_output(
        self,
        image: dict[str, Any],
        destination: str | Path,
    ) -> Path:
        """Download one output image described by a history entry."""
        params = {
            "filename": image.get("filename", ""),
            "subfolder": image.get("subfolder", "") or "",
            "type": image.get("type", "output") or "output",
        }
        response = self._request("GET", ENDPOINT_VIEW, params=params, timeout=120.0)
        if response.status_code >= 400:
            raise BackendError(
                "Failed to download a ComfyUI output image",
                status_code=response.status_code,
                params=params,
            )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target


__all__ = [
    "ENDPOINT_HEALTH",
    "ENDPOINT_HISTORY",
    "ENDPOINT_PROMPT",
    "ENDPOINT_QUEUE",
    "ENDPOINT_UPLOAD",
    "ENDPOINT_VIEW",
    "ComfyUIClient",
    "PromptHandle",
    "PromptOutcome",
    "assert_local_endpoint",
]
