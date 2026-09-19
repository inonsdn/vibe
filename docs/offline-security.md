# Offline and security model

## The claim

This system performs no outbound network access. That is a statement about the
code, so it is stated positively and tested:

```python
# src/app/offline/verify.py
CLOUD_INTEGRATIONS: tuple[str, ...] = ()
NETWORK_FEATURES: tuple[str, ...] = ()
```

`tests/unit/test_offline_and_env.py::test_no_cloud_integrations_are_declared`
asserts both remain empty, so the list cannot quietly grow.

Verify on your own machine:

```bash
app offline verify
```

It reports every configured endpoint and whether the policy allows it, which
local executables exist, whether every data path is writable, GPU/CUDA info when
available, and each adapter's status.

## What is absent, on purpose

No analytics. No telemetry. No crash reporting. No cloud storage. No model
hubs. No auto-updaters. No automatic weight downloads. No license checks. No
package installation at runtime.

The only network code in the repository is an `httpx` client pointed at
**localhost ComfyUI**, and it refuses to point anywhere else.

## Endpoint policy

| Endpoint | Default | Non-local? |
| --- | --- | --- |
| ComfyUI | `http://127.0.0.1:8188` | Refused unless `comfyui.allow_remote: true` **and** the host is in `comfyui.allowed_hosts`. |
| API bind | `127.0.0.1:8077` | Refused unless `api.allow_remote_bind: true`. |

`assert_local_endpoint()` runs in the ComfyUI client's **constructor**, so a
misconfigured backend fails before any request can be made — not mid-render.

The client we own is also built with **`trust_env=False`**. Without it, httpx
reads `HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` from the environment and
routes loopback traffic through whatever they name — so an operator's corporate
proxy could intercept local rendering traffic, and an `ALL_PROXY=socks5://…`
variable made the client refuse to construct at all unless the optional
`socksio` package happened to be installed. Both were real failures. A client a
*caller* injects is left untouched: that caller owns the decision.
`tests/unit/test_proxy_and_deps.py` sets every proxy variable to a deliberately
broken value and asserts the CLI, the API and backend inspection all still
work. It
recognises loopback by IP semantics (`ipaddress.is_loopback`), not by string
matching, so `127.0.0.2` is correctly local and `10.0.0.5` is correctly not.
`tests/unit/test_comfyui.py` covers eight non-local URL shapes including
private ranges, a hostname, and an IPv6 literal.

FFmpeg I/O is checked too: `assert_local_path()` rejects `http://`, `https://`,
`rtmp://`, `rtsp://`, `udp://`, `tcp://`, `ftp://`, `srt://`, `sftp://`,
`pipe:`, `concat:` and `data:`. A crafted "video path" cannot turn ffmpeg into
a network client.

## Tests cannot reach the network

`tests/conftest.py` installs an autouse fixture that patches
`socket.socket.connect`, `connect_ex`, `socket.create_connection`,
`socket.getaddrinfo` and `socket.gethostbyname`. Any attempt to reach — or even
*resolve* — a non-loopback address raises `NetworkAccessAttempted` and fails the
test that made it. Each guard closes over the original function captured before
patching, so delegating to the real implementation cannot recurse.

Loopback is judged by IP semantics (`ipaddress.is_loopback`), not string
matching, so `127.0.0.53` is correctly local. `0.0.0.0` is deliberately **not**
allowed: it is a bind wildcard, not a loopback destination, and treating a
public-interface bind as equivalent to a loopback API bind is exactly the
conflation the offline policy exists to prevent. Loopback stays permitted because FastAPI's
`TestClient` and the ComfyUI transport tests run in-process; blocking it would
break the harness rather than catch a real call.

ComfyUI tests use `httpx.MockTransport` — no socket is opened at all.

## Path safety

Every path an operator supplies — CLI flag, API payload, YAML value — resolves
through `DataRoot.resolve()`:

* symlinks are resolved **before** the containment check
* `..` is allowed only if the resolved result stays inside the root
* absolute paths are allowed only if they are already inside the root
* UNC and `//server/share` paths are rejected outright
* Windows drive hops (`C:\…` when the root is on `D:`) are rejected
* NUL bytes are rejected
* directory-name identifiers must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`
  and must not be a Windows reserved name (`CON`, `NUL`, `COM1`, …)

Twelve escape vectors are covered in `tests/unit/test_paths.py`, including a
symlink pointing outside the root and traversal attempted through the HTTP API.

## Destructive operations

The only deletion command is `app job delete`, and it:

1. resolves exactly **one** existing job's directory,
2. confirms that path equals the canonical `<data_root>/jobs/<job_id>`,
3. refuses without `--yes` (printing what it *would* delete),
4. records the deletion in the audit log.

There is no recursive delete of a broad or unresolved path anywhere in the
codebase. Template ingestion refuses to write into a non-empty template
directory rather than clearing it. Compose refuses to overwrite an existing
output without `--overwrite`.

## Immutability

`source_frames/` is hashed at ingestion and re-verified before every render and
compose. A single changed byte raises `ImmutabilityError` and the render stops —
the system will not produce output from a source it cannot vouch for.

Applied migrations are hashed too: editing a migration file that has already run
aborts startup rather than leaving the schema in an unverifiable state.

## Secrets

There are none to manage — no API keys, tokens or credentials, because there is
nothing to authenticate to. `.gitignore` excludes `.env*`, `config/local.*` and
`*.local.yaml` so machine-specific overrides stay out of Git regardless.

## Motion references

A motion reference is a recording of a real person. Three separate protections
apply:

1. **Its pixels are never copied.** Ingestion probes and hashes the file; it
   does not extract frames. The reference stays where the operator put it.
2. **Motion use is asserted explicitly.** `MotionUsageRights.motion_use_authorized`
   is a separate flag from holding the clip, and composition is *refused*
   without it. A `depicted_person_consent_ref` field records the consent.
3. **The guarantee is checked, not assumed.** The
   `no_source_pixels_in_motion_artifacts` QC check fails if any image or video
   file appears in a composition's pose directories, and a test plants one to
   prove the check fires.

The skeleton preview is drawn from pose data on a flat background, so it cannot
leak a reference person's appearance either.

## Dependency pinning as a safety property

`constraints/tested-py311.txt` pins the versions the suite was actually run
against. This is not bookkeeping: unconstrained resolution installed NumPy 2.5
with OpenCV 5.0, and `import cv2` aborted the interpreter with exit code 135
before a single test could run — a failure that looks like a broken repository.
`pyproject.toml` carries conservative upper bounds and a test asserts both the
bounds and the installed versions, so the combination cannot silently return.

## Consent and provenance

Every template carries a `ConsentRecord`, and every **Hero Character** carries
the same discipline (`subject_kind`, mandatory `adult_confirmed`, a required
`consent_document_ref` for a `consented_human`). Validated at the schema level:

* `subject_kind` is exactly `synthetic` or `consented_human`
* `adult_confirmed` must be `true` — a template cannot be persisted otherwise
* `consented_human` requires a `consent_document_ref`
* optional: rights holder, license, grant/expiry dates, usage restrictions,
  provenance notes

`ConsentRecord.is_expired()` supports time-limited consent. Garment usage
rights are recorded and are part of the compatibility rules — undocumented
rights produce `NEEDS_INPUT`.

## Logging and data handling

Structured JSON logs go to stderr and optionally to
`data/logs/app.jsonl` (rotating, 16 MB × 5). They contain identifiers, frame
indices, hashes and metrics — no pixel data, no image content, no credentials.
There are no network log handlers.

All media stays in the configured data root. Nothing is uploaded anywhere. The
`source_url` recorded for a garment is metadata for attribution and is
deliberately never dereferenced.

## Threat model

**Defended against.** An operator mistake that would corrupt an immutable
source; a path that would escape the data root; a misconfiguration that would
send frames to a remote host; a backend that returns pixels outside its mask; a
crashed render silently shipping a missing frame; an unreviewed render of a
blocked garment.

**Not defended against.** A malicious local operator with filesystem access; a
compromised ComfyUI install on the same machine; malicious content in a
hand-authored workflow JSON. This is a single-operator local tool, and adding
authentication theatre to a localhost service would be worse, not safer.


## Local model weights (DWPose ONNX)

The first real model integration changes nothing about the offline posture.

* **No downloads, ever.** `pose.detector_model` and `pose.pose_model` are local
  filesystem paths. An empty or missing path produces a refusal naming the path;
  there is no hub client, no URL and no fetch-on-first-use anywhere in
  `app/adapters/dwpose/`. A test greps the shipped source for URLs and fetchers
  and fails if one appears.
* **No network at inference.** onnxruntime runs a local file on the local
  device. The test suite's network guard covers the whole extraction path.
* **Weights are not committed.** `*.onnx` is gitignored alongside every other
  checkpoint format, and `config/local.yaml` — which holds machine-specific
  model paths — is gitignored too. `config/local.example.yaml` is committed and
  contains only placeholder paths.
* **Which weights ran is recorded.** Both ONNX files are hashed into the motion
  source record and the composition manifest, so a manifest identifies the
  weights rather than a filename that could later be overwritten.
* **Diagnostics can be turned off.** `--no-overlay` suppresses the only artefact
  that contains source pixels.
