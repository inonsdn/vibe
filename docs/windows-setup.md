# Windows 11 setup (RTX 5060 8GB)

## 1. Python 3.11+

Install from [python.org](https://www.python.org/downloads/windows/) and tick
**Add python.exe to PATH**. Verify:

```powershell
python --version      # 3.11.x or newer
```

The Microsoft Store build of Python works but sandboxes file access in ways that
complicate a local data directory; the python.org installer is the smoother
choice here.

## 2. FFmpeg

Required — the pipeline probes, extracts and encodes with it. Either:

```powershell
winget install --id Gyan.FFmpeg
```

or download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/),
unzip to `C:\ffmpeg`, and add `C:\ffmpeg\bin` to `Path`. Verify both binaries:

```powershell
ffmpeg -version
ffprobe -version
```

## 3. NVIDIA driver

Install the current Game Ready or Studio driver for the RTX 5060. No CUDA
toolkit is needed for this repository — nothing here compiles CUDA kernels, and
the mock backend and full test suite run on CPU. A driver is needed only when
you later bring a real model in via ComfyUI.

```powershell
nvidia-smi        # should list the RTX 5060 and ~8188 MiB
```

## 4. This project

```powershell
git clone <repository-url> garment-replacer
cd garment-replacer

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Install against the TESTED pins. The -c flag is not optional polish:
# unconstrained resolution has installed NumPy 2.5 with OpenCV 5.0, and
# `import cv2` then aborts the interpreter (exit code 135) before a single
# test can run -- a failure that looks like a broken repository.
python -m pip install -e ".[dev]" -c constraints\tested-py311.txt
```

### Dependency versions

`pyproject.toml` carries deliberately conservative ranges:

```
numpy>=1.26,<2.3
opencv-python-headless>=4.9,<5
```

`constraints/tested-py311.txt` pins the exact versions the suite was last run
against on Python 3.11. Compatibility with NumPy 2.3+ or OpenCV 5.x is
**untested and not claimed**. To move the ceiling: raise the bound in
`pyproject.toml`, update the pins, and re-run the full suite on Windows 11
before relying on it.

Verify what you actually got:

```powershell
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

A crash on that line rather than a version print is the symptom the pins exist
to prevent.

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 5. Verify

```powershell
app doctor
```

Expect: both endpoints local, `ffmpeg`/`ffprobe` found, every data path
writable, all seven adapters `not_implemented` (correct — no models yet), the
`mock` backend healthy, and `comfyui` unavailable until you start it.

```powershell
python -m pytest
```

All tests must pass with no GPU, no weights, no ComfyUI and no network.

If your machine has corporate proxy variables set, the suite must still pass
unchanged — localhost traffic deliberately ignores them. Verify both ways:

```powershell
python -m pytest                                   # as configured
$env:ALL_PROXY = "socks5://proxy.invalid:1080"     # deliberately broken
python -m pytest
Remove-Item Env:\ALL_PROXY
```

## 6. Pose extraction (DWPose ONNX)

Needed as soon as you want the system to produce pose data rather than import
it. onnxruntime is deliberately **not** a dependency of this package, so you
choose the GPU or CPU build:

```powershell
# RTX 5060: pick the build matching your installed CUDA runtime.
pip install onnxruntime-gpu
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

You want `CUDAExecutionProvider` in that list. Then place the two ONNX files
(`yolox_l.onnx` and `dw-ll_ucoco_384.onnx`) somewhere stable such as
`C:\models\dwpose\`, copy `config\local.example.yaml` to
`config\local.yaml` and point it at them.

Nothing is downloaded by the application. Full instructions, the smoke test and
the tuning guide are in [`dwpose-setup.md`](dwpose-setup.md).

## 7. ComfyUI (optional)

Only needed when you have a real workflow to run. Install it yourself — this
application never downloads anything.

```powershell
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Bind to loopback only. The backend refuses non-local endpoints by default.
python main.py --listen 127.0.0.1 --port 8188
```

Then confirm the application can see it:

```powershell
app job backends
```

For an 8GB card, start ComfyUI with low-VRAM flags — see
[`troubleshooting-8gb-vram.md`](troubleshooting-8gb-vram.md).

## Windows-specific notes

**Long paths.** Frame directories nest deeply
(`data\jobs\<job_id>\composited_frames\frame_000123.png`). Enable long paths
once:

```powershell
# Administrator PowerShell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Keeping the repository near the drive root (`C:\work\garment-replacer`) also
helps.

**Data directory location.** `data/` holds thousands of PNG frames. Put it on a
fast drive with room to spare — roughly 3–6 MB per 1080×1920 frame, so a
600-frame master plus a few jobs is tens of gigabytes. Point elsewhere with
either:

```powershell
app --data-root D:\garment-data doctor
```

or `config/local.yaml` (gitignored):

```yaml
paths:
  data_root: D:/garment-data
```

**Hardlinks.** Assembly links frames instead of copying them when the
filesystem allows it, which NTFS does within a single volume. Across volumes it
falls back to a byte copy — correct either way, just more disk.

**Antivirus.** Real-time scanning of a directory receiving thousands of PNG
writes is measurably slow. Consider excluding your `data/` directory.

**Terminal.** Windows Terminal renders the CLI's progress bars and colours
correctly; `cmd.exe` is functional but plainer.

**Path separators.** Use forward slashes or doubled backslashes in YAML;
`D:\data` in unquoted YAML is a parse hazard. `D:/data` always works.

## Where things land

| Path | Contents |
| --- | --- |
| `data\templates\<id>\` | Immutable source frames, masks, analysis data |
| `data\garments\<id>\` | Garment reference images |
| `data\jobs\<id>\` | Rendered frames, manifest, QC reports, contact sheets |
| `data\motion_sources\<id>\pose\` | Pose JSON for a motion reference (no imagery) |
| `data\compositions\<id>\` | Normalized/bridge/composed poses, preview, manifest |
| `data\heroes\<id>\images\` | Hero Character reference images |
| `data\masters\<id>\frames\` | Candidate master frames |
| `data\exports\` | Final `.mp4` files |
| `data\logs\app.jsonl` | Structured JSON logs (rotating, 16 MB × 5) |
| `data\db\app.db` | SQLite catalogue (WAL mode) |

Nothing here is ever committed to Git — `data/.gitignore` excludes all of it.
