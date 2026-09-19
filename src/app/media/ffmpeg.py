"""Thin, explicit subprocess wrappers around ffmpeg and ffprobe.

Design rules:

* every command is built as an argv list (never a shell string),
* every command is returned to the caller so it can be recorded in the
  manifest verbatim,
* nothing is inferred about the input: frame rate, frame count and duration all
  come from ffprobe,
* no network protocols are used; inputs and outputs are local files only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import MediaToolError
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 3600.0

#: Protocol prefixes that must never appear in an ffmpeg input/output path.
_REMOTE_PREFIXES = (
    "http://",
    "https://",
    "rtmp://",
    "rtsp://",
    "udp://",
    "tcp://",
    "ftp://",
    "srt://",
    "sftp://",
    "pipe:",
    "concat:",
    "data:",
)


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class StreamInfo:
    """Normalised subset of an ffprobe stream entry."""

    index: int
    codec_type: str
    codec_name: str | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    avg_frame_rate: str | None = None
    r_frame_rate: str | None = None
    nb_frames: int | None = None
    duration_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    format_name: str | None
    duration_s: float | None
    bit_rate: int | None
    streams: list[StreamInfo]
    raw: dict[str, Any]

    @property
    def video(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "video"), None)

    @property
    def audio(self) -> StreamInfo | None:
        return next((s for s in self.streams if s.codec_type == "audio"), None)

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def assert_local_path(path: str | Path, *, what: str = "path") -> Path:
    """Reject anything that looks like a remote or pipe-based ffmpeg input."""
    raw = str(path)
    lowered = raw.lower()
    for prefix in _REMOTE_PREFIXES:
        if lowered.startswith(prefix):
            raise MediaToolError(
                f"Non-local {what} refused: ffmpeg I/O must be a local file",
                value=raw,
                prefix=prefix,
            )
    return Path(raw)


def resolve_binary(binary: str) -> str:
    found = shutil.which(binary)
    if found is None:
        raise MediaToolError(
            f"Required executable not found on PATH: {binary}",
            binary=binary,
            hint="Install FFmpeg and ensure ffmpeg/ffprobe are on PATH.",
        )
    return found


def tool_available(binary: str) -> bool:
    return shutil.which(binary) is not None


def run_command(
    argv: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    check: bool = True,
) -> CommandResult:
    """Run a media tool, returning stdout/stderr and the exact argv used."""
    resolved = [resolve_binary(argv[0]), *argv[1:]]
    logger.debug("media_command", extra={"argv": resolved})
    try:
        completed = subprocess.run(
            resolved,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(
            "Media command timed out",
            argv=resolved,
            timeout_s=timeout_s,
        ) from exc
    except OSError as exc:
        raise MediaToolError("Failed to execute media command", argv=resolved) from exc

    result = CommandResult(
        argv=resolved,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and not result.ok:
        raise MediaToolError(
            f"{Path(resolved[0]).name} exited with code {result.returncode}",
            argv=resolved,
            stderr=result.stderr[-4000:],
        )
    return result


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------
def probe(
    path: str | Path,
    *,
    ffprobe: str = "ffprobe",
    count_frames: bool = False,
    timeout_s: float = 600.0,
) -> ProbeResult:
    """Probe a media file. ``count_frames`` does an exact (slower) frame count."""
    target = assert_local_path(path, what="input")
    if not target.is_file():
        raise MediaToolError("Input file does not exist", path=str(target))

    argv = [
        ffprobe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
    ]
    if count_frames:
        argv += ["-count_frames"]
    argv += [str(target)]

    result = run_command(argv, timeout_s=timeout_s)
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaToolError("ffprobe returned invalid JSON", path=str(target)) from exc

    streams: list[StreamInfo] = []
    for entry in data.get("streams", []):
        nb_frames = entry.get("nb_read_frames") or entry.get("nb_frames")
        streams.append(
            StreamInfo(
                index=int(entry.get("index", len(streams))),
                codec_type=str(entry.get("codec_type", "unknown")),
                codec_name=entry.get("codec_name"),
                width=_maybe_int(entry.get("width")),
                height=_maybe_int(entry.get("height")),
                pix_fmt=entry.get("pix_fmt"),
                avg_frame_rate=entry.get("avg_frame_rate"),
                r_frame_rate=entry.get("r_frame_rate"),
                nb_frames=_maybe_int(nb_frames),
                duration_s=_maybe_float(entry.get("duration")),
                raw=entry,
            )
        )
    fmt = data.get("format", {})
    return ProbeResult(
        format_name=fmt.get("format_name"),
        duration_s=_maybe_float(fmt.get("duration")),
        bit_rate=_maybe_int(fmt.get("bit_rate")),
        streams=streams,
        raw=data,
    )


def parse_frame_rate(value: str | None) -> float | None:
    """Parse an ffprobe rational frame rate such as ``"30000/1001"``."""
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(value)
    except ValueError:
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# ffmpeg command builders (returned, not just executed, for the manifest)
# ---------------------------------------------------------------------------
def build_extract_frames_command(
    source: str | Path,
    out_pattern: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    start_frame: int | None = None,
    end_frame: int | None = None,
    fps: float | None = None,
) -> list[str]:
    """Extract lossless PNG frames.

    Frame selection uses ``select=between(n,start,end)`` on decoded frame
    numbers, which is exact, rather than a timestamp seek which can drift.
    """
    src = assert_local_path(source, what="input")
    dst = assert_local_path(out_pattern, what="output")
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src)]

    filters: list[str] = []
    if start_frame is not None or end_frame is not None:
        lo = 0 if start_frame is None else int(start_frame)
        hi = "9999999" if end_frame is None else str(int(end_frame) - 1)
        filters.append(f"select=between(n\\,{lo}\\,{hi})")
    if fps is not None:
        filters.append(f"fps={fps}")
    if filters:
        argv += ["-vf", ",".join(filters), "-fps_mode", "passthrough"]
    argv += [
        "-pix_fmt",
        "rgb24",
        "-compression_level",
        "1",
        "-start_number",
        str(start_frame or 0),
        str(dst),
    ]
    return argv


def build_encode_from_frames_command(
    frame_pattern: str | Path,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    fps: float,
    width: int,
    height: int,
    pixel_format: str = "yuv420p",
    codec: str = "libx264",
    crf: int = 17,
    preset: str = "slow",
    start_number: int = 0,
    audio_source: str | Path | None = None,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    faststart: bool = True,
    shortest: bool = True,
) -> list[str]:
    """Encode a PNG sequence to H.264, optionally copying audio from a master."""
    pattern = assert_local_path(frame_pattern, what="input")
    out = assert_local_path(output, what="output")
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-framerate",
        _fps_arg(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(pattern),
    ]
    if audio_source is not None:
        argv += ["-i", str(assert_local_path(audio_source, what="audio input"))]
    argv += [
        "-vf",
        f"scale={width}:{height}:flags=lanczos,format={pixel_format}",
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        pixel_format,
        "-r",
        _fps_arg(fps),
        "-fps_mode",
        "cfr",
    ]
    if audio_source is not None:
        argv += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", audio_codec, "-b:a", audio_bitrate]
        if shortest:
            argv += ["-shortest"]
    else:
        argv += ["-an"]
    if faststart:
        argv += ["-movflags", "+faststart"]
    argv += [str(out)]
    return argv


def build_extract_audio_command(
    source: str | Path,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    start_s: float | None = None,
    duration_s: float | None = None,
) -> list[str]:
    """Copy an audio stream out of the master video without re-encoding."""
    src = assert_local_path(source, what="input")
    out = assert_local_path(output, what="output")
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if start_s is not None:
        argv += ["-ss", f"{start_s:.6f}"]
    argv += ["-i", str(src)]
    if duration_s is not None:
        argv += ["-t", f"{duration_s:.6f}"]
    argv += ["-vn", "-acodec", "copy", str(out)]
    return argv


def build_preview_command(
    source: str | Path,
    output: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    scale: float = 0.5,
    crf: int = 28,
    width: int,
    height: int,
) -> list[str]:
    src = assert_local_path(source, what="input")
    out = assert_local_path(output, what="output")
    pw = _even(int(width * scale))
    ph = _even(int(height * scale))
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"scale={pw}:{ph}:flags=bilinear",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out),
    ]


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _fps_arg(fps: float) -> str:
    """Render an fps value losslessly where possible (29.97 -> 30000/1001)."""
    for numerator, denominator in ((30000, 1001), (24000, 1001), (60000, 1001)):
        if abs(fps - numerator / denominator) < 1e-6:
            return f"{numerator}/{denominator}"
    if abs(fps - round(fps)) < 1e-9:
        return str(round(fps))
    return f"{fps:.6f}"


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "CommandResult",
    "ProbeResult",
    "StreamInfo",
    "assert_local_path",
    "build_encode_from_frames_command",
    "build_extract_audio_command",
    "build_extract_frames_command",
    "build_preview_command",
    "parse_frame_rate",
    "probe",
    "resolve_binary",
    "run_command",
    "tool_available",
]
