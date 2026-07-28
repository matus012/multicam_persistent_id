"""`mcreid-sync` — align independently-recorded camera clips from a clap.

Four phones started by hand are offset by seconds. The fusion stage assumes
frames handed to `step()` are simultaneous, and tolerates <= 50 ms of skew
(~1.5 frames at 30 fps), so the offsets have to be measured and removed before
anything else runs.

The clap gives two independent cues; this tool uses the audio one because it is
far sharper than any visual cue at 30 fps. Each clip's audio is extracted, the
onset of the loudest transient is located, and offsets are reported relative to
the earliest clip. Writes `sync.json` consumed by `mcreid-demo recorded`.

Audio extraction shells out to ffmpeg (already required for video export).
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer

from mcreid.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Align multi-camera clips via a clap transient.")

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")


@dataclass
class ClapOffsets:
    """Per-camera temporal offsets relative to the earliest clip."""

    reference: str
    offset_s: dict[str, float]
    clap_time_s: dict[str, float]
    sample_rate: int
    max_skew_ms: float = field(init=False)

    def __post_init__(self) -> None:
        spread = max(self.offset_s.values()) - min(self.offset_s.values())
        self.max_skew_ms = spread * 1000.0

    def frame_offset(self, camera_id: str, fps: float) -> int:
        """Whole-frame shift to apply to ``camera_id`` to align it to the reference."""
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        return int(round(self.offset_s[camera_id] * fps))

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "reference": self.reference,
                    "offset_s": self.offset_s,
                    "clap_time_s": self.clap_time_s,
                    "sample_rate": self.sample_rate,
                    "max_skew_ms": self.max_skew_ms,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Extract mono PCM audio with ffmpeg. Fails fast if ffmpeg is missing."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", str(out_wav),
    ]  # fmt: skip
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg) — "
            "clap sync needs it to read the audio track."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed on {video.name}: {exc.stderr.decode('utf-8', 'replace')[:400]}"
        ) from exc
    return out_wav


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono WAV into a float array in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return samples, rate


def find_clap(
    samples: np.ndarray, sample_rate: int, window_ms: float = 5.0, threshold: float = 0.55
) -> float:
    """Return the time in seconds of the first sharp transient.

    A clap is a near-instant jump in short-term energy. The onset — the first
    sample crossing ``threshold`` of the peak envelope — is used rather than the
    peak itself, because reverberation moves the peak by tens of milliseconds
    while the onset stays put.
    """
    if samples.size == 0:
        raise ValueError("empty audio")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    window = max(int(sample_rate * window_ms / 1000.0), 1)
    energy = np.convolve(samples**2, np.ones(window) / window, mode="same")
    peak = float(energy.max())
    if peak <= 0.0:
        raise ValueError("audio is silent — no clap to find")

    crossings = np.flatnonzero(energy >= threshold * peak)
    return float(crossings[0]) / sample_rate


@app.command()
def measure(
    footage: Path = typer.Option(..., help="Directory holding one video per camera."),
    out: Path = typer.Option(Path("calib/sync.json"), help="Where to write the offsets."),
    sample_rate: int = typer.Option(16000, help="Audio resample rate, Hz."),
    max_skew_ms: float = typer.Option(50.0, help="Fail if residual skew exceeds this."),
    work_dir: Path = typer.Option(Path("outputs/sync"), help="Scratch dir for extracted audio."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Measure per-camera offsets from the clap and write sync.json."""
    setup_logging(log_level)
    if not footage.is_dir():
        raise typer.BadParameter(f"footage directory not found: {footage}")

    videos = sorted(p for p in footage.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if len(videos) < 2:
        raise typer.BadParameter(f"need >= 2 clips in {footage}, found {len(videos)}")

    clap_time: dict[str, float] = {}
    for video in videos:
        wav = extract_audio(video, work_dir / f"{video.stem}.wav", sample_rate)
        samples, rate = read_wav_mono(wav)
        clap_time[video.stem] = find_clap(samples, rate)
        typer.echo(f"  {video.stem}: clap at {clap_time[video.stem]:.3f} s")

    reference = min(clap_time, key=lambda k: clap_time[k])
    base = clap_time[reference]
    offsets = ClapOffsets(
        reference=reference,
        offset_s={k: v - base for k, v in clap_time.items()},
        clap_time_s=clap_time,
        sample_rate=sample_rate,
    )

    path = offsets.to_json(out)
    typer.echo(f"\nreference camera: {reference}")
    for camera_id, offset in sorted(offsets.offset_s.items()):
        typer.echo(f"  {camera_id}: +{offset * 1000:8.1f} ms")
    typer.echo(f"wrote {path}")

    # The spread is what has to be *removed*; residual skew after applying whole
    # -frame shifts is what actually matters, so report both.
    typer.echo(f"raw spread: {offsets.max_skew_ms:.1f} ms")
    residual = max(
        abs(offsets.offset_s[c] * 1000 - round(offsets.offset_s[c] * 30) / 30 * 1000)
        for c in offsets.offset_s
    )
    typer.echo(f"residual skew after whole-frame alignment at 30 fps: {residual:.1f} ms")
    if residual > max_skew_ms:
        typer.echo(
            f"\nWARNING: residual skew {residual:.1f} ms exceeds the {max_skew_ms:.0f} ms budget. "
            "Re-shoot the clap, or record at a higher frame rate."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
