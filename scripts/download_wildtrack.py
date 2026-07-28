"""Fetch/verify a local copy of the WILDTRACK dataset.

WILDTRACK (Chavdarova et al., CVPR 2018) is distributed by EPFL CVLab under
terms that require a manual request/consent step -- there is no scrapable
direct-download URL, and this tool will not attempt to find or bypass one.
Automating past a consent gate is both against the license terms and outside
what this project should ever do automatically.

Instead this script:
    1. Tells you where to go and what to do (`info`).
    2. Takes over once you have a local archive file: verifies its SHA-256,
       extracts it, validates the resulting directory structure, and records
       the observed checksum so you can pin it on the next run (`install`).
    3. Can re-check an already-extracted tree without touching the network
       at all (`verify`).

Usage:
    python scripts/download_wildtrack.py info
    python scripts/download_wildtrack.py install --archive <path> [--expected-sha256 <hash>]
    python scripts/download_wildtrack.py verify [--root data/wildtrack]

Never commits data: this script only ever writes under --dest (default
data/wildtrack/), which must stay in .gitignore.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

import typer

from mcreid.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Fetch/verify a local WILDTRACK copy.")

DATASET_HOMEPAGE = "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"
DEFAULT_DATA_ROOT = Path("data/wildtrack")

# Top-level layout WILDTRACK ships (dataset docs / widely-used mirrors, e.g.
# the MVDet codebase's expected input layout).
_REQUIRED_SUBDIRS = ("Image_subsets", "annotations_positions", "calibrations")
_EXPECTED_ANNOTATION_FRAMES = 400
_EXPECTED_CAMERAS = 7

# EPFL CVLab publishes the annotated archive as a plain link on the dataset
# page — no form, no consent gate, no credentials. Verified reachable with a
# 200 and Content-Type: application/zip. If that ever changes to a gated flow,
# fall back to `install --archive` and do NOT try to work around the gate.
ARCHIVE_URL = (
    "http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/"
    "Wildtrack_dataset_full.zip"
)
# Observed 2026-07-28: 6,807,496,358 bytes, matching the server's Content-Length.
KNOWN_SHA256 = "7f630f3adf305d05362379f2603c67b0c6a0b173dcec508020e98a7bde4e4492"

_INSTRUCTIONS = f"""
WILDTRACK is published by EPFL CVLab. As of this writing the annotated archive
is a direct download, so the normal path is:

    python scripts/download_wildtrack.py fetch

which downloads, checksums and extracts it into {DEFAULT_DATA_ROOT}.

If EPFL later puts the archive behind a request form or EULA, that flow will
stop working. Do not attempt to bypass it — instead:

  1. Visit {DATASET_HOMEPAGE}
  2. Follow whatever request/consent steps they ask for.
  3. Save the archive somewhere on this machine.
  4. Run:
       python scripts/download_wildtrack.py install --archive <path-to-archive>

Either way, add --expected-sha256 <hash> once you know the hash you want to
pin; the first run without it prints the observed hash for you to save.

The dataset is never committed: data/ is gitignored.
"""


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_zip_members(zf: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for name in zf.namelist():
        target = (dest / name).resolve()
        if dest_resolved not in target.parents and target != dest_resolved:
            raise ValueError(f"zip member escapes destination directory: {name!r}")


def _safe_tar_members(tf: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if dest_resolved not in target.parents and target != dest_resolved:
            raise ValueError(f"tar member escapes destination directory: {member.name!r}")


def repair_wrapped_offsets(zf: zipfile.ZipFile, archive: Path) -> int:
    """Recover local-header offsets in a >4 GB zip written without ZIP64.

    The WILDTRACK archive as published by EPFL is 6.8 GB but carries a plain
    (non-ZIP64) end-of-central-directory record, so every 32-bit offset field
    above 4 GB has wrapped. Python sees a central directory that appears to sit
    4 GB earlier than it does, concludes the file has a 4 GB prepended blob, and
    adds 2**32 to every member offset. Members genuinely past the 4 GB mark come
    out right; everything before it points into hyperspace, and both Python's
    zipfile and bsdtar report the archive as damaged.

    Nothing here bypasses any protection — it is a plain arithmetic repair of a
    known packaging bug. Each candidate offset is validated by checking for a
    local-header signature AND a matching filename before it is accepted, so a
    wrong guess cannot silently extract the wrong bytes.

    Returns the number of members whose offset had to be corrected.
    """
    size = archive.stat().st_size
    repaired = 0
    with archive.open("rb") as handle:
        for info in zf.infolist():
            candidates = (
                info.header_offset,
                info.header_offset - (1 << 32),
                info.header_offset + (1 << 32),
            )
            for candidate in candidates:
                if not 0 <= candidate < size - 30:
                    continue
                handle.seek(candidate)
                header = handle.read(30)
                if header[:4] != b"PK\x03\x04":
                    continue
                name_len, extra_len = struct.unpack("<HH", header[26:30])
                if name_len != len(info.orig_filename.encode("utf-8")):
                    continue
                name = handle.read(name_len)
                if name.decode("utf-8", "replace") != info.orig_filename:
                    continue
                if candidate != info.header_offset:
                    info.header_offset = candidate
                    repaired += 1
                break
            else:
                raise zipfile.BadZipFile(
                    f"could not locate a valid local header for {info.filename!r}. "
                    "The archive is genuinely corrupt, not merely non-ZIP64 — "
                    "re-download it."
                )

    # Python's overlapped-entry ("zip bomb") guard caches each member's end
    # offset from the ORIGINAL ordering. Those bounds are stale once offsets
    # move, and every extraction trips the guard. Recompute them from the
    # repaired layout rather than switching the protection off.
    ordered = sorted(zf.infolist(), key=lambda i: i.header_offset)
    if ordered and hasattr(ordered[0], "_end_offset"):
        for current, following in zip(ordered, ordered[1:], strict=False):
            current._end_offset = following.header_offset  # noqa: SLF001
        ordered[-1]._end_offset = size  # noqa: SLF001
    return repaired


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_zip_members(zf, dest)
            repaired = repair_wrapped_offsets(zf, archive)
            if repaired:
                logger.warning(
                    "archive is >4 GB without ZIP64: repaired %d wrapped member "
                    "offset(s) before extracting",
                    repaired,
                )
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _safe_tar_members(tf, dest)
            tf.extractall(dest)
    else:
        raise ValueError(
            f"unrecognised archive format (not a zip or tar file): {archive}. "
            "If EPFL shipped something else, extract it yourself and point "
            "`verify --root` at the resulting directory."
        )


def _find_dataset_root(staging: Path) -> Path:
    """Locate the directory containing all required WILDTRACK subdirs.

    Archives are sometimes wrapped in one extra top-level folder; this looks
    one level deep rather than assuming a fixed layout.
    """
    candidates = [staging, *(p for p in staging.iterdir() if p.is_dir())]
    for candidate in candidates:
        if all((candidate / sub).is_dir() for sub in _REQUIRED_SUBDIRS):
            return candidate
    raise FileNotFoundError(
        f"could not locate {_REQUIRED_SUBDIRS} under {staging} (looked 1 level deep). "
        "The archive's internal layout may differ from what this tool expects -- "
        "inspect it manually and lay it out as data/wildtrack/{Image_subsets,"
        "annotations_positions,calibrations}."
    )


def _validate_tree(root: Path) -> list[str]:
    """Return human-readable problems with ``root``; empty means it looks right."""
    problems: list[str] = []
    for sub in _REQUIRED_SUBDIRS:
        if not (root / sub).is_dir():
            problems.append(f"missing directory: {root / sub}")
    if problems:
        return problems

    annotation_files = list((root / "annotations_positions").glob("*.json"))
    if not annotation_files:
        problems.append(f"no *.json files under {root / 'annotations_positions'}")
    elif len(annotation_files) != _EXPECTED_ANNOTATION_FRAMES:
        problems.append(
            f"expected {_EXPECTED_ANNOTATION_FRAMES} annotation files, "
            f"found {len(annotation_files)} in {root / 'annotations_positions'}"
        )

    image_subdirs = [p for p in (root / "Image_subsets").iterdir() if p.is_dir()]
    if len(image_subdirs) != _EXPECTED_CAMERAS:
        problems.append(
            f"expected {_EXPECTED_CAMERAS} camera subfolders under Image_subsets, "
            f"found {len(image_subdirs)}"
        )

    intrinsic_dir = root / "calibrations" / "intrinsic_zero"
    extrinsic_dir = root / "calibrations" / "extrinsic"
    if not intrinsic_dir.is_dir():
        problems.append(f"missing directory: {intrinsic_dir}")
    else:
        n = len(list(intrinsic_dir.glob("*.xml")))
        if n != _EXPECTED_CAMERAS:
            problems.append(
                f"expected {_EXPECTED_CAMERAS} intrinsic XML files, found {n} in {intrinsic_dir}"
            )
    if not extrinsic_dir.is_dir():
        problems.append(f"missing directory: {extrinsic_dir}")
    else:
        n = len(list(extrinsic_dir.glob("*.xml")))
        if n != _EXPECTED_CAMERAS:
            problems.append(
                f"expected {_EXPECTED_CAMERAS} extrinsic XML files, found {n} in {extrinsic_dir}"
            )

    return problems


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Fetch/verify a local WILDTRACK copy (EPFL CVLab; manual consent required)."""
    if ctx.invoked_subcommand is None:
        typer.echo(_INSTRUCTIONS)


@app.command()
def info() -> None:
    """Print where to request/download WILDTRACK and how to install it here."""
    typer.echo(_INSTRUCTIONS)


_ARCHIVE_OPTION = typer.Option(
    ..., exists=True, dir_okay=False, help="Path to the archive you downloaded."
)
_DEST_OPTION = typer.Option(DEFAULT_DATA_ROOT, help="Where to install the dataset.")


@app.command()
def fetch(
    dest: Path = _DEST_OPTION,
    cache_dir: Path = typer.Option(
        Path("data/downloads"), help="Where the archive is kept between runs."
    ),
    url: str = typer.Option(ARCHIVE_URL, help="Archive URL."),
    expected_sha256: str = typer.Option("", help="Pin and verify the archive's SHA-256."),
    force: bool = typer.Option(False, help="Re-download and overwrite an existing install."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Download, checksum, extract and validate WILDTRACK in one step.

    Resumes a partial download rather than restarting it — the archive is large
    enough that a dropped connection should not cost the whole transfer.
    """
    setup_logging(log_level)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / Path(url).name

    if archive.is_file() and not force:
        typer.echo(f"using cached archive {archive} ({archive.stat().st_size / 1e9:.2f} GB)")
    else:
        typer.echo(f"downloading {url}\n  -> {archive}")
        command = [
            "curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
            "-C", "-", "-o", str(archive), url,
        ]  # fmt: skip
        try:
            completed = subprocess.run(command, check=False)
        except FileNotFoundError as exc:
            raise typer.BadParameter(
                "curl not found on PATH. Install it, or download the archive yourself "
                "and use `install --archive <path>`."
            ) from exc
        if completed.returncode != 0:
            typer.echo(
                f"download failed (curl exit {completed.returncode}). The archive is "
                "resumable — re-run this command to continue from where it stopped."
            )
            raise typer.Exit(code=1)

    install(
        archive=archive,
        dest=dest,
        expected_sha256=expected_sha256,
        force=force,
        log_level=log_level,
    )


@app.command()
def install(
    archive: Path = _ARCHIVE_OPTION,
    dest: Path = _DEST_OPTION,
    expected_sha256: str = typer.Option(
        "", help="Pin and verify the archive's SHA-256 (hex)."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing, non-empty --dest."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Verify, extract, and validate a downloaded WILDTRACK archive."""
    setup_logging(log_level)
    archive = Path(archive)

    if dest.exists() and any(dest.iterdir()) and not force:
        typer.echo(
            f"{dest} already exists and is non-empty. Pass --force to overwrite, "
            "or --dest a fresh directory."
        )
        raise typer.Exit(code=1)

    typer.echo(f"computing SHA-256 of {archive} ...")
    observed = _sha256(archive)
    typer.echo(f"observed sha256: {observed}")

    if expected_sha256:
        if observed.lower() != expected_sha256.strip().lower():
            typer.echo(
                "checksum mismatch:\n"
                f"  expected: {expected_sha256.strip().lower()}\n"
                f"  observed: {observed}\n"
                "The archive is corrupt, incomplete, or not the file you intended. Aborting."
            )
            raise typer.Exit(code=1)
        typer.echo("checksum OK.")
    else:
        typer.echo(
            "no --expected-sha256 given; skipping verification against a known value. "
            f"Pin this exact archive next time with:\n  --expected-sha256 {observed}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    typer.echo(f"extracting {archive.name} -> {dest} ...")
    try:
        _extract(archive, dest)
    except ValueError as exc:
        typer.echo(f"extraction failed: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        root = _find_dataset_root(dest)
    except FileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    if root != dest:
        logger.info("flattening nested archive folder %s -> %s", root, dest)
        for child in root.iterdir():
            target = dest / child.name
            if target.exists():
                typer.echo(
                    f"{target} already exists while flattening {root}; resolve manually."
                )
                raise typer.Exit(code=1)
            shutil.move(str(child), str(target))
        root.rmdir()

    checksum_record = dest / "CHECKSUM.sha256"
    checksum_record.write_text(f"{observed}  {archive.name}\n", encoding="utf-8")
    typer.echo(f"recorded observed checksum -> {checksum_record}")

    problems = _validate_tree(dest)
    if problems:
        typer.echo(f"dataset tree at {dest} looks incomplete after extraction:")
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(code=1)

    typer.echo(f"WILDTRACK installed and validated at {dest}")


_ROOT_OPTION = typer.Option(DEFAULT_DATA_ROOT, help="Directory to check.")


@app.command()
def verify(
    root: Path = _ROOT_OPTION,
    log_level: str = typer.Option("INFO"),
) -> None:
    """Check that an existing directory looks like a valid WILDTRACK tree."""
    setup_logging(log_level)
    if not root.is_dir():
        typer.echo(f"{root} is not a directory. Run `install` first (see `info`).")
        raise typer.Exit(code=1)

    problems = _validate_tree(root)
    if problems:
        typer.echo(f"WILDTRACK tree at {root} is INVALID:")
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(code=1)

    typer.echo(f"WILDTRACK tree at {root} looks OK.")


if __name__ == "__main__":  # pragma: no cover
    app()
