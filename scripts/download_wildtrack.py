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

_INSTRUCTIONS = f"""
WILDTRACK is distributed by EPFL CVLab and requires a manual request/consent
step before download. This tool will not scrape, guess, or bypass that step.

  1. Visit {DATASET_HOMEPAGE}
  2. Follow EPFL's request/download instructions there (they may ask you to
     agree to dataset terms and/or fill in a request form).
  3. Save the downloaded archive somewhere on this machine.
  4. Run:
       python scripts/download_wildtrack.py install --archive <path-to-archive>

     Add --expected-sha256 <hash> once you know the hash you want to pin
     (the first run without it will print the observed hash for you to save).

This command does not download anything on your behalf.
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


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_zip_members(zf, dest)
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
