"""Per-camera intrinsics from checkerboard views (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import Intrinsics
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# Sub-pixel corner refinement — standard OpenCV settings.
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_WINDOW = (11, 11)


@dataclass(frozen=True)
class CheckerboardSpec:
    """Physical description of the calibration target.

    ``inner_corners`` is (cols, rows) of *interior* corners — a board with 10x7
    squares has 9x6 interior corners. Getting this wrong is the single most
    common calibration failure, so it is validated loudly.
    """

    inner_corners: tuple[int, int]
    square_size_m: float

    def __post_init__(self) -> None:
        cols, rows = self.inner_corners
        if cols < 3 or rows < 3:
            raise ValueError(f"inner_corners must both be >= 3, got {self.inner_corners}")
        if cols == rows:
            raise ValueError(
                f"square board {self.inner_corners} is rotationally ambiguous — "
                "use a non-square pattern (e.g. 9x6)"
            )
        if self.square_size_m <= 0.0:
            raise ValueError(f"square_size_m must be positive, got {self.square_size_m}")

    def object_points(self) -> FloatArray:
        """(N, 3) board-frame corner coordinates in metres, Z=0."""
        cols, rows = self.inner_corners
        grid = np.zeros((cols * rows, 3), dtype=np.float64)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return grid * self.square_size_m


def find_checkerboard(
    image: npt.NDArray[np.uint8], spec: CheckerboardSpec
) -> FloatArray | None:
    """Detect + sub-pixel refine checkerboard corners. Returns (N, 2) or None."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {image.shape}")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, spec.inner_corners, flags=flags)
    if not found:
        return None
    refined = cv2.cornerSubPix(gray, corners, _WINDOW, (-1, -1), _CRITERIA)
    return np.asarray(refined, dtype=np.float64).reshape(-1, 2)


def list_images(directory: Path) -> list[Path]:
    """Sorted image files in ``directory``. Fails fast if empty."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images ({', '.join(IMAGE_SUFFIXES)}) in {directory}")
    return files


def calibrate_intrinsics_from_corners(
    corner_sets: list[FloatArray],
    spec: CheckerboardSpec,
    image_size: tuple[int, int],
    min_views: int = 8,
) -> Intrinsics:
    """Run ``cv2.calibrateCamera`` on already-detected corner sets.

    Split out from the image-loading path so tests can drive it with synthetic
    projections and no files on disk.
    """
    if len(corner_sets) < min_views:
        raise ValueError(
            f"need >= {min_views} checkerboard views for a stable calibration, "
            f"got {len(corner_sets)}. Re-shoot with more angles/tilts."
        )
    expected = spec.inner_corners[0] * spec.inner_corners[1]
    for i, corners in enumerate(corner_sets):
        if corners.shape != (expected, 2):
            raise ValueError(f"view {i}: expected ({expected}, 2) corners, got {corners.shape}")

    obj = spec.object_points().astype(np.float32)
    object_points = [obj for _ in corner_sets]
    image_points = [c.astype(np.float32).reshape(-1, 1, 2) for c in corner_sets]

    # cv2's type stubs do not model the "pass None to let OpenCV allocate"
    # overload, which is the documented way to calibrate from scratch.
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(  # type: ignore[call-overload]
        object_points, image_points, image_size, None, None
    )
    logger.info(
        "intrinsics: %d views, RMS reproj = %.4f px, fx=%.1f fy=%.1f",
        len(corner_sets),
        rms,
        K[0, 0],
        K[1, 1],
    )
    if rms > 1.5:
        logger.warning(
            "RMS reprojection %.3f px is high (>1.5). Blurry frames or a wrong "
            "square_size_m/inner_corners spec are the usual causes.",
            rms,
        )
    return Intrinsics.from_matrices(
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
        image_size=image_size,
        rms_reproj_px=float(rms),
        n_views=len(corner_sets),
    )


def sharpness(image: npt.NDArray[np.uint8]) -> float:
    """Variance of the Laplacian — a cheap, reliable blur score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def collect_corners_from_video(
    video: Path,
    spec: CheckerboardSpec,
    sample_every: int = 10,
    max_views: int = 40,
    min_sharpness: float = 40.0,
) -> tuple[list[FloatArray], tuple[int, int]]:
    """Detect the checkerboard across a calibration video.

    Phones shoot video, not stills, so this is the path the capture guide
    actually produces. Frames are subsampled (consecutive frames of a slowly
    waved board are near-duplicates and add nothing but runtime), blurry ones are
    dropped, and the sharpest ``max_views`` detections are kept.

    Returns (corner_sets, image_size).
    """
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if not video.is_file():
        raise FileNotFoundError(f"calibration video not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video}")

    scored: list[tuple[float, FloatArray]] = []
    image_size: tuple[int, int] | None = None
    index = 0
    n_blurry = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every != 0:
                index += 1
                continue
            index += 1

            image = np.asarray(frame, dtype=np.uint8)
            size = (image.shape[1], image.shape[0])
            if image_size is None:
                image_size = size
            elif size != image_size:
                raise ValueError(f"{video.name}: frame size changed mid-clip ({size} vs {image_size})")

            score = sharpness(image)
            if score < min_sharpness:
                n_blurry += 1
                continue
            corners = find_checkerboard(image, spec)
            if corners is not None:
                scored.append((score, corners))
    finally:
        capture.release()

    if image_size is None:
        raise OSError(f"{video}: no readable frames")

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = [corners for _score, corners in scored[:max_views]]
    logger.info(
        "%s: sampled %d frames, %d too blurry, board found in %d, kept %d",
        video.name,
        (index + sample_every - 1) // sample_every,
        n_blurry,
        len(scored),
        len(kept),
    )
    return kept, image_size


def calibrate_intrinsics_from_video(
    video: Path,
    spec: CheckerboardSpec,
    min_views: int = 8,
    sample_every: int = 10,
    max_views: int = 40,
) -> Intrinsics:
    """Detect the board across a video and calibrate."""
    corner_sets, image_size = collect_corners_from_video(
        video, spec, sample_every=sample_every, max_views=max_views
    )
    return calibrate_intrinsics_from_corners(corner_sets, spec, image_size, min_views=min_views)


def calibrate_intrinsics_from_dir(
    directory: Path,
    spec: CheckerboardSpec,
    min_views: int = 8,
) -> Intrinsics:
    """Detect the board in every image in ``directory`` and calibrate."""
    files = list_images(directory)
    corner_sets: list[FloatArray] = []
    image_size: tuple[int, int] | None = None

    for path in files:
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise OSError(f"could not read image: {path}")
        image: npt.NDArray[np.uint8] = np.asarray(raw, dtype=np.uint8)
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(
                f"{path.name} is {size} but earlier frames are {image_size}; "
                "all calibration frames must come from one camera at one resolution"
            )
        corners = find_checkerboard(image, spec)
        if corners is None:
            logger.debug("no board found in %s", path.name)
            continue
        corner_sets.append(corners)

    assert image_size is not None  # list_images guarantees >= 1 file
    logger.info("board detected in %d/%d frames", len(corner_sets), len(files))
    return calibrate_intrinsics_from_corners(corner_sets, spec, image_size, min_views=min_views)
