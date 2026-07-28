"""WILDTRACK dataset support: grid geometry, calibration/annotation loaders,
and multi-view detection (MODA/MODP) scoring.

WILDTRACK (Chavdarova et al., "WILDTRACK: A Multi-camera HD Dataset for Dense
Unscripted Pedestrian Detection", CVPR 2018) ships:
    - 7 synchronised static HD cameras, 1920x1080, overlapping views of a
      public square (``N_CAMERAS``, ``IMAGE_WIDTH``, ``IMAGE_HEIGHT`` below).
    - 400 annotated frames, described by the dataset as sampled at 2 fps from
      60 fps video (``N_ANNOTATED_FRAMES``, ``FRAME_SAMPLING_HZ``).
    - A ground plane discretised into a 480x1440 grid of 2.5 cm cells,
      covering a 12 m x 36 m area (``GRID_WIDTH_CELLS``, ``GRID_HEIGHT_CELLS``,
      ``CELL_SIZE_M``).
    - Per-camera OpenCV calibration XML (``calibrations/intrinsic_zero/*.xml``,
      ``calibrations/extrinsic/*.xml``) with rvec/tvec extrinsics.
    - Per-frame annotation JSON (``annotations_positions/*.json``) with a
      ``personID``, a ground-plane ``positionID`` (grid id), and one bbox per
      camera view, where a bbox of -1 means "not visible in that view".

ASSUMPTION (frame indexing): secondhand descriptions of the video-to-annotation
sampling step disagree (e.g. "every 15th frame of 60 fps video" implies 4 fps,
not the 2 fps the same sentence claims — 2 fps at 60 fps would need a step of
30). Rather than guess which is right, :func:`load_annotations` never computes
a frame index from an assumed step: it reads the absolute frame index directly
out of each annotation JSON's filename stem (EPFL's own file naming already
encodes it). The "35 frames of unannotated lead-in" some descriptions mention
refers to raw-video synchronisation before frame 0 and is not needed here,
since this loader consumes the pre-extracted per-frame JSON/image files rather
than raw video.

ASSUMPTION (viewNum <-> camera_id): the annotation JSON identifies a camera by
a 0-based ``viewNum``, not by name. This module assumes ``viewNum`` indexes
into the *same order* the camera_id list is given in (by default, the order
:func:`load_rig` discovers camera XMLs in, alphabetically by filename). This
cannot be verified without the real archive in hand; callers who know the
true EPFL viewNum<->camera mapping should pass an explicit ``camera_ids``
sequence to :func:`load_annotations` in that order.

ASSUMPTION (extrinsic units): WILDTRACK's ground-plane grid is defined in
centimetres with origin (-300, -900) cm (see ``ORIGIN_X_M``/``ORIGIN_Y_M``
below, i.e. -3.0 m / -9.0 m). This module assumes the shipped extrinsic
``tvec`` is calibrated in that same centimetre world frame, and converts it to
metres before building the ground homography. If this is wrong for the actual
archive, :func:`load_camera_calibration` will produce a homography that is
off by a constant scale factor — easy to spot (camera heights and floor
extents come out ~100x wrong) and to fix at the ``tvec`` conversion line.

World-frame convention here matches ``mcreid.calib.schema``: right-handed,
metres, Z=0 is the floor.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from mcreid.calib.geometry import horizon_sign
from mcreid.calib.schema import CameraCalib, GroundPlane, Intrinsics, RigCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

# --- dataset-level constants (see module docstring for sources) -----------

N_CAMERAS = 7
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
N_ANNOTATED_FRAMES = 400
FRAME_SAMPLING_HZ = 2.0

# --- ground-plane grid convention -------------------------------------------
# grid_id = grid_y * GRID_WIDTH_CELLS + grid_x
# world_x_m = ORIGIN_X_M + grid_x * CELL_SIZE_M   (grid_x in [0, GRID_WIDTH_CELLS))
# world_y_m = ORIGIN_Y_M + grid_y * CELL_SIZE_M   (grid_y in [0, GRID_HEIGHT_CELLS))

GRID_WIDTH_CELLS = 480
GRID_HEIGHT_CELLS = 1440
CELL_SIZE_M = 0.025
ORIGIN_X_M = -3.0
ORIGIN_Y_M = -9.0
N_GRID_CELLS = GRID_WIDTH_CELLS * GRID_HEIGHT_CELLS

# (x_min, y_min, x_max, y_max) metres — the ground area the grid covers.
FLOOR_EXTENT_M: tuple[float, float, float, float] = (
    ORIGIN_X_M,
    ORIGIN_Y_M,
    ORIGIN_X_M + GRID_WIDTH_CELLS * CELL_SIZE_M,
    ORIGIN_Y_M + GRID_HEIGHT_CELLS * CELL_SIZE_M,
)

DEFAULT_CAMERA_IDS: tuple[str, ...] = tuple(f"cam{i}" for i in range(N_CAMERAS))

_INTRINSIC_SUBDIR = "intrinsic_zero"
_EXTRINSIC_SUBDIR = "extrinsic"

_INFEASIBLE = 1e5


# --- pure grid <-> world conversions ----------------------------------------


def grid_id_to_world_m(grid_id: int) -> tuple[float, float]:
    """Convert a WILDTRACK ground-grid id to world (x, y) metres.

    Exact inverse of :func:`world_m_to_grid_id` for every ``grid_id`` in
    ``[0, N_GRID_CELLS)``: the returned point lies exactly on the grid, so
    feeding it back through ``world_m_to_grid_id`` recovers ``grid_id``.

    Raises:
        ValueError: if ``grid_id`` is outside ``[0, N_GRID_CELLS)``.
    """
    if not 0 <= grid_id < N_GRID_CELLS:
        raise ValueError(f"grid_id={grid_id} outside valid range [0, {N_GRID_CELLS})")
    grid_x = grid_id % GRID_WIDTH_CELLS
    grid_y = grid_id // GRID_WIDTH_CELLS
    x = ORIGIN_X_M + grid_x * CELL_SIZE_M
    y = ORIGIN_Y_M + grid_y * CELL_SIZE_M
    return x, y


def world_m_to_grid_id(x: float, y: float) -> int:
    """Convert world (x, y) metres to the nearest WILDTRACK ground-grid id.

    Exact inverse of :func:`grid_id_to_world_m`: a point produced by that
    function round-trips back to the same id. For an arbitrary (x, y) not
    exactly on a cell centre, this returns the id of the containing cell
    (nearest-cell quantisation), which is the standard WILDTRACK convention.

    Raises:
        ValueError: if (x, y) falls outside the 12 m x 36 m grid area.
    """
    grid_x = round((x - ORIGIN_X_M) / CELL_SIZE_M)
    grid_y = round((y - ORIGIN_Y_M) / CELL_SIZE_M)
    if not 0 <= grid_x < GRID_WIDTH_CELLS:
        raise ValueError(
            f"x={x} m maps to grid_x={grid_x}, outside [0, {GRID_WIDTH_CELLS}) "
            f"— world x must be in [{ORIGIN_X_M}, {ORIGIN_X_M + GRID_WIDTH_CELLS * CELL_SIZE_M}) m"
        )
    if not 0 <= grid_y < GRID_HEIGHT_CELLS:
        raise ValueError(
            f"y={y} m maps to grid_y={grid_y}, outside [0, {GRID_HEIGHT_CELLS}) "
            f"— world y must be in [{ORIGIN_Y_M}, {ORIGIN_Y_M + GRID_HEIGHT_CELLS * CELL_SIZE_M}) m"
        )
    return grid_y * GRID_WIDTH_CELLS + grid_x


# --- calibration loaders -----------------------------------------------------

_INTRINSIC_MATRIX_KEYS = ("camera_matrix", "CameraMatrix", "K", "intrinsic_matrix")
_DISTORTION_KEYS = ("distortion_coefficients", "DistortionCoeffs", "distCoeffs", "distortion")
_RVEC_KEYS = ("rvec", "rvecs", "R")
_TVEC_KEYS = ("tvec", "tvecs", "T")


def _open_filestorage(xml_path: Path) -> cv2.FileStorage:
    if not xml_path.is_file():
        raise FileNotFoundError(f"calibration XML not found: {xml_path}")
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise OSError(f"OpenCV could not open calibration XML: {xml_path}")
    return fs


def _read_fs_matrix(fs: cv2.FileStorage, keys: tuple[str, ...], xml_path: Path) -> FloatArray:
    """Read a numeric node, whether it is a typed matrix or a bare sequence.

    WILDTRACK mixes both styles: the intrinsics files use
    ``type_id="opencv-matrix"`` nodes, while the extrinsics store rvec/tvec as
    plain whitespace-separated sequences. Calling ``.mat()`` on the latter
    raises an assertion inside OpenCV rather than returning None, so the
    sequence form has to be read element by element.
    """
    for key in keys:
        node = fs.getNode(key)
        if node.empty():
            continue
        if node.isSeq():
            values = [float(node.at(i).real()) for i in range(node.size())]
            if values:
                return np.asarray(values, dtype=np.float64)
            continue
        try:
            mat = node.mat()
        except cv2.error:
            mat = None
        if mat is not None and np.asarray(mat).size > 0:
            return np.asarray(mat, dtype=np.float64)
        if node.isReal():
            return np.asarray([node.real()], dtype=np.float64)
    raise KeyError(f"{xml_path}: none of the expected keys {keys} found in this calibration XML")


def _read_intrinsic_xml(xml_path: Path) -> tuple[FloatArray, FloatArray]:
    fs = _open_filestorage(xml_path)
    try:
        K = _read_fs_matrix(fs, _INTRINSIC_MATRIX_KEYS, xml_path)  # noqa: N806
        dist = _read_fs_matrix(fs, _DISTORTION_KEYS, xml_path)
    finally:
        fs.release()
    if K.shape != (3, 3):
        raise ValueError(f"{xml_path}: camera matrix must be 3x3, got {K.shape}")
    return K, dist.ravel()


def _read_extrinsic_xml(xml_path: Path) -> tuple[FloatArray, FloatArray]:
    fs = _open_filestorage(xml_path)
    try:
        rvec = _read_fs_matrix(fs, _RVEC_KEYS, xml_path).ravel()
        tvec = _read_fs_matrix(fs, _TVEC_KEYS, xml_path).ravel()
    finally:
        fs.release()
    if rvec.size != 3 or tvec.size != 3:
        raise ValueError(
            f"{xml_path}: expected 3-element rvec/tvec, got sizes {rvec.size}/{tvec.size}"
        )
    return rvec, tvec


def load_camera_calibration(
    intrinsic_xml: Path,
    extrinsic_xml: Path,
    camera_id: str,
) -> CameraCalib:
    """Build a `CameraCalib` from one camera's WILDTRACK OpenCV XML pair.

    The ground homography is the analytic Z=0 restriction of the projection
    matrix — ``H = P[:, [0, 1, 3]]`` — exactly as
    ``mcreid.sim.virtual_camera.VirtualCamera.H_world2img`` computes it for
    the synthetic cameras, so the same downstream geometry code
    (``mcreid.calib.geometry``) applies unchanged.

    ``GroundPlane.method`` has no Literal value for "derived analytically
    from an OpenCV extrinsic calibration"; ``"four_point"`` is used as the
    closest existing option (both are closed-form fits rather than iterative
    RANSAC refinement) rather than inventing a new schema value.

    Raises:
        FileNotFoundError: if either XML is missing.
        KeyError: if the expected OpenCV FileStorage keys are absent.
        ValueError: if the calibration is malformed or geometrically degenerate.
    """
    intrinsic_xml = Path(intrinsic_xml)
    extrinsic_xml = Path(extrinsic_xml)
    if not camera_id:
        raise ValueError("camera_id must be non-empty")

    K, dist = _read_intrinsic_xml(intrinsic_xml)  # noqa: N806
    rvec, tvec_cm = _read_extrinsic_xml(extrinsic_xml)
    R, _ = cv2.Rodrigues(rvec)  # noqa: N806
    R = np.asarray(R, dtype=np.float64)  # noqa: N806
    # ASSUMPTION: tvec is calibrated in the grid's centimetre world frame —
    # see module docstring.
    tvec_m = tvec_cm / 100.0

    intrinsics = Intrinsics.from_matrices(
        K=K,
        dist=dist,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        rms_reproj_px=0.0,  # not published per-camera by WILDTRACK
        n_views=0,  # not published by WILDTRACK
    )

    projection = K @ np.hstack([R, tvec_m.reshape(3, 1)])  # 3x4, world m -> homogeneous pixels
    H_world2img = projection[:, [0, 1, 3]]  # noqa: N806 - restrict to Z=0
    if abs(H_world2img[2, 2]) < 1e-12:
        raise ValueError(f"camera {camera_id!r}: degenerate ground homography (H[2, 2] ~ 0)")
    H_world2img = H_world2img / H_world2img[2, 2]  # noqa: N806
    H_img2world = np.linalg.inv(H_world2img)  # noqa: N806
    H_img2world = H_img2world / H_img2world[2, 2]  # noqa: N806

    try:
        horizon_sign(H_img2world, (IMAGE_WIDTH, IMAGE_HEIGHT))
    except ValueError as exc:
        raise ValueError(f"camera {camera_id!r}: {exc}") from exc

    ground = GroundPlane.from_matrix(
        H=H_img2world,
        method="four_point",
        rms_error_m=0.0,  # analytic homography, not fit -> no correspondence residual to report
        n_correspondences=4,  # schema minimum; not applicable to an analytic homography
        floor_extent_m=FLOOR_EXTENT_M,
    )

    camera_center_m = -R.T @ tvec_m
    height_m = float(camera_center_m[2]) if camera_center_m[2] > 0.0 else None
    if height_m is None:
        logger.warning(
            "camera %s: computed height %.3f m is not positive; leaving height_m unset",
            camera_id,
            float(camera_center_m[2]),
        )

    return CameraCalib(
        camera_id=camera_id,
        intrinsics=intrinsics,
        ground=ground,
        height_m=height_m,
        notes=f"WILDTRACK calibration: {intrinsic_xml.name} / {extrinsic_xml.name}",
    )


def _camera_id_from_stem(stem: str) -> str:
    for prefix in ("intr_", "intrinsic_", "extr_", "extrinsic_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def load_rig(calib_root: Path) -> RigCalib:
    """Load all cameras under a WILDTRACK ``calibrations/`` directory.

    ASSUMPTION (file naming): intrinsic/extrinsic XML pairs share a common
    suffix once an ``intr_``/``extr_`` (or ``intrinsic_``/``extrinsic_``)
    prefix is stripped, e.g. ``intr_CVLab1.xml`` <-> ``extr_CVLab1.xml``. If
    the real archive names files differently, this raises a clear
    ``FileNotFoundError`` naming the camera it could not pair, rather than
    guessing.

    Raises:
        FileNotFoundError: if the expected subdirectories or XML files are missing.
    """
    calib_root = Path(calib_root)
    intrinsic_dir = calib_root / _INTRINSIC_SUBDIR
    extrinsic_dir = calib_root / _EXTRINSIC_SUBDIR
    if not intrinsic_dir.is_dir():
        raise FileNotFoundError(f"WILDTRACK intrinsic directory not found: {intrinsic_dir}")
    if not extrinsic_dir.is_dir():
        raise FileNotFoundError(f"WILDTRACK extrinsic directory not found: {extrinsic_dir}")

    intrinsic_files = sorted(intrinsic_dir.glob("*.xml"))
    if not intrinsic_files:
        raise FileNotFoundError(f"no *.xml intrinsic files in {intrinsic_dir}")
    extrinsic_by_id = {
        _camera_id_from_stem(p.stem): p for p in sorted(extrinsic_dir.glob("*.xml"))
    }
    if not extrinsic_by_id:
        raise FileNotFoundError(f"no *.xml extrinsic files in {extrinsic_dir}")

    cameras: list[CameraCalib] = []
    for intr_path in intrinsic_files:
        camera_id = _camera_id_from_stem(intr_path.stem)
        extr_path = extrinsic_by_id.get(camera_id)
        if extr_path is None:
            raise FileNotFoundError(
                f"no extrinsic XML matching camera {camera_id!r} (from {intr_path.name}) "
                f"in {extrinsic_dir}; found extrinsic ids: {sorted(extrinsic_by_id)}"
            )
        cameras.append(load_camera_calibration(intr_path, extr_path, camera_id))

    if len(cameras) != N_CAMERAS:
        logger.warning(
            "loaded %d camera(s) from %s; WILDTRACK ships %d", len(cameras), calib_root, N_CAMERAS
        )
    return RigCalib(cameras=cameras)


# --- annotation loader --------------------------------------------------------


@dataclass(frozen=True)
class WildtrackAnnotation:
    """One annotated person in one frame.

    ``bboxes`` maps ``camera_id -> (4,) xyxy pixel box``, or ``None`` when the
    person is not visible in that view (WILDTRACK encodes this as -1 in the
    bbox fields).
    """

    frame: int
    person_id: int
    world_xy: FloatArray  # (2,) metres, from grid_id_to_world_m(positionID)
    bboxes: dict[str, FloatArray | None]

    def __post_init__(self) -> None:
        xy = np.asarray(self.world_xy, dtype=np.float64)
        if xy.shape != (2,):
            raise ValueError(f"world_xy must be (2,), got {xy.shape}")


def load_annotations(
    annotation_dir: Path,
    camera_ids: Sequence[str] = DEFAULT_CAMERA_IDS,
) -> dict[int, list[WildtrackAnnotation]]:
    """Load every per-frame annotation JSON in ``annotation_dir``.

    Args:
        annotation_dir: WILDTRACK's ``annotations_positions/`` directory.
        camera_ids: camera id per 0-based ``viewNum``, in ``viewNum`` order.
            See the module docstring's viewNum<->camera_id assumption; pass
            ``rig.camera_ids`` here to align annotations with a loaded rig.

    Returns:
        ``{frame_index: [WildtrackAnnotation, ...]}``, one entry per
        ``*.json`` file, keyed by the integer frame index encoded in the
        filename stem (see module docstring).

    Raises:
        FileNotFoundError: if the directory or its JSON files are missing.
        ValueError: if a filename or record is malformed.
    """
    annotation_dir = Path(annotation_dir)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"WILDTRACK annotation directory not found: {annotation_dir}")
    files = sorted(annotation_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json annotation files in {annotation_dir}")

    out: dict[int, list[WildtrackAnnotation]] = {}
    for path in files:
        try:
            frame = int(path.stem)
        except ValueError as exc:
            raise ValueError(
                f"{path}: filename stem {path.stem!r} is not an integer frame index "
                "(WILDTRACK annotation files are named by absolute frame number)"
            ) from exc

        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list of person records, got {type(records)}")

        people: list[WildtrackAnnotation] = []
        for record in records:
            person_id = int(record["personID"])
            grid_id = int(record["positionID"])
            world_xy = np.asarray(grid_id_to_world_m(grid_id), dtype=np.float64)

            bboxes: dict[str, FloatArray | None] = dict.fromkeys(camera_ids)
            for view in record.get("views", []):
                view_num = int(view["viewNum"])
                if not 0 <= view_num < len(camera_ids):
                    raise ValueError(
                        f"{path}: viewNum {view_num} out of range for "
                        f"{len(camera_ids)} camera_ids"
                    )
                camera_id = camera_ids[view_num]
                xmin, ymin = float(view["xmin"]), float(view["ymin"])
                xmax, ymax = float(view["xmax"]), float(view["ymax"])
                if min(xmin, ymin, xmax, ymax) < 0:
                    bboxes[camera_id] = None
                else:
                    bboxes[camera_id] = np.array([xmin, ymin, xmax, ymax], dtype=np.float64)

            people.append(
                WildtrackAnnotation(
                    frame=frame, person_id=person_id, world_xy=world_xy, bboxes=bboxes
                )
            )
        out[frame] = people

    logger.info("loaded %d annotated frame(s) from %s", len(out), annotation_dir)
    return out


# --- multi-view detection metrics (MODA / MODP) ------------------------------


@dataclass(frozen=True)
class MultiviewDetectionMetrics:
    """Standard multi-view ground-plane detection metrics.

    Definitions (distance threshold ``threshold_m`` on the ground plane,
    Hungarian-matched per frame, then aggregated over the whole sequence):

        MODA = 1 - (FP + FN) / N_gt
        MODP = mean over matched (gt, pred) pairs of (1 - d / threshold_m)
        precision = TP / (TP + FP)
        recall = TP / N_gt = TP / (TP + FN)

    where a prediction and a ground-truth point are matched only if their
    distance is <= ``threshold_m`` (Hungarian assignment restricted to
    feasible pairs), TP is the number of matched pairs, FP is unmatched
    predictions, and FN is unmatched ground truth.
    """

    moda: float
    modp: float
    precision: float
    recall: float
    n_gt: int
    n_pred: int
    n_tp: int
    n_fp: int
    n_fn: int


def _as_points(arr: npt.ArrayLike) -> FloatArray:
    out = np.asarray(arr, dtype=np.float64)
    if out.size == 0:
        return out.reshape(0, 2)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"expected (N, 2) world points, got shape {out.shape}")
    return out


def compute_moda_modp(
    predictions: Sequence[npt.ArrayLike],
    ground_truth: Sequence[npt.ArrayLike],
    threshold_m: float = 0.5,
) -> MultiviewDetectionMetrics:
    """Score per-frame ground-plane detections against ground truth.

    Args:
        predictions: per-frame sequence of (N_i, 2) predicted world points (metres).
        ground_truth: per-frame sequence of (M_i, 2) ground-truth world points (metres),
            same length as ``predictions``.
        threshold_m: maximum ground-plane distance for a valid match (WILDTRACK
            standard is 0.5 m).

    Returns:
        Aggregate :class:`MultiviewDetectionMetrics` over every frame.

    Raises:
        ValueError: if lengths mismatch, a frame's array is malformed, or
            ``threshold_m`` is not positive.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions has {len(predictions)} frame(s) but "
            f"ground_truth has {len(ground_truth)}"
        )
    if threshold_m <= 0.0:
        raise ValueError(f"threshold_m must be positive, got {threshold_m}")

    n_gt = n_pred = n_tp = n_fp = n_fn = 0
    matched_distances: list[float] = []

    for frame_idx, (pred_raw, gt_raw) in enumerate(
        zip(predictions, ground_truth, strict=True)
    ):
        pred = _as_points(pred_raw)
        gt = _as_points(gt_raw)
        n_gt += gt.shape[0]
        n_pred += pred.shape[0]

        if gt.shape[0] == 0 or pred.shape[0] == 0:
            n_fp += pred.shape[0]
            n_fn += gt.shape[0]
            continue

        distance = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
        cost = np.where(distance <= threshold_m, distance, _INFEASIBLE)
        rows, cols = linear_sum_assignment(cost)

        n_matched = 0
        matched_pred_cols: set[int] = set()
        for r, c in zip(rows, cols, strict=True):
            if distance[r, c] <= threshold_m:
                n_matched += 1
                matched_pred_cols.add(int(c))
                matched_distances.append(float(distance[r, c]))
        n_tp += n_matched
        n_fn += gt.shape[0] - n_matched
        n_fp += pred.shape[0] - len(matched_pred_cols)
        logger.debug(
            "frame %d: gt=%d pred=%d matched=%d", frame_idx, gt.shape[0], pred.shape[0], n_matched
        )

    moda = 1.0 - (n_fp + n_fn) / n_gt if n_gt > 0 else float("nan")
    modp = (
        float(np.mean([1.0 - d / threshold_m for d in matched_distances]))
        if matched_distances
        else float("nan")
    )
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else float("nan")
    recall = n_tp / n_gt if n_gt > 0 else float("nan")

    return MultiviewDetectionMetrics(
        moda=moda,
        modp=modp,
        precision=precision,
        recall=recall,
        n_gt=n_gt,
        n_pred=n_pred,
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
    )
