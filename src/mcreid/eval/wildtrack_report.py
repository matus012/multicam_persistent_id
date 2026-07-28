"""Calibration sanity for a dataset that ships its own calibration.

`mcreid-calibrate report` validates a calibration *we* produced, using AprilTags
we placed. WILDTRACK has no tags and we did not calibrate it, so that tool does
not apply. The question here is different and narrower:

    does our converter turn WILDTRACK's rvec/tvec into a `calib.json` that
    reproduces WILDTRACK's own geometry?

The check uses the dataset's two independent annotation streams against each
other: take a person's annotated ground-plane position, project it with the
converted calibration, and measure the distance to the bottom-centre of that
same person's annotated bounding box in that camera. Agreement means the
conversion is faithful.

This validates the CONVERTER, not the dataset. A systematic error in WILDTRACK's
own calibration would pass this check, because both sides of the comparison come
from WILDTRACK.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import ground_to_image
from mcreid.calib.schema import RigCalib
from mcreid.eval.wildtrack import load_annotations
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

# A bbox bottom-centre is only an approximation of the ground-contact point
# (feet spread, partial occlusion, annotation style), so this tolerance is about
# annotation semantics rather than calibration precision.
DEFAULT_MAX_MEDIAN_PX = 25.0


@dataclass
class CameraConversionReport:
    camera_id: str
    n_samples: int
    mean_px: float
    median_px: float
    p90_px: float
    max_px: float
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class ConversionReport:
    cameras: list[CameraConversionReport]
    max_median_px: float
    n_frames: int

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.cameras)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ok": self.ok,
                    "what_this_validates": (
                        "the WILDTRACK -> calib.json converter, by cross-checking the "
                        "dataset's ground-plane annotations against its per-view box "
                        "annotations. It does NOT validate WILDTRACK's own calibration."
                    ),
                    "n_frames_sampled": self.n_frames,
                    "threshold_median_px": self.max_median_px,
                    "cameras": [asdict(c) for c in self.cameras],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def to_markdown(self, path: Path) -> Path:
        lines = [
            "# WILDTRACK calibration conversion report",
            "",
            f"**Verdict: {'PASS' if self.ok else 'FAIL'}**  "
            f"(sampled {self.n_frames} annotated frames)",
            "",
            "Each number is the pixel distance between a person's annotated",
            "ground-plane position, projected through our converted calibration, and",
            "the bottom-centre of that same person's annotated box in that camera.",
            "",
            "This validates the **converter**, not the dataset: both sides of the",
            "comparison are WILDTRACK's own annotations, so a systematic error in",
            "their calibration would pass unnoticed.",
            "",
            "| camera | samples | mean px | median px | p90 px | ok |",
            "|---|---|---|---|---|---|",
        ]
        for cam in self.cameras:
            lines.append(
                f"| {cam.camera_id} | {cam.n_samples} | {cam.mean_px:.1f} | "
                f"{cam.median_px:.1f} | {cam.p90_px:.1f} | {'yes' if cam.ok else 'NO'} |"
            )
        if not self.ok:
            lines += ["", "## Problems", ""]
            for cam in self.cameras:
                for problem in cam.problems:
                    lines.append(f"- **{cam.camera_id}**: {problem}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def check_conversion(
    rig: RigCalib,
    annotation_dir: Path,
    max_frames: int = 20,
    max_median_px: float = DEFAULT_MAX_MEDIAN_PX,
) -> ConversionReport:
    """Cross-check the converted rig against WILDTRACK's own annotations."""
    # Key the annotation boxes by the rig's own camera ids: the loader defaults
    # to WILDTRACK's C1..C7 image-directory names, while the rig is named after
    # the calibration files (CVLab1.., IDIAP1..). Same cameras, same order.
    annotations = load_annotations(annotation_dir, camera_ids=rig.camera_ids)
    frames = sorted(annotations)[:max_frames]
    if not frames:
        raise ValueError(f"no annotations found under {annotation_dir}")

    known = set(rig.camera_ids)
    errors: dict[str, list[float]] = {camera_id: [] for camera_id in rig.camera_ids}
    for frame in frames:
        for record in annotations[frame]:
            world = np.asarray(record.world_xy, dtype=np.float64)
            for camera_id, box in record.bboxes.items():
                if box is None or camera_id not in known:
                    continue
                pixels, valid = ground_to_image(rig.get(camera_id), world.reshape(1, 2))
                if not valid[0] or not np.isfinite(pixels[0]).all():
                    continue
                foot = np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float64)
                errors[camera_id].append(float(np.linalg.norm(pixels[0] - foot)))

    reports: list[CameraConversionReport] = []
    for camera_id in rig.camera_ids:
        values = np.asarray(errors[camera_id], dtype=np.float64)
        problems: list[str] = []
        if values.size == 0:
            problems.append(
                "no annotated person was visible in this camera across the sampled "
                "frames — check the camera ordering in the converter"
            )
            reports.append(
                CameraConversionReport(camera_id, 0, *(float("nan"),) * 4, problems)
            )
            continue
        median = float(np.median(values))
        if median > max_median_px:
            problems.append(
                f"median reprojection error {median:.1f} px exceeds {max_median_px:.0f} px. "
                "The converted homography does not reproduce WILDTRACK's geometry — "
                "check the rvec/tvec units (theirs are centimetres) and the grid origin."
            )
        reports.append(
            CameraConversionReport(
                camera_id=camera_id,
                n_samples=int(values.size),
                mean_px=float(values.mean()),
                median_px=median,
                p90_px=float(np.percentile(values, 90)),
                max_px=float(values.max()),
                problems=problems,
            )
        )

    return ConversionReport(
        cameras=reports, max_median_px=max_median_px, n_frames=len(frames)
    )
