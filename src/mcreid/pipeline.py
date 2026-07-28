"""End-to-end orchestration: per-view tracking -> ground fusion -> global IDs.

`run_toy_scene` is the CI entry point. `MultiViewPipeline` is the reusable core
that the recorded/live demos drive with real frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcreid.calib.schema import RigCalib
from mcreid.eval.id_metrics import IdConsistencyReport, evaluate_id_consistency
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import GlobalTrackSnapshot, ViewObservation
from mcreid.sim.toy import ToyScene
from mcreid.track.per_view import Detection, PerViewConfig, PerViewTracker
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)


class MultiViewPipeline:
    """One per-view tracker per camera plus one global ID manager."""

    def __init__(
        self,
        rig: RigCalib,
        fusion_config: FusionConfig | None = None,
        per_view_config: PerViewConfig | None = None,
    ) -> None:
        self.rig = rig
        self.per_view_config = per_view_config or PerViewConfig()
        self.trackers = {
            camera_id: PerViewTracker(camera_id, self.per_view_config)
            for camera_id in rig.camera_ids
        }
        self.manager = GlobalIDManager(rig, fusion_config)

    def step(
        self, detections: dict[str, list[Detection]], frame: int, dt: float
    ) -> list[GlobalTrackSnapshot]:
        """Feed one synchronised frame from every camera."""
        unknown = set(detections) - set(self.trackers)
        if unknown:
            raise KeyError(f"detections from cameras not in the rig: {sorted(unknown)}")

        views: list[ViewObservation] = []
        for camera_id, tracker in self.trackers.items():
            views.extend(tracker.update(detections.get(camera_id, []), frame))
        return self.manager.step(views, frame, dt)

    @property
    def n_ids_issued(self) -> int:
        return self.manager.n_ids_issued


@dataclass
class ToyRunResult:
    """Everything a test or a report needs from one toy run."""

    snapshots: list[list[GlobalTrackSnapshot]]
    report: IdConsistencyReport
    n_ids_issued: int

    def summary(self) -> str:
        return self.report.summary()


def run_toy_scene(
    scene: ToyScene,
    fusion_config: FusionConfig | None = None,
    per_view_config: PerViewConfig | None = None,
    match_radius_m: float = 1.0,
) -> ToyRunResult:
    """Run the full pipeline over a synthetic scene and score it."""
    pipeline = MultiViewPipeline(scene.rig, fusion_config, per_view_config)
    dt = 1.0 / scene.config.fps
    snapshots: list[list[GlobalTrackSnapshot]] = []

    for frame in range(scene.n_frames):
        per_camera = {
            camera_id: [
                Detection(bbox_xyxy=d.bbox_xyxy, score=d.score, embedding=d.embedding)
                for d in dets
            ]
            for camera_id, dets in scene.frame_detections(frame).items()
        }
        snapshots.append(pipeline.step(per_camera, frame, dt))

    report = evaluate_id_consistency(
        gt_world=scene.gt_world,
        gt_visible=scene.gt_visible,
        results=snapshots,
        n_ids_issued=pipeline.n_ids_issued,
        match_radius_m=match_radius_m,
    )
    logger.info("toy run complete:\n%s", report.summary())
    return ToyRunResult(
        snapshots=snapshots, report=report, n_ids_issued=pipeline.n_ids_issued
    )
