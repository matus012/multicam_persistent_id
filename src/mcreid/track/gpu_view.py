"""GPU per-view backend: YOLO11 detection + ReID embedding -> `ViewObservation`.

Emits exactly the same contract as the torch-free `PerViewTracker`, so the whole
fusion stack downstream is unchanged and stays covered by the CPU test suite.
Only this file needs a GPU, and it is the only file real footage introduces.

Tracking is deliberately kept in `PerViewTracker` rather than delegated to
Ultralytics' built-in BoT-SORT: the fusion stage needs a stable appearance
vector per tracklet, and the shipped tracker's ReID plumbing varies between
releases. Detection is what the GPU is for; association is already tested.

The appearance model is an ImageNet ResNet-18 trunk with the classifier removed
— the same zero-training baseline P1 uses. That is a weak person-ReID model and
it is *supposed* to be: v1 claims a geometric baseline with no ReID training,
and swapping in a trained embedder would invalidate that claim rather than
support it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from mcreid.track.per_view import Detection, PerViewConfig, PerViewTracker
from mcreid.track.reid_models import DEFAULT_EMBEDDER, Embedder, build_embedder
from mcreid.utils.device import resolve_device
from mcreid.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from mcreid.fusion.types import ViewObservation

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]

PERSON_CLASS = 0


@dataclass(frozen=True)
class GpuViewConfig:
    """Detector + embedder settings."""

    weights: Path = Path("weights/yolo11x.pt")
    conf_threshold: float = 0.25
    """Low on purpose: the per-view tracker's second stage rescues low-score
    detections, so throwing them away here loses tracks that ByteTrack-style
    association would have kept."""
    iou_threshold: float = 0.7
    imgsz: int = 1280
    half: bool = True
    max_detections: int = 60
    embed_batch: int = 32
    device: str = "auto"
    embedder: str = DEFAULT_EMBEDDER
    """Appearance model, by name from `mcreid.track.reid_models.REGISTRY`.
    Defaults to the ReID-trained OSNet; `imagenet_resnet18` reproduces the v1
    baseline for ablation."""
    weights_dir: Path = Path("weights")

    def __post_init__(self) -> None:
        if not 0.0 < self.conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")
        if self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32, got {self.imgsz}")
        if self.max_detections < 1 or self.embed_batch < 1:
            raise ValueError("max_detections and embed_batch must be >= 1")


class GpuPerViewBackend:
    """One camera: detect people, embed them, track them, emit ViewObservations."""

    def __init__(
        self,
        camera_id: str,
        config: GpuViewConfig | None = None,
        per_view_config: PerViewConfig | None = None,
        detector: Any | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or GpuViewConfig()
        self.device = resolve_device(self.config.device, allow_half=self.config.half)
        self.tracker = PerViewTracker(camera_id, per_view_config)

        if detector is None:
            from ultralytics import YOLO

            weights = Path(self.config.weights)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"detector weights not found: {weights}. Ultralytics will download "
                    "them on first use, or copy them in manually."
                )
            detector = YOLO(str(weights))
        self.detector = detector
        self.embedder = embedder or build_embedder(
            self.config.embedder,
            device=self.device,
            batch_size=self.config.embed_batch,
            weights_dir=self.config.weights_dir,
        )
        self._predict_kwargs = self._precision_kwargs()

    def _precision_kwargs(self) -> dict[str, Any]:
        """Half-precision flag under whichever name this Ultralytics uses.

        8.4 renamed `half` to `quantize` and warns on every call with the old
        one. Probing the config once keeps a single install from either spamming
        deprecation warnings or silently losing fp16 on a future rename.
        """
        if not (self.device.kind == "cuda" and self.device.use_half):
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except ImportError:  # pragma: no cover - very old/new ultralytics
            return {"half": True}
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": "fp16"}
        if "half" in DEFAULT_CFG_DICT:
            return {"half": True}
        logger.warning("no half-precision flag found in this Ultralytics; running fp32")
        return {}

    def detect(self, image: Image) -> tuple[FloatArray, FloatArray]:
        """Returns (boxes (N,4) xyxy, scores (N,)) for the person class only."""
        # Ultralytics' Results type is loosely annotated in its stubs; the shapes
        # are checked at runtime below instead.
        predictions: Any = self.detector.predict(
            source=image,
            device=self.device.torch_device,
            classes=[PERSON_CLASS],
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            max_det=self.config.max_detections,
            verbose=False,
            **self._predict_kwargs,
        )
        result = predictions[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores = result.boxes.conf.cpu().numpy().astype(np.float64)
        return boxes, scores

    def step(self, image: Image, frame: int) -> list[ViewObservation]:
        boxes, scores = self.detect(image)
        embeddings = self.embedder(image, boxes)
        detections = [
            Detection(
                bbox_xyxy=boxes[i],
                score=float(np.clip(scores[i], 0.0, 1.0)),
                embedding=embeddings[i],
            )
            for i in range(boxes.shape[0])
        ]
        return self.tracker.update(detections, frame)
