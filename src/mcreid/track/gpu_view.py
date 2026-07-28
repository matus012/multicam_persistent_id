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
from mcreid.utils.device import DeviceSpec, resolve_device
from mcreid.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from mcreid.fusion.types import ViewObservation

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]

PERSON_CLASS = 0
REID_INPUT_HW = (128, 64)  # (height, width), the standard person-ReID crop


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

    def __post_init__(self) -> None:
        if not 0.0 < self.conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {self.conf_threshold}")
        if self.imgsz % 32 != 0:
            raise ValueError(f"imgsz must be a multiple of 32, got {self.imgsz}")
        if self.max_detections < 1 or self.embed_batch < 1:
            raise ValueError("max_detections and embed_batch must be >= 1")


class ReidEmbedder:
    """L2-normalised appearance vectors from person crops."""

    def __init__(self, device: DeviceSpec, batch_size: int = 32) -> None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = device
        self.batch_size = batch_size
        self._torch = torch

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device.torch_device)
        if device.kind == "cuda" and device.use_half:
            self.model = self.model.half()

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device.torch_device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device.torch_device).view(
            1, 3, 1, 1
        )
        self.dim = 512
        logger.info("ReID embedder: ImageNet ResNet-18, %d-d, on %s", self.dim, device)

    def __call__(self, image: Image, boxes: FloatArray) -> FloatArray:
        """(N, 4) xyxy boxes -> (N, 512) unit vectors."""
        torch = self._torch
        if boxes.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float64)

        import cv2

        height, width = image.shape[:2]
        crops = []
        for box in boxes:
            x1 = int(np.clip(box[0], 0, width - 1))
            y1 = int(np.clip(box[1], 0, height - 1))
            x2 = int(np.clip(box[2], x1 + 1, width))
            y2 = int(np.clip(box[3], y1 + 1, height))
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                patch = np.zeros((*REID_INPUT_HW, 3), dtype=np.uint8)
            crops.append(
                cv2.resize(
                    patch,
                    (REID_INPUT_HW[1], REID_INPUT_HW[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            )

        stacked = np.stack(crops)[:, :, :, ::-1]  # BGR -> RGB
        tensor = torch.from_numpy(np.ascontiguousarray(stacked)).permute(0, 3, 1, 2)
        tensor = tensor.to(self.device.torch_device).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        if self.device.kind == "cuda" and self.device.use_half:
            tensor = tensor.half()

        outputs = []
        with torch.inference_mode():
            for start in range(0, tensor.shape[0], self.batch_size):
                outputs.append(self.model(tensor[start : start + self.batch_size]).float())
        features = torch.cat(outputs, dim=0)
        features = torch.nn.functional.normalize(features, dim=1)
        return features.cpu().numpy().astype(np.float64)


class GpuPerViewBackend:
    """One camera: detect people, embed them, track them, emit ViewObservations."""

    def __init__(
        self,
        camera_id: str,
        config: GpuViewConfig | None = None,
        per_view_config: PerViewConfig | None = None,
        detector: Any | None = None,
        embedder: ReidEmbedder | None = None,
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
        self.embedder = embedder or ReidEmbedder(self.device, self.config.embed_batch)
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
