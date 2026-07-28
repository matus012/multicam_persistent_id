"""Appearance embedders, selectable by name.

Two backends, both *pretrained by someone else* and neither trained by us:

``imagenet_resnet18``
    An ImageNet classification trunk with the classifier removed. This is the
    v1 baseline. Measured on WILDTRACK it separates same-person-cross-camera
    (0.377 cosine distance) from different-person-cross-camera (0.408) by 0.031
    — i.e. it barely works across viewpoints. Kept selectable so that number
    stays reproducible as an ablation, not as a recommendation.

``osnet_x1_0_msmt17``
    OSNet trained for person re-identification on MSMT17 (combineall). MSMT17 is
    a different domain from WILDTRACK, so this is genuinely zero-shot: no
    training by us, and no exposure to the evaluation data.

Using a pretrained ReID network does not weaken the project's "zero training"
claim. The detector (YOLO11x on COCO) is equally pretrained; the claim is that
*we* train nothing, and that no component has seen the evaluation set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.utils.device import DeviceSpec
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class EmbedderSpec:
    """Everything needed to build and fetch one embedder."""

    name: str
    crop_hw: tuple[int, int]
    dim: int
    weights_file: str | None = None
    url: str | None = None
    sha256: str | None = None
    trained_for_reid: bool = False
    provenance: str = ""


OSNET_MSMT17 = EmbedderSpec(
    name="osnet_x1_0_msmt17",
    crop_hw=(256, 128),
    dim=512,
    weights_file="osnet_x1_0_msmt17.pth",
    url=(
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
        "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    ),
    sha256="48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    trained_for_reid=True,
    provenance="OSNet (Zhou et al.), MIT, trained on MSMT17-combineall by the authors",
)

IMAGENET_RESNET18 = EmbedderSpec(
    name="imagenet_resnet18",
    crop_hw=(128, 64),
    dim=512,
    trained_for_reid=False,
    provenance="torchvision ResNet-18, ImageNet1K classification weights",
)

REGISTRY: dict[str, EmbedderSpec] = {
    OSNET_MSMT17.name: OSNET_MSMT17,
    IMAGENET_RESNET18.name: IMAGENET_RESNET18,
}
DEFAULT_EMBEDDER = OSNET_MSMT17.name


class Embedder(Protocol):
    """Anything that turns person crops into L2-normalised vectors."""

    dim: int

    def __call__(self, image: Image, boxes: FloatArray) -> FloatArray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(spec: EmbedderSpec, weights_dir: Path = Path("weights")) -> Path:
    """Return a verified local path to ``spec``'s weights, downloading if needed."""
    if spec.weights_file is None or spec.url is None:
        raise ValueError(f"{spec.name} has no downloadable weights")
    path = weights_dir / spec.weights_file
    if not path.is_file():
        import subprocess

        weights_dir.mkdir(parents=True, exist_ok=True)
        logger.info("downloading %s weights -> %s", spec.name, path)
        result = subprocess.run(
            ["curl", "-L", "--fail", "-o", str(path), spec.url], check=False
        )
        if result.returncode != 0 or not path.is_file():
            raise RuntimeError(
                f"failed to download {spec.name} weights from {spec.url}. "
                "Download it manually and place it at " + str(path)
            )
    if spec.sha256:
        observed = _sha256(path)
        if observed != spec.sha256:
            raise RuntimeError(
                f"{path} checksum mismatch:\n  expected {spec.sha256}\n  observed {observed}\n"
                "The file is corrupt or is not the weights this code was written against."
            )
    return path


class _CropEmbedder:
    """Shared crop -> tensor -> normalise plumbing."""

    def __init__(self, spec: EmbedderSpec, device: DeviceSpec, batch_size: int) -> None:
        import torch

        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.dim = spec.dim
        self._torch = torch
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device.torch_device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device.torch_device).view(
            1, 3, 1, 1
        )
        self.model = self._build()

    def _build(self) -> Any:
        """Return an eval-mode torch module mapping crop tensors to features."""
        raise NotImplementedError

    def __call__(self, image: Image, boxes: FloatArray) -> FloatArray:
        torch = self._torch
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float64)

        height, width = image.shape[:2]
        crop_h, crop_w = self.spec.crop_hw
        crops = []
        for box in boxes:
            x1 = int(np.clip(box[0], 0, width - 1))
            y1 = int(np.clip(box[1], 0, height - 1))
            x2 = int(np.clip(box[2], x1 + 1, width))
            y2 = int(np.clip(box[3], y1 + 1, height))
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                patch = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            crops.append(cv2.resize(patch, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR))

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


class ImageNetResnet18Embedder(_CropEmbedder):
    """The v1 baseline: an ImageNet trunk pressed into service as a ReID model."""

    def _build(self) -> Any:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.eval().to(self.device.torch_device)
        if self.device.kind == "cuda" and self.device.use_half:
            model = model.half()
        logger.info("embedder: ImageNet ResNet-18 (NOT trained for ReID), 512-d")
        return model


class OsnetEmbedder(_CropEmbedder):
    """OSNet trained for person ReID on MSMT17 — zero-shot on WILDTRACK."""

    def __init__(
        self, spec: EmbedderSpec, device: DeviceSpec, batch_size: int, weights: Path
    ) -> None:
        self._weights = weights
        super().__init__(spec, device, batch_size)

    def _build(self) -> Any:
        import torch

        from mcreid.track.vendor.osnet import osnet_x1_0

        model = osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
        state = torch.load(self._weights, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        # Drop the classification head: it is sized to the training set's
        # identity count (4101 for MSMT17-combineall), we never use it, and
        # keeping it only produces a shape-mismatch error.
        cleaned = {
            k.removeprefix("module."): v
            for k, v in state.items()
            if not k.removeprefix("module.").startswith("classifier.")
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        # Anything else missing means the weights do not match the architecture
        # and the features would be silently meaningless.
        real_missing = [k for k in missing if not k.startswith("classifier.")]
        if real_missing:
            raise RuntimeError(
                f"OSNet weights do not match the architecture; missing {real_missing[:5]}"
            )
        if unexpected:
            logger.debug("ignored %d unexpected keys (e.g. classifier)", len(unexpected))
        model = model.eval().to(self.device.torch_device)
        if self.device.kind == "cuda" and self.device.use_half:
            model = model.half()
        logger.info("embedder: OSNet x1.0, MSMT17-trained, %d-d", self.spec.dim)
        return model


def build_embedder(
    name: str = DEFAULT_EMBEDDER,
    device: DeviceSpec | None = None,
    batch_size: int = 32,
    weights_dir: Path = Path("weights"),
) -> Embedder:
    """Construct an embedder by registry name."""
    if name not in REGISTRY:
        raise ValueError(f"unknown embedder {name!r}; available: {sorted(REGISTRY)}")
    from mcreid.utils.device import resolve_device

    spec = REGISTRY[name]
    resolved = device or resolve_device("auto")

    if spec.name == IMAGENET_RESNET18.name:
        return ImageNetResnet18Embedder(spec, resolved, batch_size)
    weights = ensure_weights(spec, weights_dir)
    return OsnetEmbedder(spec, resolved, batch_size, weights)
