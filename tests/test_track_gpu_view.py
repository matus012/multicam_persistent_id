"""Contract tests for the GPU per-view backend.

Runs on CPU with a fake detector and a fake embedder: the point is that the GPU
path emits exactly the same `ViewObservation` contract as the torch-free one, so
the entire fusion stack stays covered by tests that need no GPU and no footage.
Detection quality is not testable here and is not claimed to be.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.track.gpu_view import GpuViewConfig

if TYPE_CHECKING:
    from mcreid.track.gpu_view import GpuPerViewBackend

FloatArray = npt.NDArray[np.float64]


class _FakeBoxes:
    def __init__(self, boxes: FloatArray, scores: FloatArray) -> None:
        self.xyxy = _FakeTensor(boxes)
        self.conf = _FakeTensor(scores)

    def __len__(self) -> int:
        return int(self.xyxy.value.shape[0])


class _FakeTensor:
    """Mimics the `.cpu().numpy()` chain ultralytics results expose."""

    def __init__(self, value: FloatArray) -> None:
        self.value = value

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> FloatArray:
        return self.value


class _FakeResult:
    def __init__(self, boxes: FloatArray | None, scores: FloatArray | None) -> None:
        self.boxes = None if boxes is None else _FakeBoxes(boxes, scores)


class _FakeDetector:
    def __init__(self, boxes: FloatArray | None, scores: FloatArray | None = None) -> None:
        self.boxes = boxes
        self.scores = scores
        self.calls: list[dict] = []

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        self.calls.append(dict(kwargs))
        return [_FakeResult(self.boxes, self.scores)]


class _FakeEmbedder:
    """Deterministic unit vectors, one per box."""

    dim = 8

    def __call__(self, image, boxes: FloatArray) -> FloatArray:
        n = boxes.shape[0]
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        out = np.zeros((n, self.dim), dtype=np.float64)
        for i in range(n):
            out[i, i % self.dim] = 1.0
        return out


def _backend(detector: _FakeDetector) -> GpuPerViewBackend:
    from mcreid.track.gpu_view import GpuPerViewBackend

    return GpuPerViewBackend(
        "cam0",
        GpuViewConfig(device="cpu"),
        detector=detector,
        embedder=_FakeEmbedder(),
    )


def _image() -> npt.NDArray[np.uint8]:
    return np.full((480, 640, 3), 120, dtype=np.uint8)


# --- config validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"conf_threshold": 0.0}, "conf_threshold"),
        ({"conf_threshold": 1.0}, "conf_threshold"),
        ({"imgsz": 641}, "imgsz"),
        ({"max_detections": 0}, "max_detections"),
        ({"embed_batch": 0}, "embed_batch"),
    ],
)
def test_config_rejects_garbage(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        GpuViewConfig(**kwargs)


# --- contract ------------------------------------------------------------------------------


def test_emits_view_observations_with_normalised_embeddings() -> None:
    boxes = np.array([[10.0, 20.0, 60.0, 160.0], [200.0, 30.0, 250.0, 170.0]])
    scores = np.array([0.9, 0.8])
    backend = _backend(_FakeDetector(boxes, scores))

    backend.step(_image(), 0)
    observations = backend.step(_image(), 1)

    assert observations, "two stable detections should confirm a tracklet by frame 1"
    for obs in observations:
        assert obs.camera_id == "cam0"
        assert obs.frame == 1
        assert np.isclose(np.linalg.norm(obs.embedding), 1.0, atol=1e-6)
        assert 0.0 <= obs.score <= 1.0


def test_no_detections_yields_no_observations() -> None:
    backend = _backend(_FakeDetector(None))
    assert backend.step(_image(), 0) == []
    assert backend.step(_image(), 1) == []


def test_local_ids_are_stable_across_frames() -> None:
    boxes = np.array([[10.0, 20.0, 60.0, 160.0]])
    backend = _backend(_FakeDetector(boxes, np.array([0.9])))

    seen = set()
    for frame in range(6):
        for obs in backend.step(_image(), frame):
            seen.add(obs.local_track_id)
    assert len(seen) == 1, f"a stationary person must keep one local id, got {seen}"


def test_person_class_and_imgsz_are_passed_to_the_detector() -> None:
    detector = _FakeDetector(np.zeros((0, 4)), np.zeros(0))
    backend = _backend(detector)
    backend.step(_image(), 0)

    assert detector.calls, "the detector was never called"
    call = detector.calls[0]
    assert call["classes"] == [0], "must request the person class only"
    assert call["imgsz"] == GpuViewConfig().imgsz
    # fp16 is a CUDA-only concern; on CPU neither flag should be sent.
    assert "half" not in call and "quantize" not in call


def test_scores_are_clipped_into_range() -> None:
    """A detector returning a marginally out-of-range score must not crash the
    ViewObservation contract, which validates 0 <= score <= 1."""
    boxes = np.array([[10.0, 20.0, 60.0, 160.0]])
    backend = _backend(_FakeDetector(boxes, np.array([1.0000001])))
    backend.step(_image(), 0)
    observations = backend.step(_image(), 1)
    assert all(0.0 <= o.score <= 1.0 for o in observations)
