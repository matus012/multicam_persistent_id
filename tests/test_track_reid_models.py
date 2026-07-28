"""Tests for the selectable appearance-embedder registry.

Kept off the GPU and off the network: what matters here is that the registry is
coherent, that weight verification actually rejects a bad file, and that the
zero-shot provenance of each backend is recorded rather than implied.
"""

from __future__ import annotations

import pytest

from mcreid.track.reid_models import (
    DEFAULT_EMBEDDER,
    IMAGENET_RESNET18,
    OSNET_MSMT17,
    REGISTRY,
    EmbedderSpec,
    ensure_weights,
)


def test_registry_is_coherent() -> None:
    for name, spec in REGISTRY.items():
        assert spec.name == name, f"{name} is filed under the wrong key"
        assert spec.dim > 0
        assert len(spec.crop_hw) == 2 and all(v > 0 for v in spec.crop_hw)
        assert spec.provenance, f"{name} must record where its weights came from"


def test_default_is_the_reid_trained_model() -> None:
    """The ImageNet trunk is an ablation, not the shipping default.

    Measured on WILDTRACK it separates same-person from different-person across
    cameras by 0.03 cosine; OSNet manages 0.10. Defaulting to the weaker one
    would quietly reproduce the v1 result.
    """
    assert OSNET_MSMT17.name == DEFAULT_EMBEDDER
    assert REGISTRY[DEFAULT_EMBEDDER].trained_for_reid


def test_imagenet_backend_is_flagged_as_not_a_reid_model() -> None:
    assert not IMAGENET_RESNET18.trained_for_reid
    assert IMAGENET_RESNET18.url is None, "the ImageNet trunk comes from torchvision"


def test_osnet_pins_a_checksum_and_a_direct_url() -> None:
    assert OSNET_MSMT17.url and OSNET_MSMT17.url.startswith("https://")
    assert OSNET_MSMT17.sha256 and len(OSNET_MSMT17.sha256) == 64
    assert "msmt17" in OSNET_MSMT17.url.lower(), (
        "the pinned weights must be the MSMT17-trained ones — a Market1501 or "
        "ImageNet checkpoint would change what 'zero-shot' means here"
    )


def test_ensure_weights_rejects_a_corrupt_file(tmp_path) -> None:
    """A silently wrong checkpoint produces meaningless features, so the
    checksum is a hard gate rather than a warning."""
    spec = EmbedderSpec(
        name="fake",
        crop_hw=(256, 128),
        dim=512,
        weights_file="fake.pth",
        url="https://example.invalid/fake.pth",
        sha256="0" * 64,
    )
    (tmp_path / "fake.pth").write_bytes(b"not the weights you are looking for")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_weights(spec, weights_dir=tmp_path)


def test_ensure_weights_accepts_a_matching_file(tmp_path) -> None:
    import hashlib

    payload = b"deterministic bytes"
    (tmp_path / "ok.pth").write_bytes(payload)
    spec = EmbedderSpec(
        name="ok",
        crop_hw=(256, 128),
        dim=512,
        weights_file="ok.pth",
        url="https://example.invalid/ok.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert ensure_weights(spec, weights_dir=tmp_path) == tmp_path / "ok.pth"


def test_ensure_weights_refuses_a_spec_without_weights() -> None:
    with pytest.raises(ValueError, match="no downloadable weights"):
        ensure_weights(IMAGENET_RESNET18)


def test_build_embedder_rejects_unknown_names() -> None:
    from mcreid.track.reid_models import build_embedder

    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("definitely_not_a_model")
