"""Explicit device selection with a hard CPU fallback.

torch is an optional dependency (perception extra); everything here degrades to
``cpu`` when torch is absent so the core pipeline stays importable.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcreid.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved compute device."""

    kind: str  # "cuda" | "cpu"
    index: int | None
    name: str
    total_memory_mb: int | None
    use_half: bool

    @property
    def torch_device(self) -> str:
        return self.kind if self.index is None else f"{self.kind}:{self.index}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mem = f", {self.total_memory_mb} MiB" if self.total_memory_mb else ""
        return f"{self.torch_device} ({self.name}{mem}, half={self.use_half})"


def resolve_device(requested: str = "auto", allow_half: bool = True) -> DeviceSpec:
    """Resolve ``requested`` ("auto" | "cpu" | "cuda" | "cuda:N") to a DeviceSpec.

    Raises:
        RuntimeError: if CUDA was explicitly requested but is unavailable (fail fast —
            silently degrading to CPU would hide a 20x slowdown behind a green demo).
    """
    requested = requested.strip().lower()
    if requested not in {"auto", "cpu"} and not requested.startswith("cuda"):
        raise ValueError(f"unsupported device string: {requested!r}")

    if requested == "cpu":
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    try:
        import torch
    except ImportError:
        if requested.startswith("cuda"):
            raise RuntimeError(
                "device='cuda' requested but torch is not installed. "
                "Install the perception extra: uv pip install -e '.[perception]'"
            ) from None
        logger.info("torch unavailable — falling back to cpu")
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
        logger.info("CUDA unavailable — falling back to cpu")
        return DeviceSpec(kind="cpu", index=None, name="cpu", total_memory_mb=None, use_half=False)

    index = 0
    if requested.startswith("cuda:"):
        index = int(requested.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} requested but only {torch.cuda.device_count()} found")

    props = torch.cuda.get_device_properties(index)
    spec = DeviceSpec(
        kind="cuda",
        index=index,
        name=props.name,
        total_memory_mb=int(props.total_memory // (1024 * 1024)),
        use_half=allow_half,
    )
    logger.info("resolved device: %s", spec)
    return spec
