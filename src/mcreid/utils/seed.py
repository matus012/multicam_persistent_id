"""Deterministic seeding across stdlib / numpy / torch (torch optional)."""

from __future__ import annotations

import os
import random

import numpy as np

from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> int:
    """Seed every RNG we may touch. Returns the seed for logging/provenance.

    torch is imported lazily so the core (torch-free) install stays usable.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; seeded stdlib+numpy only (seed=%d)", seed)
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("seeded stdlib+numpy+torch (seed=%d)", seed)
    return seed


def new_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return an isolated numpy Generator — preferred over global numpy state."""
    return np.random.default_rng(seed)
