"""Cross-view association: appearance gallery, cost construction, Hungarian solve.

Association runs **per camera**: each camera's ground observations are matched
one-to-one against the global track set, and a global track may collect one
measurement from every camera in the same frame. Trying to solve all cameras in
a single assignment would be wrong — it forbids exactly the many-to-one matching
that multi-view fusion exists to produce.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

FloatArray = npt.NDArray[np.float64]

# Cost written into gated-out cells. Large enough to never be chosen over a
# feasible pair, small enough to keep the solver numerically well-behaved.
INFEASIBLE = 1e5


class AppearanceGallery:
    """Per-track ReID memory.

    Holds a bounded ring of recent embeddings *per camera* plus a global EMA.
    Distance to the gallery is ``1 - max similarity`` over all stored vectors:
    a person seen from a new angle only needs to resemble their best previous
    view, not their average one, which is what makes cross-camera handoff and
    post-occlusion re-lock work.
    """

    def __init__(self, per_camera: int = 12, ema_alpha: float = 0.9) -> None:
        if per_camera < 1:
            raise ValueError(f"per_camera must be >= 1, got {per_camera}")
        if not 0.0 < ema_alpha < 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1), got {ema_alpha}")
        self.per_camera = per_camera
        self.ema_alpha = ema_alpha
        self._banks: dict[str, deque[FloatArray]] = {}
        self._ema: FloatArray | None = None

    def __len__(self) -> int:
        return sum(len(bank) for bank in self._banks.values())

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted(self._banks))

    @property
    def ema(self) -> FloatArray | None:
        return None if self._ema is None else self._ema.copy()

    def add(self, camera_id: str, embedding: npt.ArrayLike) -> None:
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            raise ValueError("cannot add a zero embedding to the gallery")
        emb = emb / norm

        bank = self._banks.setdefault(camera_id, deque(maxlen=self.per_camera))
        bank.append(emb)

        if self._ema is None:
            self._ema = emb.copy()
        else:
            blended = self.ema_alpha * self._ema + (1.0 - self.ema_alpha) * emb
            self._ema = blended / np.linalg.norm(blended)

    def seed(self, camera_id: str, embeddings: npt.ArrayLike) -> None:
        """Add historical vectors for matching breadth WITHOUT moving the EMA.

        Used when a resurrected identity inherits its stored appearance. Those
        stored vectors are deliberately diverse — they exist to match the person
        from any past viewpoint — so blending them into the EMA would leave it
        resembling no actual observation. The EMA must keep meaning "what this
        person looks like right now", so it is left for the first live
        observation to set.
        """
        data = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        bank = self._banks.setdefault(camera_id, deque(maxlen=self.per_camera))
        for vector in data:
            norm = float(np.linalg.norm(vector))
            if norm < 1e-12:
                raise ValueError("cannot seed a zero embedding")
            bank.append(vector / norm)

    def items(self) -> list[tuple[str, FloatArray]]:
        """(camera_id, embedding) pairs for every stored vector — used when one
        track absorbs a duplicate and inherits its appearance evidence."""
        return [(cam, vec) for cam, bank in self._banks.items() for vec in bank]

    def matrix(self) -> FloatArray:
        """(M, D) stack of every stored vector plus the EMA. Empty -> (0, 0)."""
        vectors = [v for bank in self._banks.values() for v in bank]
        if self._ema is not None:
            vectors.append(self._ema)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float64)
        return np.stack(vectors, axis=0)

    def robust_distance(self, embeddings: npt.ArrayLike, top_k: int = 3) -> FloatArray:
        """Cosine distance using the mean of the ``top_k`` best gallery matches.

        `distance` (1 - max similarity) is deliberately optimistic: it lets a
        person seen from a new angle match their single most similar past view.
        That is right for frame-to-frame association, but for irreversible
        decisions — reviving a lost identity — the false-accept rate of a max
        over a large gallery is far too high. Averaging the best few matches
        keeps most of the viewpoint tolerance and rejects lucky single hits.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        query = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        gallery = self.matrix()
        if gallery.size == 0:
            return np.ones(query.shape[0], dtype=np.float64)
        if gallery.shape[1] != query.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: query {query.shape[1]} vs gallery {gallery.shape[1]}"
            )
        similarity = query @ gallery.T
        k = min(top_k, similarity.shape[1])
        best = np.sort(similarity, axis=1)[:, -k:]
        return 1.0 - best.mean(axis=1)

    def distance(self, embeddings: npt.ArrayLike) -> FloatArray:
        """Cosine distance from each of (N, D) ``embeddings`` to this gallery.

        Returns (N,) in [0, 2]; all-ones when the gallery is empty (no evidence
        either way, and the geometric term decides).
        """
        query = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        gallery = self.matrix()
        if gallery.size == 0:
            return np.ones(query.shape[0], dtype=np.float64)
        if gallery.shape[1] != query.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: query {query.shape[1]} vs gallery {gallery.shape[1]}"
            )
        similarity = query @ gallery.T  # both sides are unit vectors
        return 1.0 - similarity.max(axis=1)


@dataclass(frozen=True)
class AssociationConfig:
    """Gating and cost-blending parameters for cross-view association."""

    # Geometry: squared-Mahalanobis gate (chi-square, 2 DOF).
    chi2_gate: float = 9.2103
    # Hard cap on how far a measurement may sit from the prediction, whatever
    # the covariance says. Stops a wildly uncertain coasted track from swallowing
    # a detection on the far side of the room.
    max_distance_m: float = 2.5
    # Appearance: reject matches whose cosine distance exceeds this.
    # Sits between the measured same-identity cross-camera p95 (~0.32) and the
    # hardest different-identity confuser (~0.45) on the toy generator, which is
    # calibrated to published person-ReID operating points.
    max_appearance_distance: float = 0.40
    # Blend. Emphasis on appearance — single-target occlusion survival is the
    # ship criterion, and geometry goes uninformative exactly when it matters.
    weight_geometry: float = 0.4
    weight_appearance: float = 0.6
    # Reject any assignment whose blended cost exceeds this.
    max_cost: float = 0.85

    def __post_init__(self) -> None:
        if not np.isclose(self.weight_geometry + self.weight_appearance, 1.0):
            raise ValueError(
                f"weights must sum to 1, got {self.weight_geometry} + {self.weight_appearance}"
            )
        for name in ("chi2_gate", "max_distance_m", "max_appearance_distance", "max_cost"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        # Cosine distance between unit vectors is bounded by 2; a larger ceiling
        # silently disables the appearance gate entirely.
        if self.max_appearance_distance > 2.0:
            raise ValueError(
                f"max_appearance_distance must be <= 2 (the maximum cosine distance "
                f"between unit vectors); {self.max_appearance_distance} disables the gate"
            )
        # Gated-out cells carry INFEASIBLE; a ceiling at or above it would accept
        # every rejected pair and defeat every gate in the system at once.
        if self.max_cost >= INFEASIBLE:
            raise ValueError(
                f"max_cost must be < {INFEASIBLE} (the gated-out sentinel), "
                f"got {self.max_cost} — this would accept every rejected pair"
            )


def build_cost_matrix(
    mahalanobis_sq: FloatArray,
    euclidean_m: FloatArray,
    appearance: FloatArray,
    config: AssociationConfig,
) -> FloatArray:
    """Blend geometric and appearance costs, writing INFEASIBLE into gated cells.

    All three inputs are (n_observations, n_tracks).
    """
    shapes = {mahalanobis_sq.shape, euclidean_m.shape, appearance.shape}
    if len(shapes) != 1:
        raise ValueError(f"cost component shapes disagree: {shapes}")

    geo = np.clip(mahalanobis_sq / config.chi2_gate, 0.0, 1.0)
    app = np.clip(appearance / config.max_appearance_distance, 0.0, 1.0)
    cost = config.weight_geometry * geo + config.weight_appearance * app

    gated = (
        (mahalanobis_sq > config.chi2_gate)
        | (euclidean_m > config.max_distance_m)
        | (appearance > config.max_appearance_distance)
        | ~np.isfinite(cost)
    )
    cost = np.where(gated, INFEASIBLE, cost)
    return cost


def linear_assignment(
    cost: FloatArray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian solve with a cost ceiling.

    Returns (matches, unmatched_rows, unmatched_cols) where ``matches`` are
    (row, col) index pairs whose cost is <= ``max_cost``.
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {cost.shape}")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
