"""Identity-consistency metrics — the numbers the cardboard test is graded on.

An ID switch is counted only when a ground-truth person who *was* being tracked
under global id A is later tracked under global id B. Frames where the person is
unmatched (fully occluded, or the tracker lost them) do not themselves count as
switches — they are reported separately as coverage and gap statistics, so a
tracker cannot buy a clean ID-switch score by refusing to output anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from mcreid.fusion.types import GlobalTrackSnapshot
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

DEFAULT_MATCH_RADIUS_M = 1.0


@dataclass
class IdConsistencyReport:
    """Per-run identity metrics."""

    n_frames: int
    n_gt_agents: int
    n_ids_issued: int
    id_switches: dict[int, int]
    """gt_agent_id -> number of global-ID changes."""
    ids_per_agent: dict[int, list[int]]
    """gt_agent_id -> ordered distinct global IDs it was tracked under."""
    coverage: dict[int, float]
    """gt_agent_id -> fraction of frames the agent exists in which it was matched
    to a global track. Coasted matches during total occlusion count, so a value
    above the visible-frame fraction means the tracker successfully carried the
    identity through a blackout."""
    coverage_visible: dict[int, float]
    """gt_agent_id -> same ratio restricted to frames where at least one camera
    could actually see the agent. This is the honest detection-recall figure."""
    position_rmse_m: float
    mean_position_error_m: float
    longest_coast_survived: dict[int, int]
    """gt_agent_id -> longest total-occlusion run it kept its ID across."""
    false_positive_tracks: int
    """Confirmed global tracks that never matched any ground-truth agent."""
    per_frame_matches: list[dict[int, int]] = field(default_factory=list, repr=False)

    @property
    def total_id_switches(self) -> int:
        return sum(self.id_switches.values())

    def summary(self) -> str:
        lines = [
            f"frames={self.n_frames}  gt_agents={self.n_gt_agents}  "
            f"global_ids_issued={self.n_ids_issued}",
            f"ID switches: {self.total_id_switches} {dict(self.id_switches)}",
            f"position RMSE: {self.position_rmse_m:.3f} m  "
            f"(mean {self.mean_position_error_m:.3f} m)",
            f"coverage (all frames): { {k: round(v, 3) for k, v in self.coverage.items()} }",
            f"coverage (visible only): "
            f"{ {k: round(v, 3) for k, v in self.coverage_visible.items()} }",
            f"longest occlusion survived (frames): {dict(self.longest_coast_survived)}",
            f"false-positive tracks: {self.false_positive_tracks}",
        ]
        return "\n".join(lines)


def _match_frame(
    gt_positions: dict[int, FloatArray],
    snapshots: list[GlobalTrackSnapshot],
    match_radius_m: float,
) -> dict[int, int]:
    """Hungarian-match ground-truth agents to global tracks. Returns {agent: gid}."""
    if not gt_positions or not snapshots:
        return {}

    agents = sorted(gt_positions)
    gt = np.stack([gt_positions[a] for a in agents])
    pred = np.stack([s.world_xy for s in snapshots])
    distance = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    cost = np.where(distance <= match_radius_m, distance, 1e5)

    rows, cols = linear_sum_assignment(cost)
    return {
        agents[r]: snapshots[c].global_id
        for r, c in zip(rows, cols, strict=True)
        if distance[r, c] <= match_radius_m
    }


def evaluate_id_consistency(
    gt_world: dict[int, FloatArray],
    gt_visible: dict[int, npt.NDArray[np.bool_]],
    results: list[list[GlobalTrackSnapshot]],
    n_ids_issued: int,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
) -> IdConsistencyReport:
    """Score a run.

    Args:
        gt_world: agent_id -> (n_frames, 2) ground-truth floor positions.
        gt_visible: agent_id -> (n_frames, n_cameras) per-camera visibility.
        results: per-frame lists of visible global-track snapshots.
        n_ids_issued: total global IDs minted by the manager.
        match_radius_m: a prediction farther than this is not a match at all.
    """
    n_frames = len(results)
    agents = sorted(gt_world)
    for agent in agents:
        if gt_world[agent].shape[0] != n_frames:
            raise ValueError(
                f"agent {agent}: ground truth has {gt_world[agent].shape[0]} frames "
                f"but results have {n_frames}"
            )

    current_id: dict[int, int | None] = dict.fromkeys(agents)
    switches: dict[int, int] = dict.fromkeys(agents, 0)
    id_history: dict[int, list[int]] = {a: [] for a in agents}
    matched_frames: dict[int, int] = dict.fromkeys(agents, 0)
    matched_visible_frames: dict[int, int] = dict.fromkeys(agents, 0)
    present_frames: dict[int, int] = dict.fromkeys(agents, 0)
    visible_frames: dict[int, int] = dict.fromkeys(agents, 0)
    errors: list[float] = []
    per_frame: list[dict[int, int]] = []
    matched_gids: set[int] = set()

    # Longest run of "invisible in every camera" that the agent came out of
    # still holding the same global ID. The run is measured over the blackout
    # itself, independently of whether the coasted track happened to stay within
    # the match radius mid-blackout — otherwise a successful coast would reset
    # its own counter and under-report the headline number.
    blackout_run: dict[int, int] = dict.fromkeys(agents, 0)
    id_before_blackout: dict[int, int | None] = dict.fromkeys(agents)
    best_coast: dict[int, int] = dict.fromkeys(agents, 0)

    for frame in range(n_frames):
        gt_positions = {
            a: gt_world[a][frame]
            for a in agents
            if np.isfinite(gt_world[a][frame]).all()
        }
        snapshots = results[frame]
        assignment = _match_frame(gt_positions, snapshots, match_radius_m)
        per_frame.append(assignment)
        matched_gids.update(assignment.values())

        by_gid = {s.global_id: s for s in snapshots}
        for agent in agents:
            any_view = bool(gt_visible[agent][frame].any()) if agent in gt_visible else True
            present = agent in gt_positions
            if present:
                present_frames[agent] += 1
                if any_view:
                    visible_frames[agent] += 1

            # Accumulate the total-occlusion run independently of matching.
            if present and not any_view:
                if blackout_run[agent] == 0:
                    id_before_blackout[agent] = current_id[agent]
                blackout_run[agent] += 1

            gid = assignment.get(agent)
            if gid is None:
                continue

            matched_frames[agent] += 1
            if any_view:
                matched_visible_frames[agent] += 1
            errors.append(float(np.linalg.norm(gt_positions[agent] - by_gid[gid].world_xy)))

            previous = current_id[agent]
            if previous is None:
                current_id[agent] = gid
                id_history[agent].append(gid)
            elif gid != previous:
                switches[agent] += 1
                current_id[agent] = gid
                id_history[agent].append(gid)

            # Credit a survived blackout only once the agent is genuinely
            # visible again and is still carrying the identity it went in with.
            if any_view and blackout_run[agent] > 0:
                if id_before_blackout[agent] is not None and gid == id_before_blackout[agent]:
                    best_coast[agent] = max(best_coast[agent], blackout_run[agent])
                blackout_run[agent] = 0
                id_before_blackout[agent] = None

    error_array = np.asarray(errors, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(error_array**2))) if error_array.size else float("nan")
    mean_error = float(error_array.mean()) if error_array.size else float("nan")

    confirmed_gids = {s.global_id for frame_result in results for s in frame_result}
    false_positive_tracks = len(confirmed_gids - matched_gids)

    coverage = {
        a: (matched_frames[a] / present_frames[a]) if present_frames[a] else float("nan")
        for a in agents
    }
    coverage_visible = {
        a: (matched_visible_frames[a] / visible_frames[a]) if visible_frames[a] else float("nan")
        for a in agents
    }

    return IdConsistencyReport(
        n_frames=n_frames,
        n_gt_agents=len(agents),
        n_ids_issued=n_ids_issued,
        id_switches=switches,
        ids_per_agent=id_history,
        coverage=coverage,
        coverage_visible=coverage_visible,
        position_rmse_m=rmse,
        mean_position_error_m=mean_error,
        longest_coast_survived=best_coast,
        false_positive_tracks=false_positive_tracks,
        per_frame_matches=per_frame,
    )
