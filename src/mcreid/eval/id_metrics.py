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
    longest_blackout_id_held: dict[int, int]
    """gt_agent_id -> longest total-occlusion run after which it still carried the
    same global ID.

    This says nothing about whether the tracker was *alive* during the blackout:
    a tracker that emits nothing for the whole occlusion and then re-emits the
    same ID scores full marks here. Always read it next to
    ``longest_blackout_coasted``."""
    longest_blackout_alive: dict[int, int]
    """gt_agent_id -> longest total-occlusion run during which the pre-blackout
    global ID was still being emitted at all: the BEV dot did not vanish.

    Together with ``longest_blackout_id_held`` this is what the demo claims.
    Neither can be earned by going silent."""
    longest_blackout_coasted: dict[int, int]
    """gt_agent_id -> longest run during which the emitted prediction was also
    within the match radius of the truth — the honest accuracy figure. Expect
    this to be well short of the full blackout: constant-velocity coasting
    degrades, and the identity is ultimately recovered by ReID re-lock."""
    blackout_position_error_m: float
    """Mean ground-plane error of coasted predictions during total occlusion —
    how far the BEV dot drifts while nothing can see the target. NaN if the
    tracker never emitted anything during a blackout."""
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
            f"blackout: ID held {dict(self.longest_blackout_id_held)} frames, "
            f"dot alive {dict(self.longest_blackout_alive)}, "
            f"within match radius {dict(self.longest_blackout_coasted)}",
            f"coasted position error: {self.blackout_position_error_m:.3f} m",
            f"false-positive tracks: {self.false_positive_tracks}",
        ]
        return "\n".join(lines)


def _match_frame(
    gt_positions: dict[int, FloatArray],
    snapshots: list[GlobalTrackSnapshot],
    match_radius_m: float,
    hidden_agents: set[int] | None = None,
) -> dict[int, int]:
    """Hungarian-match ground-truth agents to global tracks. Returns {agent: gid}.

    ``hidden_agents`` are agents no camera can see this frame. They may only be
    matched to a track that is itself *coasting*: a track being actively measured
    is demonstrably following somebody the cameras can see, so it cannot also be
    following someone invisible. Without this rule a hidden agent who happens to
    walk past a stranger's track gets credited to it, which shows up as a
    spurious ID switch that no tracker behaviour caused.
    """
    if not gt_positions or not snapshots:
        return {}

    hidden = hidden_agents or set()
    agents = sorted(gt_positions)
    gt = np.stack([gt_positions[a] for a in agents])
    pred = np.stack([s.world_xy for s in snapshots])
    distance = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    cost = np.where(distance <= match_radius_m, distance, 1e5)

    measured = np.array([not s.is_coasting for s in snapshots], dtype=bool)
    for row, agent in enumerate(agents):
        if agent in hidden:
            cost[row, measured] = 1e5
            distance[row, measured] = np.inf

    rows, cols = linear_sum_assignment(cost)
    return {
        agents[r]: snapshots[c].global_id
        for r, c in zip(rows, cols, strict=True)
        if distance[r, c] <= match_radius_m
    }


def _blackout_intervals(visible: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of frames where no camera can see the agent."""
    hidden = ~visible.any(axis=1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for frame, flag in enumerate(hidden):
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            runs.append((start, frame))
            start = None
    if start is not None:
        runs.append((start, len(hidden)))
    return runs


def _score_blackouts(
    agent: int,
    visible: npt.NDArray[np.bool_],
    per_frame: list[dict[int, int]],
    gt_world: FloatArray,
    results: list[list[GlobalTrackSnapshot]],
    regain_window: int,
) -> tuple[int, int, int, list[float]]:
    """Score every total-occlusion interval for one agent.

    Returns (longest_id_held, longest_alive, longest_coasted, coasted_errors).

    - *id held*: the tracker re-acquires the agent within ``regain_window``
      frames of the blackout ending, under the same global ID it went in with.
      Taking a frame or two to re-lock still counts as holding the identity.
    - *alive*: the longest run inside the blackout during which that ID was
      still being emitted at all — i.e. the BEV dot did not disappear. This is
      the "dot persists" claim, and it is deliberately separate from accuracy.
    - *coasted*: the longest run inside the blackout during which the emitted
      prediction was also **within the match radius** of the truth. This is the
      honest accuracy figure, and unlike *id held* neither can be earned by
      going silent.
    """
    longest_held = 0
    longest_alive = 0
    longest_coasted = 0
    errors: list[float] = []

    for start, end in _blackout_intervals(visible):
        before: int | None = None
        for frame in range(start - 1, -1, -1):
            if agent in per_frame[frame]:
                before = per_frame[frame][agent]
                break
        if before is None:
            continue  # never tracked going in; nothing to hold

        run = 0
        alive_run = 0
        for frame in range(start, end):
            snap = next((s for s in results[frame] if s.global_id == before), None)
            if snap is None:
                alive_run = 0
            else:
                alive_run += 1
                longest_alive = max(longest_alive, alive_run)
                if np.isfinite(gt_world[frame]).all():
                    errors.append(float(np.linalg.norm(gt_world[frame] - snap.world_xy)))

            if per_frame[frame].get(agent) == before:
                run += 1
                longest_coasted = max(longest_coasted, run)
            else:
                run = 0

        for frame in range(end, min(end + regain_window, len(per_frame))):
            gid = per_frame[frame].get(agent)
            if gid is None:
                continue
            if gid == before:
                longest_held = max(longest_held, end - start)
            break  # the first re-acquisition decides it

    return longest_held, longest_alive, longest_coasted, errors


def evaluate_id_consistency(
    gt_world: dict[int, FloatArray],
    gt_visible: dict[int, npt.NDArray[np.bool_]],
    results: list[list[GlobalTrackSnapshot]],
    n_ids_issued: int,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    regain_window: int = 5,
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


    for frame in range(n_frames):
        gt_positions = {
            a: gt_world[a][frame]
            for a in agents
            if np.isfinite(gt_world[a][frame]).all()
        }
        snapshots = results[frame]
        hidden_agents = {
            a
            for a in gt_positions
            if a in gt_visible and not bool(gt_visible[a][frame].any())
        }
        assignment = _match_frame(gt_positions, snapshots, match_radius_m, hidden_agents)
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

    best_id_held: dict[int, int] = dict.fromkeys(agents, 0)
    best_alive: dict[int, int] = dict.fromkeys(agents, 0)
    best_coasted: dict[int, int] = dict.fromkeys(agents, 0)
    blackout_errors: list[float] = []
    for agent in agents:
        if agent not in gt_visible:
            continue
        held, alive, coasted, agent_errors = _score_blackouts(
            agent=agent,
            visible=gt_visible[agent],
            per_frame=per_frame,
            gt_world=gt_world[agent],
            results=results,
            regain_window=regain_window,
        )
        best_id_held[agent] = held
        best_alive[agent] = alive
        best_coasted[agent] = coasted
        blackout_errors.extend(agent_errors)

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
        longest_blackout_id_held=best_id_held,
        longest_blackout_alive=best_alive,
        longest_blackout_coasted=best_coasted,
        blackout_position_error_m=(
            float(np.mean(blackout_errors)) if blackout_errors else float("nan")
        ),
        false_positive_tracks=false_positive_tracks,
        per_frame_matches=per_frame,
    )
