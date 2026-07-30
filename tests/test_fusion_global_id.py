"""Tests for mcreid.fusion.global_id — the global track lifecycle manager.

Uses hand-built `ViewObservation`s against a real `bedroom_rig()` calibration
(so `project_observations` exercises real geometry), never the toy scene
generator — that is covered separately by the integration tests.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.calib.schema import RigCalib
from mcreid.fusion.associate import AssociationConfig
from mcreid.fusion.dormant import DormantConfig
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import TrackState, ViewObservation
from mcreid.sim.toy import bedroom_rig
from mcreid.sim.virtual_camera import VirtualCamera

FloatArray = npt.NDArray[np.float64]
DT = 1.0 / 30.0
ROOM = (6.0, 5.0)


def _rig() -> RigCalib:
    return RigCalib(cameras=[c.to_calib(floor_extent_m=(0.0, 0.0, *ROOM)) for c in bedroom_rig()])


def _unit_embedding(dim: int, index: int) -> FloatArray:
    vec = np.zeros(dim)
    vec[index] = 1.0
    return vec


def _make_view(
    cam: VirtualCamera,
    foot_xy: tuple[float, float],
    embedding: FloatArray,
    frame: int,
    local_id: int = 0,
    height_m: float = 1.75,
    width_m: float = 0.55,
    score: float = 0.9,
) -> ViewObservation | None:
    box = cam.person_bbox(np.array(foot_xy), height_m, width_m)
    if box is None:
        return None
    return ViewObservation(
        camera_id=cam.camera_id,
        frame=frame,
        local_track_id=local_id,
        bbox_xyxy=box,
        embedding=embedding,
        score=score,
    )


def _views_for_all_cameras(
    cams: tuple[VirtualCamera, ...],
    foot_xy: tuple[float, float],
    embedding: FloatArray,
    frame: int,
    local_id: int = 0,
) -> list[ViewObservation]:
    out = []
    for cam in cams:
        view = _make_view(cam, foot_xy, embedding, frame, local_id=local_id)
        if view is not None:
            out.append(view)
    return out


# --- birth clustering: one person, many cameras, one frame -----------------------------


def test_single_person_seen_by_multiple_cameras_gets_one_global_id() -> None:
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    embed = _unit_embedding(16, 0)
    views = _views_for_all_cameras(cams, (3.0, 2.5), embed, frame=0)
    assert len(views) >= 3, "test setup should place the person in view of several cameras"

    mgr.step(views, frame=0, dt=DT)
    assert mgr.n_ids_issued == 1, "one person seen by several cameras must birth exactly one ID"


# --- id stability over steady motion -----------------------------------------------------


def test_id_is_stable_across_20_frames_of_steady_motion() -> None:
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    embed = _unit_embedding(16, 0)

    gids_seen: set[int] = set()
    for frame in range(20):
        foot = (2.5 + 0.02 * frame, 2.0 + 0.01 * frame)
        views = _views_for_all_cameras(cams, foot, embed, frame)
        snaps = mgr.step(views, frame, DT)
        gids_seen.update(s.global_id for s in snaps)

    assert gids_seen == {1}, f"expected a single stable global id, saw {gids_seen}"
    assert mgr.n_ids_issued == 1


# --- lifecycle: CONFIRMED -> COASTING -> LOST -> DEAD -------------------------------------


def test_lifecycle_confirmed_to_coasting_to_lost_to_dead_on_schedule() -> None:
    cams = bedroom_rig()
    cfg = FusionConfig(n_init=2, max_coast_frames=3, reid_window_frames=3)
    mgr = GlobalIDManager(_rig(), cfg)
    embed = _unit_embedding(16, 0)
    foot = (3.0, 2.5)

    # Two measured frames -> hits=2 -> CONFIRMED (n_init=2).
    for frame in range(2):
        views = _views_for_all_cameras(cams, foot, embed, frame)
        mgr.step(views, frame, DT)
    assert len(mgr.tracks) == 1
    assert mgr.tracks[0].state is TrackState.CONFIRMED

    # frames_since_measurement: 1, 2, 3 -> still COASTING (<= max_coast_frames=3).
    for frame in range(2, 5):
        mgr.step([], frame, DT)
        assert mgr.tracks[0].state is TrackState.COASTING, f"frame {frame}: expected COASTING"

    # frames_since_measurement becomes 4 (> 3) -> LOST.
    mgr.step([], 5, DT)
    assert mgr.tracks[0].state is TrackState.LOST

    # frames_since_measurement becomes 5 (> reid_window_frames=3) -> DEAD, and
    # DEAD tracks are pruned from the manager's list at the end of step().
    mgr.step([], 6, DT)
    assert mgr.tracks == [], "a track past its reid window must be dropped (DEAD)"
    assert mgr.n_ids_issued == 1, "IDs are never recycled"


# --- step() input validation --------------------------------------------------------------


def test_step_rejects_non_increasing_frame_numbers() -> None:
    mgr = GlobalIDManager(_rig())
    mgr.step([], 0, DT)
    mgr.step([], 1, DT)
    with pytest.raises(ValueError, match="frames must increase"):
        mgr.step([], 1, DT)
    with pytest.raises(ValueError, match="frames must increase"):
        mgr.step([], 0, DT)


def test_step_rejects_non_positive_dt() -> None:
    mgr = GlobalIDManager(_rig())
    with pytest.raises(ValueError, match="dt must be positive"):
        mgr.step([], 0, 0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        mgr.step([], 0, -0.01)


# --- two well-separated people never merge -------------------------------------------------


def test_two_well_separated_people_get_distinct_ids_and_never_merge() -> None:
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    embed_a = _unit_embedding(16, 0)
    embed_b = _unit_embedding(16, 1)
    foot_a = (1.0, 1.0)
    foot_b = (5.0, 4.0)

    all_gid_sets = []
    for frame in range(10):
        views = _views_for_all_cameras(cams, foot_a, embed_a, frame, local_id=1)
        views += _views_for_all_cameras(cams, foot_b, embed_b, frame, local_id=2)
        snaps = mgr.step(views, frame, DT)
        all_gid_sets.append({s.global_id for s in snaps})

    assert mgr.n_ids_issued == 2, "two well-separated people must birth exactly two IDs"
    confirmed_sets = [s for s in all_gid_sets if s]
    assert confirmed_sets, "tracks should be confirmed and visible before frame 10"
    for gids in confirmed_sets:
        assert gids == {1, 2}, f"expected both ids visible together, got {gids}"


# --- FusionConfig validation ------------------------------------------------------------------


def test_fusion_config_rejects_reid_window_shorter_than_max_coast() -> None:
    with pytest.raises(ValueError, match="reid_window_frames"):
        FusionConfig(max_coast_frames=90, reid_window_frames=50)


def test_fusion_config_rejects_revive_distance_looser_than_association_gate() -> None:
    assoc = AssociationConfig(
        max_appearance_distance=0.30, weight_geometry=0.4, weight_appearance=0.6
    )
    with pytest.raises(ValueError, match="revive_appearance_distance"):
        FusionConfig(association=assoc, revive_appearance_distance=0.40)


def test_fusion_config_rejects_bad_lifecycle_params() -> None:
    with pytest.raises(ValueError, match="n_init"):
        FusionConfig(n_init=0)
    with pytest.raises(ValueError, match="max_coast_frames"):
        FusionConfig(max_coast_frames=0)


# --- the leave/return cascade: one missed resurrection must not deadlock -----------------


def _visit(
    mgr: GlobalIDManager,
    cams: tuple[VirtualCamera, ...],
    embedding: FloatArray,
    start: int,
    frames: int,
    foot_xy: tuple[float, float] = (2.5, 2.5),
) -> int:
    """Someone stands in the room for `frames` frames. Returns the next frame index."""
    for offset in range(frames):
        frame = start + offset
        mgr.step(_views_for_all_cameras(cams, foot_xy, embedding, frame), frame, DT)
    return start + frames


def _absence(mgr: GlobalIDManager, start: int, frames: int) -> int:
    """Nobody in view: long enough for the track to coast, go lost, and retire."""
    for offset in range(frames):
        frame = start + offset
        mgr.step([], frame, DT)
    return start + frames


def _split_pair(dim: int, separation: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Two appearances of one person `separation` apart, and their midpoint."""
    first, axis = np.zeros(dim), np.zeros(dim)
    first[0], axis[1] = 1.0, 1.0
    cosine = 1.0 - separation
    second = cosine * first + np.sqrt(1.0 - cosine**2) * axis
    midpoint = first + second
    return first, second, midpoint / np.linalg.norm(midpoint)


def _leave_return_cascade(near_miss_margin: float) -> tuple[GlobalIDManager, list[int]]:
    """Sit / leave / return-and-be-missed / leave / return again.

    The appearance on the second visit sits at 0.45 from the stored vector —
    just past the 0.42 dormant gate, which is what the real webcam run did — so
    the first return is *missed* and mints a fresh ID. That failure is what puts
    a second copy of this person in the gallery. The third visit is then the one
    that must still recover the original identity.
    """
    cams = bedroom_rig()
    config = FusionConfig(dormant=DormantConfig(near_miss_margin=near_miss_margin))
    mgr = GlobalIDManager(_rig(), config)
    first, second, midpoint = _split_pair(16, separation=0.45)

    # Absence must outlast coasting *and* the revive window, so the track truly
    # retires into the dormant gallery instead of being revived by motion.
    gone = config.max_coast_frames + config.reid_window_frames + 10

    frame = _visit(mgr, cams, first, start=0, frames=40)
    frame = _absence(mgr, frame, gone)
    ids_after_first_absence = mgr.dormant.ids

    frame = _visit(mgr, cams, second, start=frame, frames=20)
    frame = _absence(mgr, frame, gone)

    frame = _visit(mgr, cams, midpoint, start=frame, frames=20)
    live = [t.global_id for t in mgr.tracks if t.is_visible]
    assert ids_after_first_absence == [1], "the first identity must reach the gallery"
    return mgr, live


def test_a_missed_resurrection_deadlocks_the_gallery_without_provenance() -> None:
    """The bug, reproduced: proves the test below is testing something real."""
    mgr, live = _leave_return_cascade(near_miss_margin=0.0)

    assert mgr.dormant.n_resurrected == 0, "the deadlock means nothing is ever recovered"
    assert live and 1 not in live, f"the original identity stays lost, got {live}"
    assert mgr.dormant.n_rejected_ambiguous >= 1, (
        "and it is the ratio test doing it — two stored copies of the same person"
    )
    assert len(mgr.dormant) == 2, "both rival records are on file, which is the cause"


def test_the_original_identity_survives_a_missed_resurrection() -> None:
    """The fix: the third visit recovers ID 1, not a fourth new number."""
    mgr, live = _leave_return_cascade(near_miss_margin=0.10)

    assert mgr.dormant.n_suppressed_duplicates == 1, (
        "visit 2's identity must never have been stored as a rival record"
    )
    assert mgr.dormant.n_resurrected == 1
    assert live == [1], f"the returning person must carry the original id, got {live}"


def test_recovery_destroys_no_stored_identity() -> None:
    """Nothing existing may be merged, renamed or deleted to make this work.

    Two earlier attempts at this failed exactly here: one merged the two entries
    (which on real crops fuses two different people 40.5% of the time it fires),
    the other let a near-miss *assign* an identity (wrong person 45% of the time
    in a two-entry gallery). Suppression only ever withholds a new record, so the
    worst case is forgetting someone, never renaming them.
    """
    mgr, _live = _leave_return_cascade(near_miss_margin=0.10)
    assert mgr.dormant.n_admitted == 1, "exactly one record was ever stored"
    assert mgr.dormant.n_resurrected == 1, "and it was handed back to its owner"


def test_an_adopted_track_stops_shopping_the_gallery() -> None:
    """A track that recovered an identity must not hop to another record of it.

    An adopted track stays TENTATIVE until it earns confirmation, so without a
    guard it probes again every frame of that window. If a second record of the
    same person is on file for any reason — suppression disabled, or the origin
    expired between visits — it trades the identity it just correctly recovered
    for the duplicate, which recovers the ID switch without undoing it.

    ``n_init=5`` is deliberate: at the shipped ``n_init=3`` an adopted track
    confirms on the next measured frame, so the window closes by arithmetic
    rather than by the guard, and the test would pass with the guard removed.
    """
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig(), FusionConfig(n_init=5))
    person = _unit_embedding(16, 0)

    mgr.dormant.admit(41, person[None, :], frame=0, hits=500)
    frame = _visit(mgr, cams, person, start=1, frames=2)
    adopted = [t.global_id for t in mgr.tracks]
    assert adopted == [41], f"the track should have adopted id 41, got {adopted}"
    assert mgr.dormant.n_resurrected == 1

    # A rival record of the same person appears while the track is still tentative.
    mgr.dormant.admit(77, person[None, :], frame=frame, hits=500)
    assert any(t.state is TrackState.TENTATIVE for t in mgr.tracks), (
        "test setup: the adopted track must still be tentative for this to bite"
    )
    _visit(mgr, cams, person, start=frame, frames=2)

    assert [t.global_id for t in mgr.tracks] == [41], "the adopted identity must stick"
    assert mgr.dormant.n_resurrected == 1, "and no second identity may be consumed"
    assert 77 in mgr.dormant


# --- probe instrumentation (D1): a rejection is only actionable with provenance ----------


def test_a_long_gap_probe_records_its_provenance() -> None:
    """Every dormant probe must say what it was made from, not just its distance.

    The path that fires on a return is `_resurrect`, which probes from a birth
    cluster rather than from a track. It was shipped without provenance while the
    rarely-taken tentative-track path had it, so the live log showed a bare
    distance and could not distinguish a too-tight gate from a half-body query.
    """
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    person = _unit_embedding(16, 0)

    frame = _visit(mgr, cams, person, start=0, frames=40)
    frame = _absence(mgr, frame, mgr.config.max_coast_frames + mgr.config.reid_window_frames + 10)
    assert mgr.dormant.ids == [1], "test setup: the identity must be dormant"

    _visit(mgr, cams, person, start=frame, frames=3)

    assert mgr.dormant.attempts, "the return must have produced a recorded probe"
    context = mgr.dormant.attempts[0].context
    assert context, "a probe with no provenance is the defect this test guards"
    for token in ("camera", "truncated", "sigma"):
        assert token in context, f"provenance must report {token!r}, got {context!r}"


def test_a_clipped_box_is_flagged_truncated() -> None:
    """The half-body signal must survive into the observation, not just the covariance.

    A clipped box already inflates the positional covariance, but a covariance
    says nothing about whether the *appearance* vector is trustworthy, and that
    is what a long-gap probe rides on.
    """
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    cam = cams[0]
    width, height = cam.to_calib().intrinsics.image_size
    embed = _unit_embedding(16, 0)

    whole = ViewObservation(
        camera_id=cam.camera_id,
        frame=0,
        local_track_id=1,
        bbox_xyxy=np.array([width * 0.4, height * 0.3, width * 0.5, height * 0.6]),
        embedding=embed,
        score=0.9,
    )
    clipped = ViewObservation(
        camera_id=cam.camera_id,
        frame=0,
        local_track_id=2,
        # Runs off the bottom edge: the feet are outside the frame entirely.
        bbox_xyxy=np.array([width * 0.4, height * 0.5, width * 0.5, float(height)]),
        embedding=embed,
        score=0.9,
    )

    by_track = {o.local_track_id: o for o in mgr.project_observations([whole, clipped], frame=0)}
    assert by_track[1].truncated is False, "a box inside the frame is not truncated"
    assert by_track[2].truncated is True, "a box clipped by the frame edge must be flagged"
    assert by_track[2].position_sigma_m > by_track[1].position_sigma_m
