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


def test_the_tentative_track_probe_also_reports_truncation() -> None:
    """Both probe paths must be interpretable, not just the cluster one.

    Cycle 2 produced a rejection from each path. The cluster probe named its
    truncated count and was diagnosable; the track probe did not, so an
    identical-looking 0.610 rejection could not be attributed. A probe that
    cannot be attributed is the same defect D1 exists to remove, one path over.
    """
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    person = _unit_embedding(16, 0)
    other = _unit_embedding(16, 1)

    frame = _visit(mgr, cams, person, start=0, frames=40)
    frame = _absence(mgr, frame, mgr.config.max_coast_frames + mgr.config.reid_window_frames + 10)
    assert mgr.dormant.ids == [1]

    # A different-looking person arrives, so the birth cluster is refused and a
    # tentative track exists to take the second probe.
    _visit(mgr, cams, other, start=frame, frames=3)

    track_probes = [
        a for a in mgr.dormant.attempts if "tentative track" in a.context
    ]
    assert track_probes, "the tentative-track path must have probed and been recorded"
    context = track_probes[0].context
    for token in ("hits", "truncated", "sigma"):
        assert token in context, f"track provenance must report {token!r}, got {context!r}"


def test_truncation_counts_reach_the_track_from_its_measurements() -> None:
    """The count must come from real observations, not be hardcoded to zero."""
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    cam = cams[0]
    width, height = cam.to_calib().intrinsics.image_size
    embed = _unit_embedding(16, 0)

    clipped = ViewObservation(
        camera_id=cam.camera_id,
        frame=0,
        local_track_id=1,
        bbox_xyxy=np.array([width * 0.4, height * 0.5, width * 0.5, float(height)]),
        embedding=embed,
        score=0.9,
    )
    mgr.step([clipped], 0, DT)

    assert len(mgr.tracks) == 1
    assert mgr.tracks[0].last_observations == 1
    assert mgr.tracks[0].last_truncated == 1, "a clipped measurement must be counted"


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


# --- n_init: two jobs, both pinned -------------------------------------------
#
# n_init is overloaded. It is the false-positive filter AND, because
# _adopt_dormant_identity only considers tracks that are still TENTATIVE with
# hits >= 2, it silently sets the dormant-adoption window to `hits in
# [2, n_init)`. Shadow session s1 measured what that costs: recovery of the four
# recorded track-EMA gate rejections by window alone was 1/4 at n_init=3, 2/4 at
# 5, 3/4 at 8, 4/4 at 10. The retry policy now owns recovery, which is what frees
# n_init to be chosen as a filter — but the coupling still exists, so it is
# pinned here rather than left to be rediscovered by a third live session.


@pytest.mark.parametrize("n_init", [2, 3, 5, 8, 10])
def test_adoption_window_is_hits_2_to_n_init(n_init: int) -> None:
    """A track must stay adoptable for every hit in [2, n_init).

    If a future change shrinks this — by confirming earlier, or by raising the
    hits>=2 floor — long-gap recall drops silently, because nothing downstream
    reports "the identity was recoverable but the window had closed".
    """
    cams = bedroom_rig()
    cfg = FusionConfig(n_init=n_init)
    mgr = GlobalIDManager(_rig(), cfg)
    embed = _unit_embedding(16, 0)

    adoptable = 0
    for frame in range(n_init + 4):
        views = _views_for_all_cameras(cams, (3.0, 2.5), embed, frame)
        mgr.step(views, frame, DT)
        for track in mgr.tracks:
            if track.state is TrackState.TENTATIVE and track.hits >= 2:
                adoptable += 1

    expected = max(n_init - 2, 0)
    assert adoptable == expected, (
        f"n_init={n_init} should leave {expected} adoptable frame(s), got {adoptable}. "
        "The dormant-adoption window is hits in [2, n_init) — see FusionConfig.n_init."
    )


def test_n_init_default_is_5_and_kills_every_flicker_measured_in_s1() -> None:
    """The four flicker mints in shadow session s1 reached 1, 2, 3 and 4 hits.

    Tracks 5, 3, 4 and 10 respectively. The next track up survived 24 hits, so
    nothing between 5 and 24 is distinguishable from that session — 5 is the
    smallest value that kills all four, and any larger choice buys confirmation
    latency for no measured benefit.
    """
    assert FusionConfig().n_init == 5

    s1_flicker_hits = {"track 5": 1, "track 3": 2, "track 4": 3, "track 10": 4}
    for name, hits in s1_flicker_hits.items():
        assert hits < FusionConfig().n_init, f"{name} ({hits} hits) would still confirm"

    # And the honest limit of this parameter: the phantom that actually polluted
    # the s1 gallery reached 24 hits and retired as a rival record. No n_init in
    # any sane range reaches it — only the retry does.
    assert FusionConfig().n_init < 24, (
        "s1's rival-record track is out of n_init's reach by construction; the "
        "retry policy is what covers that case"
    )


def test_retry_ships_off_by_default() -> None:
    """OFF by default, on measured evidence — see DormantConfig.retry_offsets.

    Adversarial review measured the retry buying +2.0 to +3.4 points of identity
    theft for +0.0 points of owner recall on real WILDTRACK crops, and showed a
    stranger taking a 500-frame-old confirmed identity through the entry-keyed
    version of the schedule. It is available under --single-occupant, where the
    harm provably cannot occur, and nowhere else until a second live session.
    """
    mgr = GlobalIDManager(_rig())
    assert mgr.dormant.config.retry_offsets == (), "retry must not ship on by default"
    assert mgr.dormant.config.appearance_distance == pytest.approx(0.42), (
        "the gate is unchanged — the retry re-asks, it does not lower the bar"
    )
    assert mgr.dormant.config.ratio_test == pytest.approx(0.85), "ratio test unchanged"


# --- retry scoping: the defects adversarial review found in the first version ---
#
# Every test below fails against a specific degenerate implementation that the
# original test suite accepted. The stub each one kills is named in its docstring.


def _dormant_manager(offsets: tuple[int, ...] = (4, 9)) -> GlobalIDManager:
    cfg = FusionConfig(dormant=DormantConfig(retry_offsets=offsets, min_hits=1))
    return GlobalIDManager(_rig(), cfg)


def _query_at(distance: float, dim: int = 16) -> FloatArray:
    """A unit query exactly ``distance`` from `_unit_embedding(dim, 0)`.

    Built analytically: a near miss has to land in a narrow band (outside the
    0.42 gate, inside the 0.483 arming ceiling) and a hand-mixed vector silently
    lands inside the gate instead, testing nothing.
    """
    sim = 1.0 - distance
    vec = sim * _unit_embedding(dim, 0) + np.sqrt(max(1.0 - sim**2, 0.0)) * _unit_embedding(dim, 1)
    return vec / np.linalg.norm(vec)


NEAR_MISS = 0.45  # outside the 0.42 gate, inside the 0.42*1.15 arming ceiling


def test_a_retry_is_scoped_to_the_track_that_earned_it() -> None:
    """Kills the stub where `retries_due` returns bare ids (stub S7).

    The first version keyed schedules by dormant entry alone and treated "any
    retry due" as a global switch, so a DIFFERENT person walking in could spend
    the exemption and take the dormant identity. Review demonstrated exactly that
    on the full pipeline. The schedule is now a (entry, track) pair.
    """
    mgr = _dormant_manager()
    embed = _unit_embedding(16, 0)
    mgr.dormant.admit(1, embed[None, :], frame=0, hits=50)

    # Track 42 misses by a hair and arms a retry against entry 1.
    near = _query_at(NEAR_MISS)
    mgr.dormant.match(near[None, :])
    assert mgr.dormant.schedule_retries(100, owners={0: 42}) == 1

    due = mgr.dormant.retries_due(104)
    assert due == {(1, 42)}, f"the exemption belongs to track 42 alone, got {due}"
    assert all(track_id == 42 for _gid, track_id in due), (
        "no other track may consume this retry"
    )


def test_arming_requires_a_near_miss_not_any_miss() -> None:
    """Kills the stub with no distance ceiling (review defect D3).

    The first version armed on ANY gate rejection, including one measured at
    d=1.284 — nearly antipodal, unambiguously a different person. Every arrival
    then opened the gallery to every candidate track for two frames.
    """
    mgr = _dormant_manager()
    mgr.dormant.admit(1, _unit_embedding(16, 0)[None, :], frame=0, hits=50)

    stranger = _unit_embedding(16, 7)  # orthogonal -> distance 1.0
    mgr.dormant.match(stranger[None, :])
    assert mgr.dormant.schedule_retries(100, owners={0: 9}) == 0, (
        "a miss by a mile is a different person, not an early query"
    )
    assert mgr.dormant.retries_due(104) == set()


def test_an_unowned_probe_never_arms() -> None:
    """A schedule with no owner would be spendable by anyone — refuse to make one.

    This is why the birth-cluster path no longer arms: a cluster has no track
    behind it yet, so there is nobody to scope the exemption to.
    """
    mgr = _dormant_manager()
    mgr.dormant.admit(1, _unit_embedding(16, 0)[None, :], frame=0, hits=50)
    mgr.dormant.match(_query_at(NEAR_MISS)[None, :])
    assert mgr.dormant.schedule_retries(100, owners=None) == 0
    assert mgr.dormant.schedule_retries(100, owners={}) == 0


def test_the_bound_is_once_per_pair_not_a_sliding_window() -> None:
    """Kills the stub whose re-arm horizon lets it fire forever (stub S2).

    The first version blocked re-arming only while `frame - ended <= horizon`, so
    it re-armed on a fixed 19-frame period: 16 schedules and 32 retries over 300
    frames of continuous rejection. That is "keep asking until the gate lets
    something through", which is what the docstring claimed it was not.
    """
    mgr = _dormant_manager()
    mgr.dormant.admit(1, _unit_embedding(16, 0)[None, :], frame=0, hits=50)
    near = _query_at(NEAR_MISS)

    for frame in range(0, 300):
        mgr.dormant.match(near[None, :])
        mgr.dormant.schedule_retries(frame, owners={0: 5})
        mgr.dormant.retries_due(frame)

    assert mgr.dormant.n_retries_scheduled == 1, (
        f"one pair, one presence, one schedule — got "
        f"{mgr.dormant.n_retries_scheduled} over 300 frames of rejection"
    )
    assert mgr.dormant.n_retries_fired <= 2, "and at most the two offsets"


def test_cancel_retries_is_observed_directly_not_through_retries_due() -> None:
    """Kills the no-op `cancel_retries` stub (stub S5).

    The old cancellation tests asserted through `retries_due`, which independently
    checks `gid in self._entries` — so they passed with cancellation removed
    entirely. Assert the state itself.
    """
    mgr = _dormant_manager()
    mgr.dormant.admit(1, _unit_embedding(16, 0)[None, :], frame=0, hits=50)
    mgr.dormant.match(_query_at(NEAR_MISS)[None, :])
    mgr.dormant.schedule_retries(100, owners={0: 3})

    assert any(k[0] == 1 for k in mgr.dormant._retry_due)
    mgr.dormant.cancel_retries(1)
    assert not any(k[0] == 1 for k in mgr.dormant._retry_due), "pending state must be gone"
    assert not any(k[0] == 1 for k in mgr.dormant._retry_armed), "and the armed record too"


def test_capacity_eviction_cancels_retry_state() -> None:
    """Review defect D7: eviction dropped the entry but kept its retry state,
    which then blocked the same id from ever arming again after re-admission."""
    cfg = FusionConfig(dormant=DormantConfig(retry_offsets=(4, 9), max_entries=2, min_hits=1))
    mgr = GlobalIDManager(_rig(), cfg)
    for gid in (1, 2):
        mgr.dormant.admit(gid, _unit_embedding(16, gid)[None, :], frame=gid, hits=50)
    sim = 1.0 - NEAR_MISS
    near = sim * _unit_embedding(16, 1) + np.sqrt(1.0 - sim**2) * _unit_embedding(16, 9)
    mgr.dormant.match((near / np.linalg.norm(near))[None, :])
    mgr.dormant.schedule_retries(100, owners={0: 8})

    mgr.dormant.admit(3, _unit_embedding(16, 3)[None, :], frame=200, hits=50)  # evicts id 1
    assert 1 not in mgr.dormant
    assert not any(k[0] == 1 for k in mgr.dormant._retry_due), "eviction must cancel too"
    assert not any(k[0] == 1 for k in mgr.dormant._retry_armed)


def test_a_retry_adoption_is_a_VISIBLE_id_switch_in_step_output() -> None:
    """The cost of the retry, made explicit and observable.

    Adopting while TENTATIVE is invisible downstream — the candidate id was never
    reported to anyone. Adopting on a retry is not: by then the track has
    CONFIRMED and its id has been in `step()` output, so the correction shows up
    as a renumbered track. That is a real ID switch and any metric counting
    switches will count it.

    It is accepted ONLY under --single-occupant, where the identity recovered can
    only be the one occupant's. In the default path `retry_offsets` is empty and
    this cannot happen at all. This test exists so the trade is pinned in the
    suite rather than living in a docstring: if someone turns the retry on by
    default, they are turning THIS on.
    """
    cams = bedroom_rig()
    cfg = FusionConfig(
        n_init=3,
        dormant=DormantConfig(retry_offsets=(4, 9), min_hits=1, ttl_s=1e6),
    )
    mgr = GlobalIDManager(_rig(), cfg)

    dormant_id = 77
    home = _unit_embedding(16, 0)
    mgr.dormant.admit(dormant_id, home[None, :], frame=0, hits=50)

    near_miss = _query_at(NEAR_MISS)  # outside the gate, inside the arming ceiling
    reported: dict[int, set[int]] = {}
    for frame in range(12):
        # Near-miss appearance until the track has confirmed, then the person's
        # real appearance — the crop filling out, which is the premise of the
        # whole policy.
        embed = near_miss if frame < 2 else home
        views = _views_for_all_cameras(cams, (3.0, 2.5), embed, frame)
        snaps = mgr.step(views, frame, DT)
        reported[frame] = {s.global_id for s in snaps}

    seen = [ids for ids in reported.values() if ids]
    assert seen, "the track must be reported at some point"
    before, after = seen[0], seen[-1]
    assert before != after, (
        f"expected a visible corrective switch in step() output, saw {before} throughout"
    )
    assert after == {dormant_id}, f"the track must end up under the dormant id, got {after}"
    assert mgr.dormant.n_retries_fired >= 1, "and it must be the retry that did it"


# --- the dormant clock's call-count invariant, pinned at its one call site ----


def test_step_expires_the_dormant_gallery_exactly_once() -> None:
    """The TTL clock is an accumulator, so the CALL COUNT is load-bearing.

    Adversarial review removed `dormant.expire()` from `step()` entirely, and
    separately made it fire twice per step — halving every TTL — and the whole
    suite passed both times. Nothing observed the call site. It does now.
    """
    cams = bedroom_rig()
    mgr = GlobalIDManager(_rig())
    embed = _unit_embedding(16, 0)

    calls: list[tuple[int, float]] = []
    real_expire = mgr.dormant.expire

    def counting_expire(frame: int, dt: float) -> list[int]:
        calls.append((frame, dt))
        return real_expire(frame, dt)

    mgr.dormant.expire = counting_expire  # type: ignore[method-assign]

    for frame in range(6):
        mgr.step(_views_for_all_cameras(cams, (3.0, 2.5), embed, frame), frame, DT)

    assert len(calls) == 6, f"expected one expire() per step, got {len(calls)}"
    assert [f for f, _ in calls] == list(range(6))
    assert all(dt == DT for _, dt in calls), "and it must be handed the step's own dt"


def test_a_dormant_identity_survives_to_its_ttl_and_dies_after() -> None:
    """End-to-end TTL, driven through step() rather than the gallery directly.

    Kills the mutant that deletes expiry from step() altogether: with no expiry
    the identity never leaves, and the second half of this test fails.
    """
    cams = bedroom_rig()
    ttl_s = 2.0
    cfg = FusionConfig(
        n_init=2,
        max_coast_frames=2,
        reid_window_frames=2,
        dormant=DormantConfig(ttl_s=ttl_s, min_hits=1),
    )
    mgr = GlobalIDManager(_rig(), cfg)
    embed = _unit_embedding(16, 0)

    frame = 0
    for _ in range(6):  # confirm, then let it die into the gallery
        mgr.step(_views_for_all_cameras(cams, (3.0, 2.5), embed, frame), frame, DT)
        frame += 1
    for _ in range(8):
        mgr.step([], frame, DT)
        frame += 1
    assert len(mgr.dormant) == 1, "the identity should have retired into the gallery"

    half = int((ttl_s / 2) / DT)
    for _ in range(half):
        mgr.step([], frame, DT)
        frame += 1
    assert len(mgr.dormant) == 1, "must still be resurrectable at half its TTL"

    for _ in range(int(ttl_s / DT) + 10):
        mgr.step([], frame, DT)
        frame += 1
    assert len(mgr.dormant) == 0, "and must be gone once the TTL has elapsed"


def test_step_rejects_a_non_finite_dt() -> None:
    """NaN fails every comparison, so `dt <= 0` let it through.

    It used to cost one bad frame. Since the TTL became an accumulator it is
    permanent: `_elapsed_s` goes NaN and nothing can ever expire again.
    """
    mgr = GlobalIDManager(_rig())
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            mgr.step([], 0, bad)
