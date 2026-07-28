"""Global ID manager — the component the whole project is graded on.

Per frame:
    1. predict every track forward (constant velocity on the floor plane)
    2. project each camera's view observations onto the ground
    3. per-camera Hungarian match against active tracks; fuse all matched
       measurements into the track with sequential Kalman updates
    4. cluster the leftovers across cameras so one person seen by three cameras
       cannot spawn three IDs
    5. try to revive a LOST track from each leftover cluster (ReID + a motion
       -plausibility radius) before minting a new ID
    6. age the lifecycle: coasting -> lost -> dead

Global IDs are never recycled: once an ID dies it is retired, so an ID switch
always shows up as a new number in the metrics rather than being masked by
reuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import feet_point, ground_covariance, image_to_ground
from mcreid.calib.schema import RigCalib
from mcreid.fusion.associate import (
    INFEASIBLE,
    AppearanceGallery,
    AssociationConfig,
    build_cost_matrix,
    linear_assignment,
)
from mcreid.fusion.motion import GroundKalman
from mcreid.fusion.types import (
    GlobalTrackSnapshot,
    GroundObservation,
    TrackState,
    ViewObservation,
)
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

# How many of a merged-away track's embeddings the survivor inherits, per camera.
_ABSORB_PER_CAMERA = 3


@dataclass(frozen=True)
class FusionConfig:
    """Lifecycle + fusion parameters. One place, no scattered magic numbers."""

    association: AssociationConfig = field(default_factory=AssociationConfig)

    # --- lifecycle ---
    n_init: int = 3
    """Frames of support before a tentative track is confirmed (kills false positives)."""
    max_coast_frames: int = 90
    """Frames a confirmed track may coast with no measurement before going LOST.
    90 @ 30 fps = 3 s, comfortably longer than the 2-3 s total-occlusion target."""
    reid_window_frames: int = 300
    """Frames a LOST track stays in the re-association gallery. 10 s @ 30 fps."""

    # --- birth clustering ---
    birth_cluster_radius_m: float = 1.0
    """Leftover observations within this ground distance (and matching on
    appearance) are treated as the same new person."""
    merge_radius_m: float = 0.75
    """Two live tracks closer than this *and* appearance-compatible are the same
    person seen twice — merge them, keeping the senior ID.

    Without this pass the manager is stable but wrong: per-camera assignment is
    one-to-one, so when two cameras match an existing track and two do not, the
    leftovers legitimately birth a second track on top of the first. Both then
    survive, and every frame the reported identity flips between them."""
    merge_appearance_distance: float = 0.34
    """EMA-to-EMA cosine distance ceiling for a merge. Two co-located tracks of
    one person must clear this; two different people walking past each other
    must not. Measured separation on the toy generator: same identity across
    cameras ~0.27, different identities ~0.53."""

    # --- revival ---
    revive_appearance_distance: float = 0.34
    """Tighter than the frame-to-frame gate: reviving the wrong ID is the one
    failure the cardboard test cannot survive, so demand stronger evidence."""
    revive_speed_margin_m: float = 1.5
    """Slack added to ``max_speed_mps * elapsed`` when bounding how far a lost
    target could have walked while unobserved."""
    revive_max_reach_m: float | None = None
    """Hard ceiling on that reach. ``max_speed * elapsed`` exceeds the diagonal of
    a bedroom after about two seconds, so past that point the motion gate admits
    the entire room and stops constraining anything. Defaults to the rig's floor
    diagonal when the rig declares an extent."""
    revive_gallery_top_k: int = 3
    """Revival ranks candidates on the mean of the ``k`` best gallery matches
    rather than the single best. Max-similarity over a large gallery is an
    optimistic statistic whose false-accept rate climbs steeply with gallery
    size, and revival is exactly where a wrong decision renames a person."""

    # --- measurement model ---
    detection_sigma_px: float = 6.0
    """Assumed foot-point localisation noise. Feeds the homography Jacobian, so
    far-away targets automatically get looser geometric gates."""
    ground_model_sigma_m: float = 0.15
    """Isotropic world-space error floor for bbox-derived foot points. Measured
    on the toy generator: pixel noise alone underestimates the true projection
    error by ~5x, because a box's bottom-centre is not the ground-contact point."""
    truncated_box_sigma_multiplier: float = 3.0
    """Covariance inflation for detections clipped by the frame border. A box cut
    off at the image bottom has lost the feet entirely, so its foot point is a
    guess — down-weight it instead of discarding an otherwise good observation."""
    border_tolerance_px: float = 2.0
    max_position_sigma_m: float = 1.5
    """Discard ground observations whose projected uncertainty exceeds this —
    grazing-angle detections near the horizon are worse than no measurement."""

    # --- motion ---
    process_noise: float = 1.5
    max_speed_mps: float = 4.0
    coast_velocity_damping: float = 0.92
    """Per-frame velocity decay while coasting with no measurement.

    Undamped constant velocity is the wrong prior for the hero case: someone
    standing behind a cardboard sheet is usually *not* still walking in a
    straight line, so after 2.5 s an undamped estimate has slid ~2.7 m away and
    the re-lock fails. Damping parks the estimate near where the target was last
    seen, which is both a better prior and what makes the BEV dot behave sanely
    on camera. 0.92^30 ~ 0.08, so the dot glides for ~0.3 s and then holds."""

    def __post_init__(self) -> None:
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1")
        if self.max_coast_frames < 1:
            raise ValueError("max_coast_frames must be >= 1")
        # Every knob below can silently produce garbage if left unchecked:
        # damping > 1 accelerates a coasted track instead of parking it, a huge
        # merge radius fuses the whole room, and a negative sigma drops every
        # observation. Fail loudly at construction instead.
        if not 0.0 < self.coast_velocity_damping <= 1.0:
            raise ValueError(
                f"coast_velocity_damping must be in (0, 1] — values above 1 make a "
                f"coasting track accelerate away from its target; got "
                f"{self.coast_velocity_damping}"
            )
        positive = (
            "birth_cluster_radius_m",
            "merge_radius_m",
            "merge_appearance_distance",
            "revive_appearance_distance",
            "detection_sigma_px",
            "max_position_sigma_m",
            "process_noise",
            "max_speed_mps",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        non_negative = ("revive_speed_margin_m", "ground_model_sigma_m", "border_tolerance_px")
        for name in non_negative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.revive_max_reach_m is not None and self.revive_max_reach_m <= 0.0:
            raise ValueError("revive_max_reach_m must be positive when set")
        if self.revive_gallery_top_k < 1:
            raise ValueError("revive_gallery_top_k must be >= 1")
        if self.truncated_box_sigma_multiplier < 1.0:
            raise ValueError(
                "truncated_box_sigma_multiplier must be >= 1 — a clipped box is never "
                f"more trustworthy than a whole one; got {self.truncated_box_sigma_multiplier}"
            )
        if self.reid_window_frames < self.max_coast_frames:
            raise ValueError(
                f"reid_window_frames ({self.reid_window_frames}) must be >= "
                f"max_coast_frames ({self.max_coast_frames}); otherwise a track dies "
                "before it can ever be re-associated"
            )
        if self.revive_appearance_distance > self.association.max_appearance_distance:
            raise ValueError(
                "revive_appearance_distance must be <= association.max_appearance_distance"
            )


class GlobalTrack:
    """One persistent identity on the floor plane."""

    def __init__(
        self,
        global_id: int,
        frame: int,
        mean: FloatArray,
        cov: FloatArray,
        config: FusionConfig,
    ) -> None:
        self.global_id = global_id
        self.birth_frame = frame
        self.mean = mean
        self.cov = cov
        self.state = TrackState.TENTATIVE
        self.gallery = AppearanceGallery()
        self.hits = 0
        self.age = 0
        self.frames_since_measurement = 0
        self.last_measured_frame = frame
        self.last_measured_xy = mean[:2].copy()
        self.supporting_cameras: tuple[str, ...] = ()
        self._config = config

    # --- properties -------------------------------------------------------

    @property
    def world_xy(self) -> FloatArray:
        return self.mean[:2].copy()

    @property
    def velocity_mps(self) -> FloatArray:
        return self.mean[2:].copy()

    @property
    def position_cov(self) -> FloatArray:
        return self.cov[:2, :2].copy()

    @property
    def is_active(self) -> bool:
        """Eligible for normal frame-to-frame association."""
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.COASTING)

    @property
    def is_visible(self) -> bool:
        """Should be drawn on the BEV (confirmed, possibly coasting)."""
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    # --- filter -----------------------------------------------------------

    def predict(self, kf: GroundKalman, dt: float) -> None:
        if self.frames_since_measurement > 0:
            self.mean[2:] *= self._config.coast_velocity_damping
        self.mean, self.cov = kf.predict(self.mean, self.cov, dt)
        self.age += 1

    def update(self, kf: GroundKalman, observations: list[GroundObservation], frame: int) -> None:
        """Fuse every camera's measurement for this frame."""
        if not observations:
            raise ValueError("update() called with no observations")
        for obs in observations:
            self.mean, self.cov = kf.update(self.mean, self.cov, obs.world_xy, obs.world_cov)
            self.gallery.add(obs.camera_id, obs.embedding)

        self.hits += 1
        self.frames_since_measurement = 0
        self.last_measured_frame = frame
        self.last_measured_xy = self.mean[:2].copy()
        self.supporting_cameras = tuple(sorted({o.camera_id for o in observations}))

        # A track that was occluded is confirmed again the moment it is measured;
        # a tentative one only after it has survived n_init frames.
        recovered = self.state in (TrackState.COASTING, TrackState.LOST)
        promoted = self.state is TrackState.TENTATIVE and self.hits >= self._config.n_init
        if recovered or promoted:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """No measurement this frame — advance the lifecycle."""
        self.frames_since_measurement += 1
        self.supporting_cameras = ()

        if self.state is TrackState.TENTATIVE:
            # An unconfirmed track that immediately vanishes was a false positive.
            self.state = TrackState.DEAD
            return
        if self.state in (TrackState.CONFIRMED, TrackState.COASTING):
            self.state = TrackState.COASTING
            if self.frames_since_measurement > self._config.max_coast_frames:
                self.state = TrackState.LOST
            return
        if (
            self.state is TrackState.LOST
            and self.frames_since_measurement > self._config.reid_window_frames
        ):
            self.state = TrackState.DEAD

    def absorb(self, other: GlobalTrack) -> None:
        """Fold a duplicate track into this one. ``self`` keeps its global ID."""
        if other.global_id == self.global_id:
            raise ValueError("a track cannot absorb itself")
        # Absorb only a bounded, most-recent slice of the loser's appearance
        # evidence. `gallery.distance` is 1 - max similarity, so wholesale
        # copying makes one bad merge permanent: the survivor gains a vector that
        # will happily match the wrong person forever, and there is no un-merge.
        for camera_id in other.gallery.cameras:
            vectors = [v for cam, v in other.gallery.items() if cam == camera_id]
            for vector in vectors[-_ABSORB_PER_CAMERA:]:
                self.gallery.add(camera_id, vector)
        self.hits = max(self.hits, other.hits)
        self.birth_frame = min(self.birth_frame, other.birth_frame)
        self.supporting_cameras = tuple(
            sorted(set(self.supporting_cameras) | set(other.supporting_cameras))
        )
        if other.frames_since_measurement < self.frames_since_measurement:
            self.frames_since_measurement = other.frames_since_measurement
            self.last_measured_frame = other.last_measured_frame
        if other.state is TrackState.CONFIRMED and self.state is TrackState.TENTATIVE:
            self.state = TrackState.CONFIRMED

    def snapshot(self, frame: int) -> GlobalTrackSnapshot:
        return GlobalTrackSnapshot(
            global_id=self.global_id,
            frame=frame,
            world_xy=self.world_xy,
            velocity_mps=self.velocity_mps,
            covariance=self.position_cov,
            state=self.state,
            supporting_cameras=self.supporting_cameras,
            frames_since_measurement=self.frames_since_measurement,
            hits=self.hits,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        x, y = self.world_xy
        return (
            f"GlobalTrack(id={self.global_id}, state={self.state.value}, "
            f"xy=({x:.2f}, {y:.2f}), hits={self.hits}, missed={self.frames_since_measurement})"
        )


@dataclass
class _Cluster:
    """Leftover observations believed to belong to one unassigned person."""

    observations: list[GroundObservation]

    @property
    def cameras(self) -> set[str]:
        return {o.camera_id for o in self.observations}

    @property
    def world_xy(self) -> FloatArray:
        """Inverse-variance weighted mean position."""
        positions = np.stack([o.world_xy for o in self.observations])
        weights = np.array([1.0 / max(o.position_sigma_m**2, 1e-6) for o in self.observations])
        return np.asarray(np.average(positions, axis=0, weights=weights), dtype=np.float64)

    @property
    def embedding(self) -> FloatArray:
        mean = np.mean([o.embedding for o in self.observations], axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-12 else mean

    @property
    def score(self) -> float:
        return max(o.score for o in self.observations)


class GlobalIDManager:
    """Owns the global track set and the ID counter."""

    def __init__(self, rig: RigCalib, config: FusionConfig | None = None) -> None:
        self.rig = rig
        self.config = config or FusionConfig()
        self.kf = GroundKalman(
            process_noise=self.config.process_noise,
            max_speed_mps=self.config.max_speed_mps,
        )
        self.tracks: list[GlobalTrack] = []
        self._ids_issued = 0
        self._frame = -1
        self._last_dt = 1.0 / 30.0
        self.last_assignment: dict[tuple[str, int], int] = {}
        """(camera_id, local_track_id) -> global_id for the most recent frame.
        The overlay needs this to label each per-view box with the *global* ID,
        which is the whole claim the demo makes."""

    # --- public API -------------------------------------------------------

    def project_observations(
        self, views: list[ViewObservation], frame: int
    ) -> list[GroundObservation]:
        """Map per-view observations onto the floor, dropping unusable ones."""
        out: list[GroundObservation] = []
        by_camera: dict[str, list[ViewObservation]] = {}
        for view in views:
            by_camera.setdefault(view.camera_id, []).append(view)

        for camera_id, group in by_camera.items():
            cam = self.rig.get(camera_id)
            boxes = np.stack([np.asarray(v.bbox_xyxy, dtype=np.float64) for v in group])
            feet = feet_point(boxes)
            world, valid = image_to_ground(cam, feet)
            covs = ground_covariance(
                cam,
                feet,
                sigma_px=self.config.detection_sigma_px,
                model_sigma_m=self.config.ground_model_sigma_m,
            )
            truncated = self._touches_border(boxes, cam.intrinsics.image_size)
            inflate = np.where(
                truncated, self.config.truncated_box_sigma_multiplier**2, 1.0
            )
            covs = covs * inflate[:, None, None]

            for i, view in enumerate(group):
                if not valid[i] or not np.isfinite(world[i]).all():
                    logger.debug(
                        "frame %d cam %s track %d: foot point beyond the horizon, dropped",
                        frame,
                        camera_id,
                        view.local_track_id,
                    )
                    continue
                cov = covs[i]
                if not np.isfinite(cov).all():
                    continue
                sigma = float(np.sqrt(np.trace(cov) / 2.0))
                if sigma > self.config.max_position_sigma_m:
                    logger.debug(
                        "frame %d cam %s track %d: ground sigma %.2f m > %.2f m, dropped",
                        frame,
                        camera_id,
                        view.local_track_id,
                        sigma,
                        self.config.max_position_sigma_m,
                    )
                    continue
                out.append(
                    GroundObservation(
                        camera_id=camera_id,
                        frame=frame,
                        local_track_id=view.local_track_id,
                        world_xy=world[i],
                        world_cov=cov,
                        embedding=np.asarray(view.embedding, dtype=np.float64),
                        score=view.score,
                    )
                )
        return out

    def step(
        self, views: list[ViewObservation], frame: int, dt: float
    ) -> list[GlobalTrackSnapshot]:
        """Advance the fusion stage by one frame. Returns the visible tracks."""
        if frame <= self._frame:
            raise ValueError(f"frames must increase: got {frame} after {self._frame}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._frame = frame
        self._last_dt = dt

        for track in self.tracks:
            track.predict(self.kf, dt)

        ground = self.project_observations(views, frame)
        matched: dict[int, list[GroundObservation]] = {}
        leftovers: list[GroundObservation] = []
        self.last_assignment = {}

        active = [t for t in self.tracks if t.is_active]
        for camera_id in sorted({o.camera_id for o in ground}):
            observations = [o for o in ground if o.camera_id == camera_id]
            pairs, unmatched_obs, _ = self._match_camera(observations, active)
            for obs_idx, track_idx in pairs:
                obs = observations[obs_idx]
                global_id = active[track_idx].global_id
                matched.setdefault(global_id, []).append(obs)
                self.last_assignment[(camera_id, obs.local_track_id)] = global_id
            leftovers.extend(observations[i] for i in unmatched_obs)

        # Every live track advances exactly once per frame: measured tracks are
        # updated, everything else (including LOST tracks waiting in the ReID
        # gallery) ages by one. Splitting this into per-state loops double-counts
        # a track that changes state inside mark_missed().
        for track in self.tracks:
            if track.state is TrackState.DEAD:
                continue
            measurements = matched.get(track.global_id)
            if measurements:
                track.update(self.kf, measurements, frame)
            else:
                track.mark_missed()

        clusters = self._cluster(leftovers)
        remaining = self._revive(clusters, frame)
        for cluster in remaining:
            self._birth(cluster, frame)

        self._merge_duplicates(frame)
        self.tracks = [t for t in self.tracks if t.state is not TrackState.DEAD]
        return [t.snapshot(frame) for t in self.tracks if t.is_visible]

    # --- internals --------------------------------------------------------

    def _touches_border(
        self, boxes: FloatArray, image_size: tuple[int, int]
    ) -> npt.NDArray[np.bool_]:
        """True for boxes clipped by the frame edge — their foot point is unreliable."""
        width, height = image_size
        tol = self.config.border_tolerance_px
        return (
            (boxes[:, 3] >= height - tol)
            | (boxes[:, 1] <= tol)
            | (boxes[:, 0] <= tol)
            | (boxes[:, 2] >= width - tol)
        )

    def _match_camera(
        self, observations: list[GroundObservation], tracks: list[GlobalTrack]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not observations or not tracks:
            return [], list(range(len(observations))), list(range(len(tracks)))

        positions = np.stack([o.world_xy for o in observations])
        covs = np.stack([o.world_cov for o in observations])
        embeddings = np.stack([o.embedding for o in observations])

        n_obs, n_tracks = len(observations), len(tracks)
        maha = np.empty((n_obs, n_tracks), dtype=np.float64)
        euclid = np.empty((n_obs, n_tracks), dtype=np.float64)
        appearance = np.empty((n_obs, n_tracks), dtype=np.float64)

        for j, track in enumerate(tracks):
            maha[:, j] = self.kf.mahalanobis_sq(track.mean, track.cov, positions, covs)
            euclid[:, j] = np.linalg.norm(positions - track.world_xy, axis=1)
            appearance[:, j] = track.gallery.distance(embeddings)

        cost = build_cost_matrix(maha, euclid, appearance, self.config.association)
        return linear_assignment(cost, self.config.association.max_cost)

    def _cluster(self, observations: list[GroundObservation]) -> list[_Cluster]:
        """Greedy cross-camera grouping of unassigned observations.

        Without this, a newly entering person visible in four cameras would be
        born as four separate global IDs on their first frame.
        """
        clusters: list[_Cluster] = []
        radius = self.config.birth_cluster_radius_m
        max_app = self.config.association.max_appearance_distance

        for obs in sorted(observations, key=lambda o: o.score, reverse=True):
            placed = False
            for cluster in clusters:
                if obs.camera_id in cluster.cameras:
                    continue  # one observation per camera per person
                if float(np.linalg.norm(cluster.world_xy - obs.world_xy)) > radius:
                    continue
                if float(1.0 - cluster.embedding @ obs.embedding) > max_app:
                    continue
                cluster.observations.append(obs)
                placed = True
                break
            if not placed:
                clusters.append(_Cluster(observations=[obs]))
        return clusters

    def _revive(self, clusters: list[_Cluster], frame: int) -> list[_Cluster]:
        """Re-attach leftover clusters to occluded tracks. Returns the unrevived rest.

        This is the mechanism that survives the total-occlusion window. Candidates
        are every track that missed at least one frame — both LOST tracks sitting
        in the ReID gallery and still-COASTING ones. Restricting this to LOST
        tracks alone is a subtle trap: with a coast budget longer than the
        blackout, the track is still COASTING when the person reappears, its
        drifted prediction fails the tight frame-to-frame gate, and the manager
        happily mints a fresh ID for someone it is actively tracking.

        Reach is measured from the last *observed* position rather than the
        coasted one, so a wrong extrapolation cannot also shrink the search
        region that has to correct it.
        """
        candidates = [
            t
            for t in self.tracks
            if t.state in (TrackState.COASTING, TrackState.LOST)
            and t.frames_since_measurement > 0
        ]
        if not clusters or not candidates:
            return clusters

        cost = np.full((len(clusters), len(candidates)), INFEASIBLE, dtype=np.float64)
        embeddings = np.stack([c.embedding for c in clusters])
        positions = np.stack([c.world_xy for c in clusters])

        reach_cap = self.config.revive_max_reach_m or self._floor_diagonal_m()
        for j, track in enumerate(candidates):
            elapsed_s = track.frames_since_measurement * self._last_dt
            reach = min(
                self.config.max_speed_mps * elapsed_s + self.config.revive_speed_margin_m,
                reach_cap,
            )
            distance = np.linalg.norm(positions - track.last_measured_xy, axis=1)
            appearance = track.gallery.robust_distance(
                embeddings, top_k=self.config.revive_gallery_top_k
            )
            feasible = (distance <= reach) & (
                appearance <= self.config.revive_appearance_distance
            )
            # Appearance alone ranks the candidates — after a long blackout the
            # geometric prior is close to worthless and should not break ties.
            cost[:, j] = np.where(feasible, appearance, INFEASIBLE)

        matches, unmatched_clusters, _ = linear_assignment(
            cost, self.config.revive_appearance_distance
        )
        for cluster_idx, track_idx in matches:
            track = candidates[track_idx]
            cluster = clusters[cluster_idx]
            gap = track.frames_since_measurement
            was_lost = track.state is TrackState.LOST
            track.update(self.kf, cluster.observations, frame)
            self._record_assignment(cluster, track.global_id)
            if gap >= self.config.n_init or was_lost:
                logger.info(
                    "frame %d: re-associated global id %d after %d frames occluded "
                    "(cameras %s)",
                    frame,
                    track.global_id,
                    gap,
                    track.supporting_cameras,
                )
        return [clusters[i] for i in unmatched_clusters]

    def _merge_duplicates(self, frame: int) -> None:
        """Collapse co-located, appearance-compatible tracks into the senior one.

        Seniority = confirmed before tentative, then more hits, then lower ID —
        so the person keeps the identity they have been carrying, which is
        exactly what the ID-switch metric measures.
        """
        live = [t for t in self.tracks if t.is_active]
        if len(live) < 2:
            return
        # Seniority is about track history, not this frame's measurement status.
        # Ranking COASTING below CONFIRMED lets a 3-hit track that just confirmed
        # absorb — and rename — a 500-hit identity that happens to be behind the
        # cardboard right now, which is precisely the failure this project exists
        # to prevent.
        live.sort(
            key=lambda t: (
                t.state is TrackState.TENTATIVE,
                -t.hits,
                t.birth_frame,
                t.global_id,
            )
        )

        absorbed: set[int] = set()
        # Merging is destructive and irreversible, so it is tested against the
        # tighter revival threshold, and on the two tracks' EMA vectors rather
        # than best-case gallery agreement. `gallery.distance` reports the
        # distance to the *most similar* stored vector; across two large
        # galleries that statistic is optimistic enough to occasionally fuse two
        # different people who walk past each other, which is the worst failure
        # this system can produce.
        max_app = self.config.merge_appearance_distance
        for i, keep in enumerate(live):
            if keep.global_id in absorbed:
                continue
            for other in live[i + 1 :]:
                if other.global_id in absorbed:
                    continue
                if float(np.linalg.norm(keep.world_xy - other.world_xy)) > (
                    self.config.merge_radius_m
                ):
                    continue
                keep_ema, other_ema = keep.gallery.ema, other.gallery.ema
                if keep_ema is None or other_ema is None:
                    continue
                if float(1.0 - keep_ema @ other_ema) > max_app:
                    continue
                logger.debug(
                    "frame %d: merged duplicate global id %d into %d (%.2f m apart)",
                    frame,
                    other.global_id,
                    keep.global_id,
                    float(np.linalg.norm(keep.world_xy - other.world_xy)),
                )
                keep.absorb(other)
                self._remap_assignment(other.global_id, keep.global_id)
                absorbed.add(other.global_id)

        if absorbed:
            self.tracks = [t for t in self.tracks if t.global_id not in absorbed]

    def _floor_diagonal_m(self) -> float:
        """Diagonal of the rig's declared floor, or +inf when none is declared."""
        try:
            x0, y0, x1, y1 = self.rig.floor_extent()
        except ValueError:
            return float("inf")
        return float(np.hypot(x1 - x0, y1 - y0))

    def _record_assignment(self, cluster: _Cluster, global_id: int) -> None:
        for obs in cluster.observations:
            self.last_assignment[(obs.camera_id, obs.local_track_id)] = global_id

    def _remap_assignment(self, old_id: int, new_id: int) -> None:
        for key, value in self.last_assignment.items():
            if value == old_id:
                self.last_assignment[key] = new_id

    def _birth(self, cluster: _Cluster, frame: int) -> None:
        first = cluster.observations[0]
        mean, cov = self.kf.initiate(cluster.world_xy, first.world_cov)
        self._ids_issued += 1
        track = GlobalTrack(
            global_id=self._ids_issued, frame=frame, mean=mean, cov=cov, config=self.config
        )
        track.update(self.kf, cluster.observations, frame)
        self._record_assignment(cluster, track.global_id)
        self.tracks.append(track)
        logger.debug(
            "frame %d: born global id %d at (%.2f, %.2f) from %d camera(s)",
            frame,
            track.global_id,
            *cluster.world_xy,
            len(cluster.cameras),
        )

    # --- introspection ----------------------------------------------------

    @property
    def n_ids_issued(self) -> int:
        """Total global IDs ever minted. IDs are never recycled, so this is the
        honest denominator for the ID-switch metric."""
        return self._ids_issued

    def active_snapshots(self) -> list[GlobalTrackSnapshot]:
        return [t.snapshot(self._frame) for t in self.tracks if t.is_visible]
