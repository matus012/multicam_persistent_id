"""Scripted multi-view toy sequences — the CI stand-in for real footage.

Everything is analytic and seeded: agents walk polylines on the floor, N virtual
cameras project them, and scripted occlusion events knock them out of chosen
views. The hero case (`cardboard_scene`) reproduces the ship criterion — a
single agent occluded camera-by-camera and then from *every* view at once — so
the ID-consistency gate can run with no footage and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import RigCalib
from mcreid.sim.virtual_camera import VirtualCamera
from mcreid.utils.logging import get_logger
from mcreid.utils.seed import DEFAULT_SEED

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AgentSpec:
    """A person walking a polyline on the floor at constant speed."""

    agent_id: int
    waypoints_m: tuple[tuple[float, float], ...]
    speed_mps: float = 1.2
    height_m: float = 1.75
    width_m: float = 0.55
    start_offset_m: float = 0.0

    def __post_init__(self) -> None:
        if len(self.waypoints_m) < 2:
            raise ValueError(f"agent {self.agent_id}: need >= 2 waypoints")
        if self.speed_mps <= 0.0:
            raise ValueError(f"agent {self.agent_id}: speed must be positive")
        if self.height_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError(f"agent {self.agent_id}: body dimensions must be positive")

    def position_at(self, t_s: float) -> FloatArray:
        """Ping-pong along the polyline so agents stay inside the room."""
        pts = np.asarray(self.waypoints_m, dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0.0:
            return pts[0].copy()

        travelled = self.start_offset_m + self.speed_mps * t_s
        cycle = travelled % (2.0 * total)
        along = cycle if cycle <= total else 2.0 * total - cycle

        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        idx = int(np.searchsorted(cumulative, along, side="right") - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        local = (along - cumulative[idx]) / seg[idx]
        return pts[idx] + local * (pts[idx + 1] - pts[idx])


@dataclass(frozen=True)
class OcclusionEvent:
    """A scripted visibility blackout — the simulated cardboard sheet.

    ``camera_ids=None`` means *every* camera: total occlusion, the case the
    global ID manager must coast through.
    """

    agent_id: int
    start_frame: int
    end_frame: int  # exclusive
    camera_ids: tuple[str, ...] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError(f"empty occlusion window [{self.start_frame}, {self.end_frame})")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")

    def hides(self, agent_id: int, camera_id: str, frame: int) -> bool:
        if agent_id != self.agent_id or not (self.start_frame <= frame < self.end_frame):
            return False
        return self.camera_ids is None or camera_id in self.camera_ids

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class NoiseSpec:
    """Detector / ReID imperfection knobs.

    The embedding knobs are expressed as magnitudes *relative to the unit
    identity prototype* (the raw gaussians are scaled by 1/sqrt(embed_dim)), so
    they mean the same thing at any ``embed_dim``.

    Defaults are tuned to reproduce published person-ReID operating points
    rather than a flattering toy: cross-camera same-identity cosine similarity
    lands around 0.7 and different-identity around 0.45. Orthogonal random
    prototypes would give ~0.0 for different identities and make the
    ID-consistency gate meaningless, so identities deliberately share a
    ``identity_similarity``-weighted "person-ness" component.
    """

    bbox_jitter_px: float = 3.0
    embed_noise: float = 0.45
    camera_bias: float = 0.40
    identity_similarity: float = 0.85
    dropout_prob: float = 0.02
    false_positive_rate: float = 0.05
    score_mean: float = 0.88
    score_std: float = 0.05

    def __post_init__(self) -> None:
        for name in ("bbox_jitter_px", "embed_noise", "camera_bias", "score_std"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("dropout_prob", "false_positive_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.identity_similarity < 0.0:
            raise ValueError("identity_similarity must be non-negative")


@dataclass(frozen=True)
class StaticFalsePositive:
    """A persistent detector hallucination at a fixed floor position.

    Real detectors do not produce i.i.d. noise: they produce a coat rack, a
    mannequin, or a poster that fires in the same place on every frame with a
    stable appearance. That is the false positive that actually costs identities,
    because it survives per-view tracking and reaches fusion; uniform per-frame
    noise cannot form a tracklet and is filtered for free.
    """

    world_xy_m: tuple[float, float]
    height_m: float = 1.70
    width_m: float = 0.5
    score: float = 0.72
    camera_ids: tuple[str, ...] | None = None  # None = every camera

    def visible_to(self, camera_id: str) -> bool:
        return self.camera_ids is None or camera_id in self.camera_ids


@dataclass(frozen=True)
class ToySceneConfig:
    cameras: tuple[VirtualCamera, ...]
    agents: tuple[AgentSpec, ...]
    occlusions: tuple[OcclusionEvent, ...] = ()
    static_false_positives: tuple[StaticFalsePositive, ...] = ()
    n_frames: int = 300
    fps: float = 30.0
    embed_dim: int = 128
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    floor_extent_m: tuple[float, float, float, float] = (0.0, 0.0, 6.0, 5.0)
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("need >= 1 camera")
        if not self.agents:
            raise ValueError("need >= 1 agent")
        if self.n_frames <= 0 or self.fps <= 0.0:
            raise ValueError("n_frames and fps must be positive")
        if self.embed_dim < 8:
            raise ValueError("embed_dim < 8 makes cosine separation meaningless")
        ids = [c.camera_id for c in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate camera_id: {ids}")
        agent_ids = [a.agent_id for a in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError(f"duplicate agent_id: {agent_ids}")
        known = set(ids)
        for occ in self.occlusions:
            if occ.agent_id not in set(agent_ids):
                raise ValueError(f"occlusion references unknown agent {occ.agent_id}")
            if occ.camera_ids is not None and not set(occ.camera_ids) <= known:
                raise ValueError(f"occlusion references unknown camera(s): {occ.camera_ids}")

    @property
    def camera_ids(self) -> list[str]:
        return [c.camera_id for c in self.cameras]


@dataclass(frozen=True)
class ToyDetection:
    """One synthetic per-view detection."""

    camera_id: str
    frame: int
    bbox_xyxy: FloatArray
    embedding: FloatArray
    score: float
    gt_agent_id: int | None  # None => false positive


@dataclass
class ToyScene:
    """Materialised sequence: calibration + per-frame detections + ground truth."""

    config: ToySceneConfig
    rig: RigCalib
    detections: list[dict[str, list[ToyDetection]]]  # [frame][camera_id] -> detections
    gt_world: dict[int, FloatArray]  # agent_id -> (n_frames, 2) floor positions
    gt_visible: dict[int, npt.NDArray[np.bool_]]  # agent_id -> (n_frames, n_cameras)

    @property
    def n_frames(self) -> int:
        return self.config.n_frames

    def frame_detections(self, frame: int) -> dict[str, list[ToyDetection]]:
        return self.detections[frame]

    def total_blackout_frames(self) -> npt.NDArray[np.bool_]:
        """Frames where at least one agent is invisible in *every* camera."""
        stacked = np.stack([v.any(axis=1) for v in self.gt_visible.values()], axis=1)
        return ~stacked.any(axis=1)

    def longest_blackout(self, agent_id: int) -> int:
        """Longest run of frames where ``agent_id`` is in no camera at all."""
        hidden = ~self.gt_visible[agent_id].any(axis=1)
        best = run = 0
        for flag in hidden:
            run = run + 1 if flag else 0
            best = max(best, run)
        return best


def _identity_prototypes(
    n: int, dim: int, rng: np.random.Generator, shared_weight: float
) -> tuple[FloatArray, FloatArray]:
    """Unit identity vectors sharing a common "person-ness" direction.

    ``shared_weight`` controls how alike two different people look: the expected
    cosine similarity between distinct identities is w^2 / (1 + w^2).

    Returns (prototypes (n, dim), shared_direction (dim,)).
    """
    shared = rng.normal(size=dim)
    shared /= np.linalg.norm(shared)
    individual = rng.normal(size=(n, dim))
    individual /= np.linalg.norm(individual, axis=1, keepdims=True)
    proto = individual + shared_weight * shared
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    return proto, shared


def _camera_biases(n: int, dim: int, rng: np.random.Generator) -> FloatArray:
    """Per-camera appearance bias — stands in for the viewpoint/colour domain gap."""
    bias = rng.normal(size=(n, dim))
    bias /= np.linalg.norm(bias, axis=1, keepdims=True)
    return bias


def _normalise(vec: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero embedding")
    return vec / norm


def generate_scene(config: ToySceneConfig) -> ToyScene:
    """Materialise a toy multi-view sequence. Fully deterministic given the seed."""
    rng = np.random.default_rng(config.seed)
    cams = list(config.cameras)
    agents = list(config.agents)
    noise = config.noise

    prototypes, shared_direction = _identity_prototypes(
        len(agents), config.embed_dim, rng, noise.identity_similarity
    )
    biases = _camera_biases(len(cams), config.embed_dim, rng)
    # Static FPs get their own stable identities, drawn from the same
    # person-like distribution — a coat rack that looks nothing like a person
    # would be trivially rejectable and would not test anything.
    static_fp_protos = (
        _identity_prototypes(
            len(config.static_false_positives), config.embed_dim, rng, noise.identity_similarity
        )[0]
        if config.static_false_positives
        else np.zeros((0, config.embed_dim))
    )
    # Gaussian perturbations are scaled so their norm equals the configured
    # magnitude regardless of embed_dim (a raw N(0,1)^D vector has norm ~sqrt(D)).
    noise_scale = 1.0 / np.sqrt(config.embed_dim)
    agent_index = {a.agent_id: i for i, a in enumerate(agents)}
    cam_index = {c.camera_id: i for i, c in enumerate(cams)}

    rig = RigCalib(
        cameras=[c.to_calib(floor_extent_m=config.floor_extent_m) for c in cams],
        world_notes=f"synthetic toy scene, seed={config.seed}",
    )

    gt_world = {
        a.agent_id: np.full((config.n_frames, 2), np.nan, dtype=np.float64) for a in agents
    }
    gt_visible = {
        a.agent_id: np.zeros((config.n_frames, len(cams)), dtype=bool) for a in agents
    }
    detections: list[dict[str, list[ToyDetection]]] = []

    x0, y0, x1, y1 = config.floor_extent_m

    for frame in range(config.n_frames):
        t_s = frame / config.fps
        per_camera: dict[str, list[ToyDetection]] = {c.camera_id: [] for c in cams}

        for agent in agents:
            foot = agent.position_at(t_s)
            gt_world[agent.agent_id][frame] = foot

            for cam in cams:
                occluded = any(
                    occ.hides(agent.agent_id, cam.camera_id, frame) for occ in config.occlusions
                )
                if occluded:
                    continue
                box = cam.person_bbox(foot, agent.height_m, agent.width_m)
                if box is None:
                    continue

                gt_visible[agent.agent_id][frame, cam_index[cam.camera_id]] = True
                if rng.random() < noise.dropout_prob:
                    continue

                jitter = rng.normal(scale=noise.bbox_jitter_px, size=4)
                noisy = box + jitter
                # Keep the box well-formed after jitter.
                noisy[2] = max(noisy[2], noisy[0] + 2.0)
                noisy[3] = max(noisy[3], noisy[1] + 4.0)

                embed = (
                    prototypes[agent_index[agent.agent_id]]
                    + noise.camera_bias * biases[cam_index[cam.camera_id]]
                    + noise.embed_noise * noise_scale * rng.normal(size=config.embed_dim)
                )
                score = float(
                    np.clip(rng.normal(noise.score_mean, noise.score_std), 0.05, 0.999)
                )
                per_camera[cam.camera_id].append(
                    ToyDetection(
                        camera_id=cam.camera_id,
                        frame=frame,
                        bbox_xyxy=noisy,
                        embedding=_normalise(embed),
                        score=score,
                        gt_agent_id=agent.agent_id,
                    )
                )

        # Persistent false positives — same place, every frame, stable appearance.
        for fp_index, fp in enumerate(config.static_false_positives):
            for cam in cams:
                if not fp.visible_to(cam.camera_id):
                    continue
                box = cam.person_bbox(fp.world_xy_m, fp.height_m, fp.width_m)
                if box is None:
                    continue
                embed = (
                    static_fp_protos[fp_index]
                    + noise.camera_bias * biases[cam_index[cam.camera_id]]
                    + noise.embed_noise * noise_scale * rng.normal(size=config.embed_dim)
                )
                per_camera[cam.camera_id].append(
                    ToyDetection(
                        camera_id=cam.camera_id,
                        frame=frame,
                        bbox_xyxy=box + rng.normal(scale=noise.bbox_jitter_px, size=4),
                        embedding=_normalise(embed),
                        score=float(np.clip(rng.normal(fp.score, noise.score_std), 0.05, 0.999)),
                        gt_agent_id=None,
                    )
                )

        # Transient false positives: a hallucination somewhere on the floor.
        for cam in cams:
            if rng.random() >= noise.false_positive_rate:
                continue
            ghost = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            box = cam.person_bbox(ghost, 1.7, 0.55)
            if box is None:
                continue
            # False positives carry the shared person-ness component too — a
            # detector hallucination is usually a person-shaped thing, and an
            # orthogonal random vector would be trivially rejectable.
            ghost_embed = (
                rng.normal(size=config.embed_dim) * noise_scale
                + noise.identity_similarity * shared_direction
            )
            per_camera[cam.camera_id].append(
                ToyDetection(
                    camera_id=cam.camera_id,
                    frame=frame,
                    bbox_xyxy=box,
                    embedding=_normalise(ghost_embed),
                    score=float(np.clip(rng.normal(0.45, 0.1), 0.05, 0.999)),
                    gt_agent_id=None,
                )
            )

        detections.append(per_camera)

    scene = ToyScene(
        config=config,
        rig=rig,
        detections=detections,
        gt_world=gt_world,
        gt_visible=gt_visible,
    )
    logger.info(
        "toy scene: %d frames, %d cameras, %d agents, longest total blackout = %s frames",
        config.n_frames,
        len(cams),
        len(agents),
        {a.agent_id: scene.longest_blackout(a.agent_id) for a in agents},
    )
    return scene


# --------------------------------------------------------------------------
# Canned scenes
# --------------------------------------------------------------------------


def bedroom_rig(
    image_size: tuple[int, int] = (1280, 720),
    room: tuple[float, float] = (6.0, 5.0),
) -> tuple[VirtualCamera, ...]:
    """Four cameras in the corners of a ``room`` metre rectangle, all looking in.

    Mirrors the mount plan in capture_guide.md so the toy sequence and the real
    cardboard shoot exercise the same geometry.
    """
    width, depth = room
    return (
        VirtualCamera("cam0", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0, image_size=image_size),
        VirtualCamera(
            "cam1", (width - 0.3, 0.3, 1.6), yaw_deg=135.0, pitch_deg=20.0, image_size=image_size
        ),
        VirtualCamera(
            "cam2",
            (width - 0.3, depth - 0.3, 2.4),
            yaw_deg=225.0,
            pitch_deg=32.0,
            image_size=image_size,
        ),
        VirtualCamera(
            "cam3", (0.3, depth - 0.3, 1.4), yaw_deg=315.0, pitch_deg=16.0, image_size=image_size
        ),
    )


def cardboard_scene(
    n_frames: int = 420,
    fps: float = 30.0,
    seed: int = DEFAULT_SEED,
    blackout_s: float = 2.5,
    tail_s: float = 1.5,
    room: tuple[float, float] = (6.0, 5.0),
) -> ToySceneConfig:
    """The hero test, in simulation — the exact shape of the cardboard clip.

    Agent 1 walks a loop and is occluded from one camera at a time, then two,
    then three, then from *every* camera for ``blackout_s`` (the ship criterion),
    followed by a ``tail_s`` reappearance window in which the ReID re-lock has to
    recover the same global ID. PASS = one global ID for the whole clip.

    The scene deliberately contains **more than the hero agent**:

    - a *distractor* who is never occluded and keeps walking through the room,
      including near the hero's reappearance point. Without a second body, "zero
      ID switches" is achievable by any tracker that never mints a second
      confirmed ID — no identity reasoning is exercised at all, and a stateless
      stub with no ReID, no filter and no lifecycle passes the gate outright.
    - a *static false positive*, the detector hallucination class that actually
      costs identities, since it forms a stable tracklet and reaches fusion.

    The schedule is defined in seconds and converted with ``fps``, so changing
    the frame rate keeps the physical timings identical.
    """
    cams = bedroom_rig(room=room)
    width, depth = room
    hero = AgentSpec(
        agent_id=1,
        waypoints_m=(
            (1.2, 1.2),
            (width - 1.2, 1.4),
            (width - 1.4, depth - 1.2),
            (1.4, depth - 1.4),
        ),
        speed_mps=1.1,
    )
    # Crosses the room on a different heading and different phase, so it is
    # near the hero at some point during the clip without walking through them.
    distractor = AgentSpec(
        agent_id=2,
        waypoints_m=((width - 1.6, depth - 1.6), (1.6, 1.8)),
        speed_mps=0.85,
        height_m=1.62,
        start_offset_m=1.3,
    )

    def f(seconds: float) -> int:
        return int(round(seconds * fps))

    blackout_frames = f(blackout_s)
    tail_frames = f(tail_s)
    blackout_start = n_frames - blackout_frames - tail_frames
    if blackout_start <= f(7.0):
        raise ValueError(
            f"n_frames={n_frames} at {fps} fps leaves no room for the pre-blackout "
            f"script plus {blackout_s}s blackout plus {tail_s}s tail; "
            f"need >= {f(7.0) + blackout_frames + tail_frames + 1} frames"
        )

    events = (
        # Escalating difficulty: one view blocked, then two, then three.
        OcclusionEvent(1, f(1.0), f(2.0), ("cam0",), label="cardboard vs cam0"),
        OcclusionEvent(1, f(2.5), f(3.5), ("cam1",), label="cardboard vs cam1"),
        OcclusionEvent(1, f(4.0), f(5.0), ("cam2",), label="cardboard vs cam2"),
        OcclusionEvent(1, f(5.5), f(6.5), ("cam3",), label="cardboard vs cam3"),
        OcclusionEvent(1, f(7.0), f(8.0), ("cam0", "cam1"), label="furniture, two views"),
        OcclusionEvent(
            1, f(8.5), f(9.5), ("cam0", "cam1", "cam2"), label="three views, only cam3 left"
        ),
        OcclusionEvent(
            1, blackout_start, blackout_start + blackout_frames, None, label="TOTAL blackout"
        ),
    )
    for event in events:
        if event.end_frame > n_frames:
            raise ValueError(f"occlusion {event.label!r} runs past the clip end")

    return ToySceneConfig(
        cameras=cams,
        agents=(hero, distractor),
        occlusions=events,
        # Sited to be visible in all four cameras — so it reliably forms a
        # persistent track and genuinely stresses the tracker — while staying
        # ~1.3 m clear of both agents' paths. Any closer and it stops testing the
        # tracker and starts confounding the *metric*: when a real person walks
        # within the evaluator's match radius of a stationary false track, the
        # ground-truth Hungarian can legitimately attribute the person to it, and
        # the resulting "ID switch" says nothing about the tracker.
        static_false_positives=(StaticFalsePositive(world_xy_m=(0.6, 2.6)),),
        n_frames=n_frames,
        fps=fps,
        seed=seed,
        floor_extent_m=(0.0, 0.0, width, depth),
    )


def crossing_scene(
    n_frames: int = 300,
    fps: float = 30.0,
    seed: int = DEFAULT_SEED,
    room: tuple[float, float] = (6.0, 5.0),
    pass_offset_m: float = 0.9,
) -> ToySceneConfig:
    """Secondary demo: two people crossing paths with mutual occlusions.

    The ID-swap trap — the two agents' paths intersect, and around the crossing
    they are close enough that ground geometry alone cannot separate them, so
    ReID has to carry the match.

    ``pass_offset_m`` staggers the second agent along its path so the two pass
    *near* each other rather than through the same floor point at the same
    instant. Two people occupying one point is not a scenario a real clip can
    contain, and scoring against it measures nothing except how the tie-break
    happens to fall.
    """
    cams = bedroom_rig(room=room)
    width, depth = room
    agents = (
        AgentSpec(agent_id=1, waypoints_m=((1.0, 1.0), (width - 1.0, depth - 1.0)), speed_mps=1.0),
        AgentSpec(
            agent_id=2,
            waypoints_m=((width - 1.0, 1.0), (1.0, depth - 1.0)),
            speed_mps=1.0,
            start_offset_m=pass_offset_m,
        ),
    )
    occlusions = (
        OcclusionEvent(1, 90, 115, ("cam0", "cam3"), label="agent 2 blocks agent 1"),
        OcclusionEvent(2, 150, 170, ("cam1", "cam2"), label="agent 1 blocks agent 2"),
    )
    return ToySceneConfig(
        cameras=cams,
        agents=agents,
        occlusions=occlusions,
        n_frames=n_frames,
        fps=fps,
        seed=seed,
        floor_extent_m=(0.0, 0.0, width, depth),
    )
