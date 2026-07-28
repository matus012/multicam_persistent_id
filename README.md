# mcreid — multi-camera persistent-ID tracking with a live BEV map

Four indoor cameras, one global ID per person. A person entering any view gets an
identity and keeps it across camera handoffs and through occlusions — including
being hidden from **every** camera at once.

Zero training. Pretrained detector, pretrained ReID, geometric late fusion.

![4-view + BEV demo](docs/assets/cardboard_demo.gif)

*Four camera views and the bird's-eye-view map. The hero is progressively
occluded — one camera, then two, then three, then all four — while a second
person and a persistent false positive keep being tracked alongside. The hero's
BEV dot switches to a hollow coasting ring, holds its ID through the 2.5 s total
blackout, and re-locks on the same ID when they reappear.*

> **Status: G-M1-1 complete.** The numbers below are from the scripted synthetic
> scene, which reproduces the real capture protocol exactly. The recorded-footage
> demo (G-M1-2) is blocked on the capture session — see
> [capture_guide.md](capture_guide.md). Repo is private.

## Results

### Occlusion survival — synthetic cardboard scene, 5 seeds

420 frames, 4 cameras. Scripted occlusions escalate from one blocked view to all
four. The scene contains the hero **plus a second person and a persistent false
positive** — see "why the scene has distractors" below.

| metric | result | target |
|---|---|---|
| ID switches | **0** (all 5 seeds) | 0 |
| ID held across total blackout | **75 frames = 2.50 s** | 2-3 s |
| BEV dot alive during blackout | **75 / 75 frames** | full |
| prediction within 1 m of truth during blackout | 38 frames = 1.27 s | reported, not gated |
| mean coast drift during blackout | 0.88 m | reported, not gated |
| coverage while visible | 98.8 % | > 95 % |
| ground-plane position RMSE | 0.22 m | — |

Note the honest split: the BEV dot survives the **whole** blackout and the
identity is retained, but constant-velocity coasting only stays accurate for
about the first half of it. The last part of the identity is recovered by the
ReID re-lock on reappearance, not by the motion model. The demo prints both
numbers and the gate asserts the first.

The tracker is scored against a ReID model deliberately calibrated to published
person-ReID difficulty — same-identity cross-camera cosine similarity ~0.73,
different-identity ~0.45. Orthogonal random embeddings would make this gate
meaningless.

### Why the scene has distractors

An earlier version of this gate had one person in the scene and reported the same
"0 ID switches". That number was worthless. With a single agent, zero switches is
achieved by any tracker that never mints a second *confirmed* ID — an adversarial
review demonstrated that a **stateless 30-line stub** with no ReID, no Kalman
filter, no coasting and no lifecycle passed all five seeds, and beat the real
tracker's position error while doing it.

The gate now includes a second person and a persistent false positive (the
detector-hallucination class that actually costs identities, since it forms a
stable tracklet instead of being filtered as per-frame noise). The same stub now
fails every seed at 19 % coverage. `longest_blackout_id_held` was also split from
`longest_blackout_alive` because the original metric credited a tracker that
emitted *nothing* for the whole occlusion and re-acquired afterwards.

### Calibration accuracy

End-to-end validation of `mcreid-calibrate` against a synthetic capture session
laid out exactly as `capture_guide.md` prescribes (checkerboard video per phone +
AprilTag floor frames), scored against the known ground-truth cameras:

| quantity | result |
|---|---|
| focal length error | < 0.4 % |
| principal point error | 1–8 px |
| intrinsics reprojection RMS | 0.41 px |
| ground homography residual | 0.1 cm |
| **floor position error, image → world** | **4–16 mm mean, 48 mm worst case** |

### Known limitation — two people crossing

The secondary scenario (two people whose paths intersect, passing within ~0.5 m)
gives **0-4 ID switches depending on seed**, not zero. This is a real limit of a
zero-training geometric baseline: when two targets occupy nearly the same floor
position, geometry is uninformative and a pretrained ReID embedding is the only
discriminator. It is explicitly **not** a v1 gate. See
[`scripts/README_v2_synthetic_engine.md`](scripts/README_v2_synthetic_engine.md)
for what fixing it would take.

### WILDTRACK

Public-benchmark evaluation is G-M1-3. The loaders, grid conversions and
MODA/MODP scoring are implemented and unit-tested; **no numbers are reported
yet** because the dataset has not been run. When it is, it will be a single row
labelled *"geometric baseline, no multi-view training"* alongside published
MVDet figures. No parity is claimed — MVDet is trained on multi-view data and
this is not.

```bash
python scripts/download_wildtrack.py info
```

WILDTRACK sits behind an EPFL consent gate, so the helper prints the request
instructions and verifies an archive you download yourself. It does not scrape
or bypass anything.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

Run the full pipeline and export the demo — no footage, no GPU, no dataset:

```bash
uv run mcreid-demo synthetic --scenario cardboard
```

That command runs per-view tracking, ground projection, cross-view association
and the global ID manager, scores identity consistency, writes
`outputs/demo/cardboard.mp4` + `.gif`, and exits non-zero if the cardboard
criterion fails. It is the same code path the recorded demo uses; only the
detector front-end differs.

Other entry points:

```bash
uv run mcreid-demo synthetic --scenario crossing
```

```bash
uv run pytest
```

## How it works

Late fusion, locked architecture:

```
per camera:  detection -> tracking -> ReID embedding
                          |
                          v  ViewObservation
             foot point -> ground-plane homography
                          |
                          v  GroundObservation (+ covariance)
             per-camera Hungarian vs the global track set
                          |
                          v
             global ID manager:  birth / coast / lost / ReID revive / merge
                          |
                          v
             BEV canvas + 4-view mosaic
```

Association blends a Mahalanobis ground-plane distance with ReID cosine distance.
Because the positional covariance grows while a track coasts unobserved, geometry
gracefully stops discriminating exactly when it should, and appearance takes over
— which is what makes the post-blackout re-lock work.

Three details that turned out to matter more than the architecture:

- **The ground covariance needs a world-space error floor.** Propagating pixel
  noise through the homography Jacobian alone underestimates true projection
  error by ~5x, because a detection box's bottom edge is not the ground-contact
  point. Without the floor, the Mahalanobis gate collapses and one person
  shatters into an ID per frame.
- **Duplicate tracks must be merged.** Assignment is one-to-one *per camera*, so
  leftover observations legitimately birth a second track on top of an existing
  one. Both then survive and the reported identity flips between them every frame.
- **Coasting velocity must be damped, and revival must see coasting tracks.**
  Undamped constant velocity slides ~2.7 m away over a 2.5 s blackout; and if
  ReID revival only considers fully-lost tracks, a still-coasting track gets a
  fresh ID minted for a target the system is actively tracking.

Each was found by measuring, not by inspection. See `context.md` §4.

A fourth, found by adversarial review: merge seniority ranked `COASTING` below
`CONFIRMED`, so a freshly-confirmed 3-hit track could absorb — and rename — a
500-hit identity that happened to be behind the cardboard at that moment. That is
exactly the failure this project exists to prevent, and it was invisible on a
single-agent scene because nothing else ever confirmed.

## Repo layout

```
src/mcreid/
  calib/    calib.json schema, checkerboard intrinsics, AprilTag ground homography
  sim/      virtual cameras, scripted toy scenes, synthetic frame rendering
  track/    per-view tracking (torch-free CI path; GPU path lands with G-M1-2)
  fusion/   ground Kalman, appearance gallery, association, global ID manager
  eval/     identity-consistency metrics, WILDTRACK protocol
  viz/      BEV canvas, per-view overlays, demo mosaic
  cli/      mcreid-calibrate, mcreid-demo, mcreid-sync, mcreid-eval
```

- `context.md` — scope, architecture, locked conventions, design rationale
- `status.txt` — current phase, blockers, next steps
- `capture_guide.md` — the recording protocol for the real 4-camera session

## Calibration

```bash
uv run mcreid-calibrate rig --capture-dir footage/calib --square-size-m 0.025
```

Per-camera intrinsics from a checkerboard, ground-plane homography from AprilTag
36h11 markers laid flat on the floor (or four measured floor points). Writes one
`calib.json` for the whole rig. `mcreid-calibrate check` re-verifies round-trip
accuracy.

Clip alignment for independently-started phones:

```bash
uv run mcreid-sync --footage footage/take3
```

## Constraints and honesty notes

- Runtime target is >= 15 FPS aggregate on 4x 720p, RTX 4060 8 GB. The
  **end-to-end** figure is not measured yet — the GPU front-end lands with
  G-M1-2. What *is* measured: per-view tracking plus fusion costs 1.0 ms/frame
  for one person and 2.1 ms/frame for two across 4 cameras, i.e. 1.5–3 % of the
  66.7 ms budget. Effectively the whole budget is available to the detector.
- All reported numbers come from the synthetic scene. No real-footage or
  benchmark numbers are claimed yet.
- Datasets, weights, footage and room calibration are never committed.

## Licence

AGPL-3.0-only.
