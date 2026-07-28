# v2 design note — synthetic multi-view data engine

**Status: NOT BUILT. v2 only. Do not implement in v1.**

v1 is deliberately zero-training: pretrained detector, pretrained ReID, geometric
fusion. Nothing in v1 consumes synthetic training data, so building a generator
now would be scope with no consumer.

This note exists so the v2 decision starts from a design instead of a blank page.

## Why v2 would want it

The v1 gate is a single-person occlusion clip. The failure mode v1 does *not*
solve is the one the crossing scene exposes: two people close together on the
ground plane, where geometry is ambiguous and a pretrained ReID embedding is the
only discriminator. Fixing that needs multi-view-trained appearance features, and
those need labelled multi-view data with:

- exact ground-truth ground-plane positions and identities
- controllable occlusion (the thing real footage cannot script repeatably)
- many identities, viewpoints, and lighting conditions
- per-camera calibration that is exact by construction

Real footage cannot supply the first and third cheaply. WILDTRACK supplies 400
annotated frames — enough to evaluate, nowhere near enough to train.

## Candidate backends

| backend | pros | cons |
|---|---|---|
| BlenderProc | mature, scriptable, good material/lighting variety, free | slower per frame, no physics-driven crowds out of the box |
| Isaac Sim | fast, physics + crowd sim, good camera models | heavy install, licence constraints, NVIDIA-only |
| Reuse P2 (CARLA) | already exists in `100_occlusion_mot` | outdoor street scenes, wrong domain for an indoor room |

Leaning BlenderProc: the target domain is a small indoor room with static
cameras, which needs asset and lighting variety far more than it needs crowd
physics.

## What it must emit

The generator's output has to drop straight into the existing contracts, not a
new format:

- one `RigCalib` per scene, exact by construction (see
  `mcreid.sim.virtual_camera.VirtualCamera.to_calib` for the pattern — it already
  derives an exact ground homography from a projection matrix)
- per-frame, per-camera person boxes with identity labels
- per-frame ground-truth world positions per identity
- per-camera visibility/occlusion fraction per identity, so occlusion-conditioned
  metrics can be computed the way `ToyScene.gt_visible` allows today

If it emits `ViewObservation` + ground truth in the shape `ToyScene` uses, the
whole evaluation stack works unchanged.

## Sequencing

1. Ship v1 (G-M1-1 .. G-M1-3).
2. Decide whether the crossing-case limitation is worth fixing at all — it is
   explicitly not a v1 gate, and a portfolio demo may not need it.
3. Only then build this, and only if a trained ReID head is the chosen fix.

## Non-goals

- Photorealism for its own sake. The consumer is a ReID embedding, not a human.
- Replacing WILDTRACK as the evaluation set. Synthetic data trains; real data
  evaluates. Reporting synthetic-only numbers would be dishonest.
