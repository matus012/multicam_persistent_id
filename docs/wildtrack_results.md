# WILDTRACK — real-footage validation

**Label: geometric baseline, zero training.** COCO-pretrained YOLO11x, an
ImageNet-pretrained ResNet-18 appearance trunk, and geometry. Nothing in this
pipeline has seen WILDTRACK. No fine-tuning, no ReID training.

Dataset: 7 static cameras over a public square, 1920x1080, 400 annotated frames
sampled at 2 fps, ground plane discretised over a 12 m x 36 m area. Acquired via
`scripts/download_wildtrack.py fetch` (direct EPFL link, SHA-256 pinned). Data
and reports are gitignored and never committed.

## 1. Calibration ingest — converter validated

WILDTRACK ships its own calibration, so nothing was re-calibrated. The converter
turns its `rvec`/`tvec` + camera matrices into our `calib.json`, deriving the
ground homography as `H = P[:, [0, 1, 3]]`.

`mcreid-calibrate report` does not apply here — it validates *our* AprilTag
calibrations, and WILDTRACK has no tags. The equivalent check uses the dataset's
two independent annotation streams against each other: project a person's
annotated ground position with the converted calibration, and measure the
distance to the bottom-centre of that same person's annotated box.

| camera | samples | median px | p90 px |
|---|---|---|---|
| CVLab1 | 522 | 5.7 | 12.1 |
| CVLab2 | 322 | 7.7 | 25.1 |
| CVLab3 | 419 | 8.0 | 31.8 |
| CVLab4 | 171 | 11.7 | 38.2 |
| IDIAP1 | 93 | 11.1 | 22.0 |
| IDIAP2 | 555 | 3.0 | 7.6 |
| IDIAP3 | 167 | 15.3 | 24.6 |

Floor-grid overlays (`reports/wildtrack/calib/*_grid.png`) show a 1 m metric grid
lying flat on the pavement, aligned with the paving stones, in every view.

**This validates the converter, not the dataset.** Both sides of the comparison
are WILDTRACK's own annotations, so a systematic error in their published
calibration would pass unnoticed.

## 2. Ground geometry is not the bottleneck

Measured with ground-truth boxes, so detector error is excluded:

| quantity | value |
|---|---|
| cross-camera disagreement, same person | 0.12 m mean, 0.24 m p95 |
| fraction beyond `birth_cluster_radius_m` (1.0 m) | **0.0 %** |
| per-view ground error vs dataset GT | 0.12 m mean |
| our assumed per-observation sigma | 0.40 m mean |

The projection stack is *more* accurate than the fusion stage assumes — the
opposite of the failure found on the synthetic data, where the covariance was 5x
overconfident. On WILDTRACK it is roughly 3x conservative.

## 3. The appearance model is the bottleneck, and no threshold fixes it

Cosine distance from the untrained ImageNet ResNet-18 on real WILDTRACK crops,
using GT identities (1584 crops, 12 frames):

| pair | mean distance |
|---|---|
| **same** person, different camera | **0.377** |
| **different** person, different camera | **0.408** |
| same person, same camera | 0.258 |
| different person, same camera | 0.369 |

The cross-camera separation is **0.031**. The distributions essentially overlap:

| gate | same-person accepted | different-person accepted |
|---|---|---|
| 0.34 (shipped) | 34.5 % | 21.6 % |
| 0.40 | 61.2 % | 47.2 % |
| 0.50 | 91.2 % | 86.2 % |
| 0.60 | 98.7 % | 98.2 % |

For comparison, the synthetic generator was calibrated to 0.27 same-identity vs
0.53 different-identity — a separation of 0.26, roughly **8x larger**. That toy
was modelled on a *trained* person-ReID operating point; v1 ships an *untrained*
ImageNet trunk. On real cross-view data those are not the same thing, and the
synthetic gate never exposed the difference.

This is not a tuning problem. No threshold separates overlapping distributions.

Confirmed by ablation — disabling appearance entirely changes almost nothing:

| | with appearance | geometry only |
|---|---|---|
| global IDs minted | 467 | 448 |
| ID switches | 76 | 99 |
| position RMSE | 0.39 m | 0.40 m |

## 4. Tracking results — 40 frames (20 s at 2 fps), 7 cameras, 63 GT people

| metric | value |
|---|---|
| ground-truth people | 63 |
| global IDs ever reported | 149 |
| global IDs minted (incl. tentative that never showed) | 467 |
| mean live IDs per frame | 54.2 |
| total ID switches | 76 (**1.21 per person**) |
| position RMSE | 0.39 m |
| coverage while visible (median) | 0.58 |
| false-positive tracks | 80 |
| mean detections reaching fusion per frame | 141 |

`global_ids_minted` is reported for completeness but overstates churn: 318 of the
467 were tentative tracks that died before ever being shown. `global_ids_ever_reported`
(149) is the honest identity-churn figure against 63 people.

### Runtime on the RTX 4060 Laptop

| | value |
|---|---|
| per camera, 1920x1080, yolo11x @ imgsz 1280 | **10.6 FPS** |
| aggregate over 7 cameras | **1.5 FPS** |

The M1 target is >= 15 FPS aggregate on **4x 720p**, which is a far lighter load
(4 streams instead of 7, 0.92 MP instead of 2.07 MP each). This number does not
meet that target and is not claimed to — it is 7x 1080p. Detection dominates;
per-view tracking plus fusion was separately measured at 1.0-2.1 ms/frame.

## 5. Failure cases

- **Cross-camera fusion under-merges.** 54 live IDs per frame against ~35-40
  visible people. The BEV shows many tracks supported by `[1 cam]`: each camera
  keeps its own identity because appearance cannot bridge views and geometry
  alone cannot disambiguate a dense crowd.
- **Crowd ID switches.** 1.21 switches per person over 20 s. Worst offenders are
  people who spend most of the clip mutually occluded near the square's centre.
- **False-positive tracks (80).** Reflections, statues, and partially-visible
  people at the frame edge form stable enough tracklets to confirm.
- **Low coverage (0.58 median).** Distant and heavily occluded people are
  detected inconsistently at 2 fps, and the per-view tracker's IoU association
  has no temporal continuity to exploit across 0.5 s gaps.
- **Detection itself is strong.** ~141 confirmed view-tracks per frame across 7
  cameras. The weakness is association, not perception.

## 6. What this means for the project

The cardboard demo (G-M1-1/G-M1-2) is a *single-person* occlusion scenario in a
small room, where geometry does nearly all the work and appearance only has to
distinguish one person from an empty room. That case is well served by this
stack.

WILDTRACK is the opposite regime: dozens of similar-looking people, wide
baselines, 2 fps. Here the untrained appearance model contributes ~nothing, and
the honest conclusion is that **a trained ReID embedder is required for crowds** —
which is exactly the v2 decision anticipated in
`scripts/README_v2_synthetic_engine.md`, now with a measured justification rather
than an assumption.

## 7. Update — OSNet embedder, full 400-frame protocol

The ImageNet trunk was replaced with **OSNet x1.0 trained for person ReID on
MSMT17** (MIT licence, authors' weights, SHA-256 pinned). MSMT17 is a different
domain from WILDTRACK, so the evaluation remains zero-shot. Nothing is trained
by us; the ImageNet trunk stays selectable as an ablation.

Cross-camera separation, re-measured on the same crops:

| embedder | same person | different person | separation | best Youden gate |
|---|---|---|---|---|
| ImageNet ResNet-18 | 0.377 | 0.409 | 0.032 | 0.38 → 53 % / 38 % |
| OSNet MSMT17 | 0.525 | 0.623 | **0.098** | 0.56 → 56 % / 20 % |

At a tight gate the difference is stark: at 0.40 OSNet accepts 16.9 % of true
pairs and only **0.5 %** of false ones, against 61 % / 47 % for ImageNet.

### Full protocol, 400 frames, 7 cameras, 313 identities

| configuration | MODA | MODP | precision | recall | ID switches | IDs reported |
|---|---|---|---|---|---|---|
| ImageNet trunk (no ReID) | −118.7 % | 64.2 % | 26.9 % | 69.3 % | 741 | 1000 |
| OSNet, zero training by us | −118.8 % | 64.1 % | 26.9 % | 69.4 % | 636 | 943 |

OSNet cuts ID switches by 14 % and reported identities by 6 %. It changes MODA,
MODP, precision and recall by nothing, because those are dominated by duplicate
detections rather than by identity confusion.

### Root cause of the duplicates — the decisive measurement

| same person, two cameras | GT boxes | detector boxes |
|---|---|---|
| mean disagreement | 0.12 m | **1.60 m** |
| p50 | — | 0.49 m |
| p90 | 0.21 m | **4.50 m** |
| beyond 0.35 m | — | 64 % |
| beyond 1.00 m (clustering radius) | 0 % | **31 %** |

The homography is fine. The *input* to it is not: a detection box bottom edge is
the ground-contact point only when the feet are visible, and in a crowd they are
occluded by whoever stands in front, truncating the box high. A third of the same
person's detections consequently land over a metre apart, beyond any clustering
radius.

Three interventions were tried and measured; all were essentially null, which is
consistent with the geometry — not appearance — being the broken input:

| intervention | MODA | ID switches |
|---|---|---|
| baseline (OSNet) | −1.188 | 636 |
| unconditional geometric merge < 0.35 m | −1.168 | 680 |
| appearance weight 0.6 → 0.2, cost ceiling raised | no change on the synthetic gate | — |

The geometric merge is retained in the code but **disabled by default**, since
the measurement did not support shipping it: it can only address the 36 % of
duplicate pairs that fall inside its radius, and it made identity slightly worse.

The fix is a better ground-contact estimate — inferring the foot point from box
height and assumed stature under the known camera geometry, rather than trusting
the bottom edge — and that is the top item for the next session.

## 8. Not done yet

- Published MVDet comparison numbers are deliberately left blank rather than
  recalled approximately.
- A robust ground-contact estimator (see §7) — the single change most likely to
  move MODA.
- IDF1 proper. Identity is currently reported as switch counts and reported-ID
  counts against ground truth, not as the IDF1 formulation.

## Reproduce

```bash
python scripts/download_wildtrack.py fetch
```

```bash
uv run mcreid-wildtrack calib-report --root data/wildtrack_full
```

```bash
uv run mcreid-wildtrack run --root data/wildtrack_full --n-frames 40
```
