# Measurement artifacts

The raw JSON behind every WILDTRACK number quoted in the top-level README and in
[`../wildtrack_results.md`](../wildtrack_results.md). They are committed so the
tables can be checked without a 6.8 GB dataset download and a GPU.

These are **metrics only** — counts, rates and timings. No dataset frames, no
crops, no imagery of any kind is redistributed here.

| file | produced by | backs |
|---|---|---|
| `wildtrack_eval_osnet.json` | `mcreid-wildtrack run --n-frames 400` (OSNet) | the headline row: MODA −118.8 %, MODP 64.1 %, precision 26.9 %, recall 69.4 %, 636 switches, 943 IDs, RMSE 0.293 m, 7.68 FPS/cam |
| `wildtrack_eval_imagenet.json` | same, `--embedder imagenet_resnet18` | the ablation row: 741 switches, 1000 IDs, 9.66 FPS/cam |
| `wildtrack_eval_osnet_geomerge.json` | same, with the unconditional geometric merge enabled | the rejected-intervention row: MODA −1.168, switches 680 (worse) |
| `reid_separation.json` | `mcreid-wildtrack` separation measurement | the embedder table: ImageNet 0.377/0.409 (sep 0.032) vs OSNet 0.525/0.623 (sep 0.098), over 35 597 same-identity and 1 519 230 different-identity cross-camera pairs |
| `wildtrack_calib_summary.json` | `mcreid-wildtrack calib-report` | the converter cross-check: 2.97–15.3 px median per camera over 2249 samples |
| `wildtrack_run_40frames.json` | `mcreid-wildtrack run --n-frames 40` | the 40-frame table in `../wildtrack_results.md`: 149 IDs reported against 63 people, 76 switches (1.21/person), RMSE 0.391 m, 80 FP tracks, coverage median 0.577, 10.6 FPS/cam |
| `wildtrack_run_40frames_geomerge.json` | same, geometric merge enabled | its ablation column: 99 switches, RMSE 0.403 m |
| `footpoint_iou0.5.json`, `footpoint_iou0.3.json`, `footpoint_iou0.1.json` | `mcreid-wildtrack footpoint --iou-threshold <t>` | the root-cause table. GT arm identical in all three (0.123 m mean, 0.207 m p90, 0 % beyond radius, 4804 pairs); detector arm 0.62 / 1.14 / 2.17 m mean as the attribution gate loosens |

## What these do and do not establish

`wildtrack_calib_summary.json` validates **our WILDTRACK→calib.json converter**,
by cross-checking the dataset's ground-plane annotations against its per-view box
annotations. It says nothing about whether WILDTRACK's own calibration is
correct — that is assumed, not tested.

The eval summaries are a **zero-shot** result: no component was trained or
fine-tuned on WILDTRACK. They are reported as failure analysis. MODA is strongly
negative because the system emits ~2.5× more ground-plane detections than there
are people, and the measured cause is occlusion-truncated detection boxes rather
than anything in the matcher. See `../wildtrack_results.md`.

## Not reproducible from this repo

Two measurements are quoted elsewhere but have **no artifact here**, because the
scripts that produced them were exploratory and were not kept. Each is flagged at
its point of use rather than left to look artifact-backed:

1. **The entry-merge false-fusion rate (40.5 %)** and **near-miss provenance error
   rate (45 %)** in `src/mcreid/fusion/dormant.py`. Recorded as the reason those
   mechanisms ship disabled. The *behaviour* they justify is pinned by
   `tests/test_fusion_dormant.py`.
2. **The 1–2 ms/frame per-view tracking + fusion cost.** No benchmark exists in
   the suite. It establishes only that detection dominates the budget.

## Partially reproduced: the foot-point table

The root-cause measurement used to be in the list above. It now has a command and
three artifacts, and the result is split:

- **GT arm: reproduced exactly.** 0.123 m mean, 0.207 m p90, 0 % beyond the
  clustering radius, 4804 camera pairs. The published 0.12 / 0.21 / 0 % stand.
- **Detector arm: confirmed in magnitude, not in exact digits.** It depends
  strongly on how permissively a detector box is attributed to an annotated
  person, which the original did not record. A strict IoU gate discards the badly
  truncated boxes that cause the worst disagreement — the statistic silently
  becomes "disagreement given a good box". The published mean of 1.60 m sits
  between the IoU 0.3 (1.14 m) and IoU 0.1 (2.17 m) runs.

The published figures were deliberately **not** overwritten with any single run.
The qualitative finding is robust at every threshold tried, and it is the finding
that matters: GT boxes agree to 0.12 m and never exceed the clustering radius,
while 58–65 % of detector-box pairs exceed the 0.35 m merge radius with a heavy
tail. The bottom edge of a detection box, not the homography, is the broken input.

The synthetic/cardboard numbers are in a different category: they need no artifact
because they regenerate deterministically from a seed in about a minute with no
dataset and no GPU — `uv run mcreid-demo synthetic --scenario cardboard --seed <s>`.
