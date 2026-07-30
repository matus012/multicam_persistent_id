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

Two development measurements are quoted in source comments but have **no artifact
here**, because the harness that produced them was exploratory and was not kept:
the entry-merge false-fusion rate (40.5 %) and the near-miss provenance error rate
(45 %) in `src/mcreid/fusion/dormant.py`. They are recorded there as the reason
those mechanisms ship disabled, and are flagged in that docstring as not citable.
The *behaviour* they justify is pinned by `tests/test_fusion_dormant.py`.
