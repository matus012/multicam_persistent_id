"""Emit the WILDTRACK ablation table from run summaries.

Reads the JSON written by `mcreid-wildtrack run` for each embedder and prints a
markdown table. Keeping this scripted means the numbers in the README are
regenerated from measurements rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Published MVDet numbers are deliberately NOT hardcoded here. Filling them in
# from memory would put a fabricated benchmark figure next to real measurements,
# which is worse than an obvious blank.
MVDET_ROW = (
    "| MVDet (Hou et al., published) — **trained on WILDTRACK** "
    "| _TODO: fill from paper_ | _TODO_ | _TODO_ | _TODO_ | n/a | n/a |"
)

LABELS = {
    "imagenet_resnet18": "geometric fusion + ImageNet trunk (no ReID)",
    "osnet_x1_0_msmt17": "geometric fusion + off-the-shelf ReID (zero training by us)",
}


def _fmt(value: float | None, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        if percent:
            return f"{value * 100:.1f} %"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    meta = []
    for path in args.summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("embedder", "unknown")
        label = LABELS.get(name, name)
        rows.append(
            f"| {label} | {_fmt(data.get('moda'), percent=True)} "
            f"| {_fmt(data.get('modp'), percent=True)} "
            f"| {_fmt(data.get('precision'), percent=True)} "
            f"| {_fmt(data.get('recall'), percent=True)} "
            f"| {data.get('total_id_switches')} "
            f"| {data.get('global_ids_ever_reported')} |"
        )
        meta.append(
            {
                "embedder": name,
                "frames": data.get("frames"),
                "gt_people": data.get("ground_truth_people"),
                "rmse": data.get("position_rmse_m"),
                "aggregate_fps": data.get("aggregate_fps"),
                "per_camera_fps": data.get("per_camera_fps"),
                "ids_minted": data.get("global_ids_minted"),
                "fp_tracks": data.get("false_positive_tracks"),
            }
        )

    print("| configuration | MODA | MODP | precision | recall | ID switches | IDs reported |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(row)
    print(MVDET_ROW)
    print()
    print("Supporting numbers:")
    print()
    print(
        "| configuration | frames | GT people | pos RMSE | IDs minted | FP tracks "
        "| FPS/cam | FPS agg |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for m in meta:
        print(
            f"| {LABELS.get(m['embedder'], m['embedder'])} | {m['frames']} | {m['gt_people']} "
            f"| {_fmt(m['rmse'], 2)} m | {m['ids_minted']} | {m['fp_tracks']} "
            f"| {_fmt(m['per_camera_fps'], 1)} | {_fmt(m['aggregate_fps'], 2)} |"
        )


if __name__ == "__main__":
    main()
