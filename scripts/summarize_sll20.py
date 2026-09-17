#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

RUN = Path("/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923")
rows = []
for arm in ["a0", "a1", "a2", "a3"]:
    p = RUN / arm / "train_log.json"
    d = json.loads(p.read_text())
    vals = d.get("val_history") or []
    hist = d.get("history") or []
    last = hist[-1] if hist else {}
    # alpha from history if present
    alphas = [h.get("alpha") for h in hist if "alpha" in h]
    rows.append(
        {
            "arm": arm,
            "steps": d.get("steps"),
            "elapsed_sec": d.get("elapsed_sec"),
            "best_val": d.get("best_mean_fg_dice"),
            "best_step": max(vals, key=lambda v: v.get("mean_fg_dice", -1)).get("step")
            if vals
            else None,
            "final_val": vals[-1].get("mean_fg_dice") if vals else None,
            "val_curve": [
                {"step": v.get("step"), "mean_fg_dice": v.get("mean_fg_dice")} for v in vals
            ],
            "last_seg": last.get("l_seg"),
            "last_jepa": last.get("l_jepa"),
            "last_lambda": last.get("lambda_j"),
        }
    )

out = {
    "run_root": str(RUN),
    "protocol": "WORD-SLL-20% dual_loader whole_val seed42 30k",
    "commit": "5ac7f87",
    "labeled_exposure": "30k seg updates each arm",
    "arms": rows,
}
print(json.dumps(out, indent=2))
(RUN / "summary.json").write_text(json.dumps(out, indent=2))
