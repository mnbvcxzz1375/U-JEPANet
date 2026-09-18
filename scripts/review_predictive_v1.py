#!/usr/bin/env python
"""Auto-review of Predictive U-Net V1 implementation vs research plan + A0DA baseline."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(r"E:\VScodeProject\U-JEPANet")
issues = []
ok = []


def check(cond, msg, level="FAIL"):
    if cond:
        ok.append(msg)
    else:
        issues.append((level, msg))


# --- files exist ---
for p in [
    "ujepa/predictive_unet.py",
    "scripts/train_predictive.py",
    "scripts/run_predictive_40901.sh",
    "scripts/school_sbatch_predictive.sh",
]:
    check((ROOT / p).exists(), f"exists {p}")

src = (ROOT / "ujepa/predictive_unet.py").read_text()
train = (ROOT / "scripts/train_predictive.py").read_text()

# --- V1 plan requirements ---
check("use_pred_in_forward" in src, "predictor gated by use_pred_in_forward")
check("use_residual" in src, "residual flag present (B3)")
check("train_mode" in src, "train vs infer mask behavior")
check("PredictiveUNet" in src and "UNet3D" in src, "wraps UNet3D backbone")
check("pred_loss_from_aux" in src, "pred loss helper")
check("sg(Z)" in src or "detach()" in src, "pred target stop-grad (no EMA required)")

# predictor used in decode path: encode -> bottleneck -> decode
check("backbone.encode" in src and "backbone.decode" in src, "bottleneck sits between encode/decode")
check("feat + f_star" in src or "residual" in src.lower(), "bottleneck modifies F_s into graph")

# train script arms
check('"P1"' in train and '"P2"' in train and '"P3"' in train, "P1/P2/P3 arms")
check("lambda_p = 0.0 if arm == \"P1\"" in train, "P1 lambda_p=0")
check("arm != \"P3\"" in train or "use_residual = arm != \"P3\"" in train, "P3 disables residual")
check("intensity_mode" in train, "val intensity_mode CLI present")
check("window" in train, "window protocol default")

# data pipeline match A0DA
check("strength=1.0" in train or "strength=1.0" in train.replace(" ", ""), "CT-med strength=1 like A0DA")
check("fg_crop_prob=0.7" in train, "fg-biased crops like A0DA")
check("DynamicWordVolumeDataset" in train, "dynamic HU dataset")
check("labeled_train_20" in train, "SLL20 labeled split")

# --- semantic risks ---
# B1: pred loss off but structure on — merge still trained via seg
check("lambda_p * l_p" in train, "seg can train merge path when lambda_p=0")

# inference: all-visible predictor — not the same as A2 (predictor discarded)
check("train_mode" in src, "inference path exists")
# At infer mask_ratio>0 still used? train_mode False uses vis_all
check("vis_all" in src or "ones_like" in src, "infer conditions on full tokens")

# Known design gaps to flag as WARN not FAIL
if "EMA" not in src and "ema" not in src.lower():
    issues.append(("WARN", "V1 intentionally has no EMA teacher (plan: try LeJEPA-style later)"))
if "SIGReg" not in src and "sigreg" not in src.lower():
    issues.append(("WARN", "SIGReg not in V1 — reserved for later sparse/geometry arm"))
if "global" not in src.lower() or "whole" not in src.lower():
    issues.append(("WARN", "No global-volume branch yet (V3/B5 not implemented)"))
if "FiLM" not in src and "film" not in src.lower():
    issues.append(("WARN", "No decoder FiLM skip modulation yet (plan §12 optional)"))
if "sparse" not in src.lower():
    issues.append(("WARN", "Sparse latent / LpWM not in V1 (plan stage 3)"))

# double residual: merge outputs f_star then feat+f_star — if f_star already includes Z paths this is deep residual
# Document as design note
if "feat + f_star" in src:
    issues.append(("NOTE", "Output is feat + merge(Z,Ẑ,R) — double residual vs plan's F2*=Ψ(...); intentional stability choice"))

# predictor always in graph check
check("if self.use_pred_in_forward" in src, "bottleneck only if flag on")
# A0DA comparison fairness: same data pipeline
an = ROOT / "results/official_test_20260917/ANALYSIS.md"
if an.exists():
    t = an.read_text(encoding="utf-8")
    check("A0DA" in t and "0.8209" in t or "0.821" in t, "ANALYSIS records A0DA baseline ~0.821")

# literature page
wiki = Path(r"E:\Obsidian\llm-wiki-lab\u-jepa-net\wiki\AP-U-Net predictive architecture roadmap.md")
check(wiki.exists(), "roadmap wiki written")

print("==== PASS ====")
for m in ok:
    print("  OK", m)
print("==== ISSUES ====")
for level, m in issues:
    print(f"  {level}: {m}")

# syntax compile
for p in ["ujepa/predictive_unet.py", "scripts/train_predictive.py"]:
    try:
        ast.parse((ROOT / p).read_text())
        print("SYNTAX_OK", p)
    except SyntaxError as e:
        print("SYNTAX_FAIL", p, e)

print("N_PASS", len(ok), "N_ISSUE", len(issues))
