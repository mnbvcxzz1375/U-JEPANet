#!/usr/bin/env python
"""Auto-review V3.2 implementation vs user design brief (948ed7f review)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(r"E:\VScodeProject\U-JEPANet")
ok, warn, fail = [], [], []


def check(cond, msg, level="FAIL"):
    if cond:
        ok.append(msg)
    elif level == "WARN":
        warn.append(msg)
    else:
        fail.append(msg)


src = (ROOT / "ujepa/aligned_global_innovation.py").read_text(encoding="utf-8")
train = (ROOT / "scripts/train_v32.py").read_text(encoding="utf-8")

for p in [
    "ujepa/aligned_global_innovation.py",
    "scripts/train_v32.py",
    "tests/test_v32_innovation.py",
]:
    check((ROOT / p).exists(), f"exists {p}")

# Design requirements
check("I = zhat_g - zhat_a" in src.replace(" ", "") or "zhat_g - zhat_a" in src, "innovation I = ZhatG - ZhatA")
check("rms_norm_delta" in src, "RMS-normalized fusion")
check("alpha_init" in src and "0.04" in src or "0.03" in src, "nonzero alpha_init ~0.03-0.05")
check("GridSample" in src or "grid_sample" in src, "hard aligned grid_sample")
check("Z_L" not in src.split("innov_merge")[-1][:200] or "z_l" not in src.split("delta = self.innov_merge")[0][-300:], "fusion path should not concat Z_L residual")
# more precise: innov_merge input is I only
check("self.innov_merge(I_v)" in src or "innov_merge(I" in src, "injection A(I) only")
check("use_image_global" in src and "H0" in src and "H1" in src, "H0/H1 causal pair")
check("S_pre" in src and "S_fuse" in src, "mechanism metrics")
check("seed + 10000" in train, "isolated DataLoader RNG in v32 trainer")
check("A0DA" in train, "A0DA_GLP control arm")
check("evaluate_word_whole_volume_glp" in train, "GLP val path")
check("mech_log" in train, "train-time mechanism logging")

# Bugs to flag
if "rand(...)" in (ROOT / "scripts" / "diag_glp_gate.py").read_text(encoding="utf-8"):
    warn.append("old diag_glp_gate uses synthetic rand+InstanceNorm — V3.2 needs real-case shuffle")

# raw DICOM origin
if "origin" in src and "world" in src.lower():
    warn.append("check body-centered coords not raw DICOM world origin")
check("gvox / fs" in src or "full_shape" in src, "coords in patient voxel frame (not raw world origin)")

for p in ["ujepa/aligned_global_innovation.py", "scripts/train_v32.py", "tests/test_v32_innovation.py"]:
    try:
        ast.parse((ROOT / p).read_text(encoding="utf-8"))
        ok.append(f"syntax {p}")
    except SyntaxError as e:
        fail.append(f"syntax {p}: {e}")

# scientific caveats
warn.append("R1 +0.0056 is trajectory-sensitive (R1_GLP did not reproduce); story must not claim stable R1 gain")
warn.append("Do not launch until user review; G2 still closed")
warn.append("Shuffle diagnostic must use real WORD whole-CT pairs, not synthetic noise")

print("==== PASS ====")
for m in ok:
    print("  OK", m)
print("==== WARN ====")
for m in warn:
    print("  WARN", m)
print("==== FAIL ====")
for m in fail:
    print("  FAIL", m)
print("N_PASS", len(ok), "N_WARN", len(warn), "N_FAIL", len(fail))
if fail:
    raise SystemExit(1)
print("AUTO_REVIEW_PASS_WITH_WARNINGS")
