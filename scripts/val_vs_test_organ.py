#!/usr/bin/env python
"""Compare val vs test per-organ for A0-A3, plus val-test organ-level rank flips."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\VScodeProject\U-JEPANet\results")
ORGAN = {
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder", 15: "Femur(L)", 16: "Femur(R)",
}


def main():
    val = json.loads((ROOT / "sll20_30k_20260916_1923" / "per_organ_dice.json").read_text())
    test_files = {a: ROOT / "official_test_20260917" / f"test_{a}.json" for a in ["A0","A1","A2","A3","A2-L","A2-LU"]}
    test = {}
    for a, p in test_files.items():
        d = json.loads(p.read_text())
        test[a] = d["arms"][a]["per_organ"]

    # val arms only A0-A3 in that file
    val_arms = val.get("arms", {})
    print("=" * 90)
    print("VAL vs TEST per-organ (A0-A3)")
    print("=" * 90)
    print("organ".ljust(14) + " | " + " | ".join(f"{a}val {a}tst".center(15) for a in ["A0","A1","A2","A3"]))
    for c in range(1, 17):
        row = []
        for a in ["A0","A1","A2","A3"]:
            v = val_arms.get(a, {}).get(str(c), val_arms.get(a, {}).get(c))
            t = test[a].get(str(c))
            if v is None or t is None:
                row.append("—".center(15))
            else:
                row.append(f"{float(v):.3f}/{float(t):.3f}".center(15))
        print(f"{c:02d} {ORGAN[c]:11s} | " + " | ".join(row))

    # organ-level test-val delta
    print()
    print("=" * 90)
    print("TEST − VAL per-organ (positive = better on test)")
    print("=" * 90)
    print("organ".ljust(14) + " | " + " | ".join(a.rjust(8) for a in ["A0","A1","A2","A3"]))
    for c in range(1, 17):
        row = []
        for a in ["A0","A1","A2","A3"]:
            v = val_arms.get(a, {}).get(str(c), val_arms.get(a, {}).get(c))
            t = test[a].get(str(c))
            if v is None or t is None:
                row.append("—".rjust(8))
            else:
                row.append(f"{float(t)-float(v):+.3f}".rjust(8))
        print(f"{c:02d} {ORGAN[c]:11s} | " + " | ".join(row))

    # hard-organ band (small/variable) vs large organ band
    print()
    print("=" * 90)
    print("Hard-organ band (small/variable) vs Large-organ band — TEST ALL contribution")
    print("=" * 90)
    hard = [6, 7, 9, 12, 13]  # Gallbladder Esophagus Duodenum Adrenal Rectum
    large = [1, 2, 3, 4, 5, 11, 14, 16]
    bone = [15, 16]
    for a in ["A0","A1","A2","A3","A2-L","A2-LU"]:
        hs = sum(test[a][str(c)] for c in hard) / len(hard)
        ls = sum(test[a][str(c)] for c in large) / len(large)
        print(f"{a:6s}  hard={hs:.4f}  large={ls:.4f}  hard-large={hs-ls:+.4f}")

    # rank correlation of ALL across arms val vs test
    print()
    print("=" * 90)
    print("ALL Dice val vs test (A0-A3)")
    print("=" * 90)
    for a in ["A0","A1","A2","A3"]:
        v = val_arms[a].get("ALL", val_arms[a].get("ALL"))
        t = test[a]["ALL"]
        print(f"{a}: val={v:.4f} test={t:.4f}  test-val={t-float(v):+.4f}")

    # A2-L/LU val numbers from earlier (not in per_organ_dice.json)
    print()
    print("A2-L/A2-LU val (from A2L_vs_A2LU.md): L=0.7641 LU=0.7577")
    print(f"A2-L/A2-LU test: L={test['A2-L']['ALL']:.4f} LU={test['A2-LU']['ALL']:.4f}")
    print(f"val Δ_U={0.7577-0.7641:+.4f}  test Δ_U={test['A2-LU']['ALL']-test['A2-L']['ALL']:+.4f}")


if __name__ == "__main__":
    main()
