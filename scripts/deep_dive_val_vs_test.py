#!/usr/bin/env python
"""Val per-case deep dive + val vs test organ contrast."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ORGAN = {
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder", 15: "Femur(L)", 16: "Femur(R)",
}
ARMS = ["A0", "A1", "A2", "A3", "A2-L", "A2-LU"]
VAL = Path(r"E:\VScodeProject\U-JEPANet\results\val_percase_20260917")
TEST = Path(r"E:\VScodeProject\U-JEPANet\results\official_test_20260917")


def load_val(arm: str):
    d = json.loads((VAL / f"val_{arm}.json").read_text())
    out = dict(d["arms"][arm])
    out["per_case"] = d.get("per_case", {}).get(arm, [])
    return out


def load_test(arm: str):
    d = json.loads((TEST / f"test_{arm}.json").read_text())
    out = dict(d["arms"][arm])
    out["per_case"] = d.get("per_case", {}).get(arm, [])
    return out


def worst_organ(pc) -> str | None:
    per = {int(k): v for k, v in pc["per_class"].items() if v is not None}
    if not per:
        return None
    return ORGAN[min(per, key=per.get)]


def case_map(data):
    return {pc["case"]: pc for pc in data["per_case"]}


def main():
    val = {a: load_val(a) for a in ARMS}
    test = {a: load_test(a) for a in ARMS}

    print("=" * 90)
    print("VAL ALL (20 cases)")
    print("=" * 90)
    print(" | ".join(f"{a}={val[a]['per_organ']['ALL']:.4f}" for a in ARMS))

    print()
    print("=" * 90)
    print("VAL per-case A0 sorted worst-first")
    print("=" * 90)
    vcases = case_map(val["A0"])
    ordered = sorted(vcases, key=lambda c: vcases[c]["mean_fg_dice"])
    print("case".ljust(12) + " | " + " | ".join(a.rjust(7) for a in ARMS) + " | worst-organ")
    for cid in ordered:
        row = []
        for a in ARMS:
            pc = case_map(val[a]).get(cid)
            row.append(f"{pc['mean_fg_dice']:.4f}".rjust(7) if pc else "—".rjust(7))
        wo = worst_organ(vcases[cid])
        print(f"{cid:12s} | " + " | ".join(row) + f" | {wo}")

    print()
    print("=" * 90)
    print("VAL worst-organ histogram (A0)")
    print("=" * 90)
    cnt = Counter(worst_organ(pc) for pc in val["A0"]["per_case"])
    for o, n in cnt.most_common():
        print(f"  {o:14s} {n:2d}/20")

    print()
    print("=" * 90)
    print("VAL per-organ spread (A0)")
    print("=" * 90)
    acc = {c: [] for c in range(1, 17)}
    for pc in val["A0"]["per_case"]:
        for k, v in pc["per_class"].items():
            if v is not None:
                acc[int(k)].append(v)
    print("organ".ljust(14) + " | mean |  min |  max |  std | n<0.1")
    for c in range(1, 17):
        vs = acc[c]
        print(f"{c:02d} {ORGAN[c]:11s} | {statistics.mean(vs):.4f} | {min(vs):.4f} | {max(vs):.4f} | {statistics.pstdev(vs):.4f} | {sum(1 for v in vs if v<0.1)}")

    print()
    print("=" * 90)
    print("VAL vs TEST organ mean (A0) — which organs are harder where")
    print("=" * 90)
    print("organ".ljust(14) + " | val | test | test-val")
    for c in range(1, 17):
        vv = val["A0"]["per_organ"].get(str(c))
        tv = test["A0"]["per_organ"].get(str(c))
        if vv is None or tv is None:
            continue
        print(f"{c:02d} {ORGAN[c]:11s} | {vv:.4f} | {tv:.4f} | {tv-vv:+.4f}")

    print()
    print("=" * 90)
    print("VAL biggest regressions vs A0 (A2-L / A2-LU / A3)")
    print("=" * 90)
    vmap = {a: case_map(val[a]) for a in ARMS}
    for arm in ["A2-LU", "A2-L", "A3"]:
        deltas = []
        for cid in vmap["A0"]:
            a0 = vmap["A0"][cid]["mean_fg_dice"]
            av = vmap[arm][cid]["mean_fg_dice"]
            deltas.append((av - a0, cid, a0, av, worst_organ(vmap["A0"][cid])))
        deltas.sort()
        print(f"-- {arm} worst 5 / best 5 --")
        for d, cid, a0, av, wo in deltas[:5]:
            print(f"   {cid} A0={a0:.4f} {arm}={av:.4f} {d:+.4f}  worstA0={wo}")
        for d, cid, a0, av, wo in deltas[-5:]:
            print(f"   {cid} A0={a0:.4f} {arm}={av:.4f} {d:+.4f}  worstA0={wo}  (best)")

    # A2-L vs A2-LU on shared organ for the hard val cases
    print()
    print("=" * 90)
    print("VAL: cases where A2-L and A2-LU disagree most")
    print("=" * 90)
    diffs = []
    for cid in vmap["A0"]:
        dl = vmap["A2-L"][cid]["mean_fg_dice"] - vmap["A0"][cid]["mean_fg_dice"]
        dlu = vmap["A2-LU"][cid]["mean_fg_dice"] - vmap["A0"][cid]["mean_fg_dice"]
        diffs.append((dlu - dl, cid, dl, dlu, worst_organ(vmap["A0"][cid])))
    diffs.sort()
    print("case | d(A2-L) | d(A2-LU) | LU-L | worstA0")
    for d, cid, dl, dlu, wo in diffs:
        if abs(d) >= 0.01 or True:
            print(f"{cid} | {dl:+.4f} | {dlu:+.4f} | {d:+.4f} | {wo}")

    print()
    print("=" * 90)
    print("TEST: cases where A2-L and A2-LU disagree most")
    print("=" * 90)
    tmap = {a: case_map(test[a]) for a in ARMS}
    diffs = []
    for cid in tmap["A0"]:
        dl = tmap["A2-L"][cid]["mean_fg_dice"] - tmap["A0"][cid]["mean_fg_dice"]
        dlu = tmap["A2-LU"][cid]["mean_fg_dice"] - tmap["A0"][cid]["mean_fg_dice"]
        diffs.append((dlu - dl, cid, dl, dlu, worst_organ(tmap["A0"][cid])))
    diffs.sort()
    for d, cid, dl, dlu, wo in diffs:
        print(f"{cid} | {dl:+.4f} | {dlu:+.4f} | {d:+.4f} | {wo}")


if __name__ == "__main__":
    main()
