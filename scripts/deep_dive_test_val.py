#!/usr/bin/env python
"""Deep-dive: per-case and per-organ comparison across val vs test for all arms."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(r"E:\VScodeProject\U-JEPANet\results")

ORGAN = {
    1: "Liver", 2: "Spleen", 3: "Kidney(L)", 4: "Kidney(R)", 5: "Stomach",
    6: "Gallbladder", 7: "Esophagus", 8: "Pancreas", 9: "Duodenum", 10: "Colon",
    11: "Intestine", 12: "Adrenal", 13: "Rectum", 14: "Bladder", 15: "Femur(L)", 16: "Femur(R)",
}
ARMS = ["A0", "A1", "A2", "A3", "A2-L", "A2-LU"]


def load_test(arm: str) -> dict:
    p = ROOT / "official_test_20260917" / f"test_{arm}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = dict(d["arms"][arm])
    out["per_case"] = d.get("per_case", {}).get(arm, [])
    return out


def load_val(arm: str) -> dict:
    # val per-organ was computed during screening; the per-organ json has per-arm organ means only.
    # Per-case val means come from train_log.json val_history (whole-volume steps) — not per-case.
    # Use the stored per-organ means for val; per-case val is not archived in the repo.
    return {}


def organ_key_map(per_organ: dict) -> dict:
    out = {}
    for k, v in per_organ.items():
        if k == "ALL":
            out["ALL"] = v
        else:
            out[int(k)] = v
    return out


def per_case_table(arm_data: dict) -> list[dict]:
    return arm_data.get("per_case", [])


def main():
    test = {a: load_test(a) for a in ARMS}
    test = {a: d for a, d in test.items() if d}

    print("=" * 90)
    print("TEST per-organ (mean over 30 cases)")
    print("=" * 90)
    hdr = "organ".ljust(14) + " | " + " | ".join(a.rjust(7) for a in test)
    print(hdr)
    for c in range(1, 17):
        row = [f"{test[a]['per_organ'][str(c)]:.4f}" if str(c) in test[a]["per_organ"] else "—" for a in test]
        print(f"{c:02d} {ORGAN[c]:11s} | " + " | ".join(r.rjust(7) for r in row))
    print("ALL".ljust(14) + " | " + " | ".join(f"{test[a]['per_organ']['ALL']:.4f}".rjust(7) for a in test))

    # per-case ALL (mean fg dice per case)
    print()
    print("=" * 90)
    print("TEST per-case mean fg Dice (sorted worst 10 by A0)")
    print("=" * 90)
    cases = {a: {pc["case"]: pc["mean_fg_dice"] for pc in per_case_table(test[a])} for a in test}
    all_ids = sorted(set().union(*[set(c) for c in cases.values()]))
    by_a0 = sorted(all_ids, key=lambda x: cases["A0"].get(x, 9))
    print("case".ljust(12) + " | " + " | ".join(a.rjust(7) for a in test) + " | worst-organ(A0)")
    worst_organ = {}
    for pc in per_case_table(test["A0"]):
        per_cls = {int(k): v for k, v in pc["per_class"].items() if v is not None}
        if per_cls:
            worst_organ[pc["case"]] = min(per_cls, key=per_cls.get)
    for cid in by_a0[:10]:
        row = []
        for a in test:
            v = cases[a].get(cid)
            row.append(f"{v:.4f}".rjust(7) if v is not None else "—".rjust(7))
        wo = worst_organ.get(cid)
        wos = f"{ORGAN.get(wo, wo)}" if wo else "?"
        print(f"{cid:12s} | " + " | ".join(row) + f" | {wos}")

    # worst organ per case aggregated
    print()
    print("=" * 90)
    print("Which organ is each case's WORST (A0, test)")
    print("=" * 90)
    from collections import Counter
    cnt = Counter()
    for pc in per_case_table(test["A0"]):
        per_cls = {int(k): v for k, v in pc["per_class"].items() if v is not None}
        if per_cls:
            cnt[ORGAN[min(per_cls, key=per_cls.get)]] += 1
    for o, n in cnt.most_common():
        print(f"  {o:14s} {n:2d}/{len(per_case_table(test['A0']))}")

    # organ-level paired: A2-LU vs A0 and A3 vs A0
    print()
    print("=" * 90)
    print("TEST per-organ delta vs A0")
    print("=" * 90)
    print("organ".ljust(14) + " | " + " | ".join(a.rjust(9) for a in test if a != "A0"))
    for c in range(1, 17):
        base = test["A0"]["per_organ"].get(str(c))
        if base is None:
            continue
        row = []
        for a in test:
            if a == "A0":
                continue
            v = test[a]["per_organ"].get(str(c))
            row.append(f"{v - base:+.4f}".rjust(9) if v is not None else "—".rjust(9))
        print(f"{c:02d} {ORGAN[c]:11s} | " + " | ".join(row))

    # spread across cases per organ (A0)
    print()
    print("=" * 90)
    print("TEST per-organ spread (A0): mean / min / max over 30 cases")
    print("=" * 90)
    acc = {c: [] for c in range(1, 17)}
    for pc in per_case_table(test["A0"]):
        for k, v in pc["per_class"].items():
            if v is not None:
                acc[int(k)].append(v)
    print("organ".ljust(14) + " | mean |  min |  max |  std | n_zero(<0.1)")
    for c in range(1, 17):
        vs = acc[c]
        if not vs:
            continue
        n0 = sum(1 for v in vs if v < 0.1)
        print(f"{c:02d} {ORGAN[c]:11s} | {statistics.mean(vs):.4f} | {min(vs):.4f} | {max(vs):.4f} | {statistics.pstdev(vs):.4f} | {n0}")

    # zero-dice cases per arm per organ
    print()
    print("=" * 90)
    print("TEST near-zero Dice (<0.10) counts per arm/organ")
    print("=" * 90)
    print("organ".ljust(14) + " | " + " | ".join(a.rjust(7) for a in test))
    for c in range(1, 17):
        row = []
        for a in test:
            n = 0
            for pc in per_case_table(test[a]):
                v = pc["per_class"].get(str(c))
                if v is not None and v < 0.10:
                    n += 1
            row.append(str(n).rjust(7))
        print(f"{c:02d} {ORGAN[c]:11s} | " + " | ".join(row))

    # case-level worst for JEPA arms
    print()
    print("=" * 90)
    print("TEST biggest case-level regressions vs A0 (A2-LU and A3)")
    print("=" * 90)
    for arm in ["A2-LU", "A3", "A2"]:
        if arm not in cases or "A0" not in cases:
            continue
        deltas = []
        for cid in all_ids:
            a0 = cases["A0"].get(cid)
            av = cases[arm].get(cid)
            if a0 is not None and av is not None:
                deltas.append((av - a0, cid, a0, av))
        deltas.sort()
        print(f"-- {arm} (worst 6 / best 6) --")
        for d, cid, a0, av in deltas[:6]:
            print(f"   {cid:10s} A0={a0:.4f} {arm}={av:.4f}  {d:+.4f}")
        for d, cid, a0, av in deltas[-6:]:
            print(f"   {cid:10s} A0={a0:.4f} {arm}={av:.4f}  {d:+.4f}  (best)")


if __name__ == "__main__":
    main()
