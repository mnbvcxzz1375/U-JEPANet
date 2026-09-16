#!/usr/bin/env python
"""Freeze WORD SLL 20% split files (labeled/unlabeled/val/test ids).

Does not copy images. Reads only file names from imagesTr/imagesVal/imagesTs.
Writes stable lists under --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def list_ids(dir_path: Path) -> list[str]:
    files = sorted(p.name.replace(".nii.gz", "").replace(".nii", "") for p in dir_path.glob("word_*.nii*"))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed-split", type=int, default=20260916)
    p.add_argument("--n-labeled", type=int, default=20)
    args = p.parse_args()

    root = Path(args.word_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ids = list_ids(root / "imagesTr")
    val_ids = list_ids(root / "imagesVal")
    test_ids = list_ids(root / "imagesVal")  # placeholder, overwritten below
    test_dir = root / "imagesTs"
    test_ids = list_ids(test_dir) if test_dir.exists() else []

    # Deterministic shuffle of train ids with seed_split
    import random

    rng = random.Random(args.seed_split)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)
    labeled = sorted(shuffled[: args.n_labeled])
    unlabeled = sorted(shuffled[args.n_labeled :])
    if len(labeled) != args.n_labeled:
        raise SystemExit(f"need {args.n_labeled} labeled, got {len(labeled)}")
    if len(train_ids) != len(labeled) + len(unlabeled):
        raise SystemExit("split mismatch")

    def write_list(name: str, ids: list[str]):
        (out / name).write_text("\n".join(ids) + "\n", encoding="utf-8")
        blob = "\n".join(ids).encode()
        return hashlib.sha256(blob).hexdigest()

    hashes = {
        "labeled_train_20.txt": write_list("labeled_train_20.txt", labeled),
        "unlabeled_train_80.txt": write_list("unlabeled_train_80.txt", unlabeled),
        "val_20.txt": write_list("val_20.txt", val_ids),
        "test_30.txt": write_list("test_30.txt", test_ids),
    }
    meta = {
        "word_root": str(root),
        "seed_split": args.seed_split,
        "n_train": len(train_ids),
        "n_labeled": len(labeled),
        "n_unlabeled": len(unlabeled),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "hashes_sha256": hashes,
        "labeled": labeled,
        "note": "test list is names-only; do not train/eval on test in screening",
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in meta if k != "labeled"}, indent=2))
    print("labeled:", ",".join(labeled))


if __name__ == "__main__":
    main()
