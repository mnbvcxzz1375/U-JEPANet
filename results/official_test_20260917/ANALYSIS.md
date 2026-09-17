# Official test complete C-series + attribution

**date:** 2026-09-17  
**protocol:** WORD official `imagesTs` 30 cases, whole-volume 128×128×96 stride 64×64×48  
**selection:** locked `best.pt` from imagesVal screening; no test-based reselection  
**authorization:** user, 2026-09-17

## ALL Dice summary

| Arm | crop | CT-med seg | JEPA | val best | **test ALL** | Δ vs A0 |
|---|---|---|---|---:|---:|---:|
| A0 | fixed | no | no | 0.7617 | 0.7674 | — |
| A2 | fixed | no | yes | 0.7524 | 0.7636 | −0.0039 |
| A3 | fixed dual-path | no | yes | 0.7514 | 0.7580 | −0.0094 |
| A2-LU | fixed | no | yes (20L+80U) | 0.7577 | 0.7700 | +0.0026 |
| **A0D** | **dynamic** | no | **no** | 0.8042 | **0.8110** | **+0.0436** |
| C1 | dynamic | no | yes | 0.8050 | 0.8118 | +0.0444 |
| C2 | dynamic | yes | yes | 0.8105 | 0.8169 | +0.0495 |
| C4 | dynamic | yes | yes + weak/strong | 0.8177 | 0.8168 | +0.0494 |
| **C3** | dynamic | yes | yes + same aug | 0.8081 | **0.8203** | **+0.0529** |

## Attribution (test ALL)

```
A0 fixed, no JEPA          0.7674
A0D dynamic, no JEPA       0.8110   ← +0.0436 from dynamic crop ALONE
C1  dynamic + JEPA         0.8118   ← +0.0008 JEPA on top of dynamic crop
C2  + CT-med (seg)         0.8169   ← +0.0059 vs A0D
C3  + CT-med (seg+JEPA)    0.8203   ← +0.0093 vs A0D  (best)
C4  + weak/strong views    0.8168   ← +0.0058 vs A0D  (no extra over C3)
```

| Source of gain | Δ test | share of C3−A0 |
|---|---:|---:|
| Dynamic crop | +0.0436 | **~82%** |
| CT-med (seg ± JEPA) | +0.005~0.009 | ~11–18% |
| JEPA alone (C1−A0D) | +0.0008 | **~2%** |
| JEPA specialized views (C4−C3) | −0.0035 | **0 / negative** |

## Pre-registered gates

| Gate | Result |
|---|---|
| C1 > C0 | **PASS** (val & test) — fixed crop is the main bottleneck |
| C2 > C1 | **PASS** (val & test) — CT-med helps |
| C3 > C2 | **PASS** on test (+0.0034); val C3 < C2 slightly |
| C4 > C3 | **FAIL** on test (−0.0035) — specialized weak/strong JEPA views not useful |
| JEPA on dynamic crop (C1 vs A0D) | **~tie** — JEPA not justified by C1 |

## Per-organ (test)

| Organ | A0 | A0D | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|---:|
| Liver | 0.937 | ~0.94 | 0.950 | 0.953 | 0.954 | 0.951 |
| Gallbladder | 0.620 | ~0.71 | 0.719 | 0.707 | 0.711 | 0.661 |
| Esophagus | 0.631 | ~0.69 | 0.690 | 0.712 | 0.706 | 0.711 |
| Duodenum | 0.543 | ~0.60 | 0.611 | 0.601 | 0.628 | 0.613 |
| Adrenal | 0.575 | ~0.62 | 0.619 | 0.632 | 0.632 | 0.607 |
| **Rectum** | 0.588 | ~0.66 | 0.657 | 0.687 | 0.705 | **0.743** |
| Femur(L) | 0.846 | ~0.90 | 0.914 | 0.909 | 0.911 | 0.914 |

C4 lifts Rectum highest but loses Gallbladder — ALL ties C2.

## Screening val vs test rank flip (fixed-crop only)

| Arm | val | test |
|---|---|---|
| A2-L | 0.7641 | 0.7663 |
| A2-LU | 0.7577 | **0.7700** |

Val was Gallbladder-heavy (A0 val GB 0.540 vs test 0.620). Do not treat val Δ_U as a test claim.

## Conclusions (screening, single seed, 30k)

1. **Dynamic crop is the dominant fix.** A0D alone recovers +0.044 test Dice.
2. **Current JEPA implementation adds almost nothing on top of dynamic crop** (C1−A0D ≈ +0.001).
3. **CT intensity augmentation is the second real lever** (+0.006–0.009).
4. **Specialized JEPA weak/strong views do not help** (C4≤C3 on test).
5. Dual-path remains closed (A3 −0.009).
6. A2-LU vs A2-L val ordering did not transfer to test — organ mix, not ALL noise.

## What this does *not* license

- No test-based hyperparameter selection.
- No paper claim that JEPA is useless — only that *this* I-JEPA-style deep head, under 30k SLL20, does not beat dynamic-crop + CT-med without it.
- C3 best (+0.053 vs A0) still lacks multi-seed, HD95/NSD/ASSD, and PL-Seg/nnU-Net comparators.

## Artifacts

- `results/official_test_20260917/test_{A0,A1,A2,A3,A2-L,A2-LU,A0D,C1,C2,C3,C4}.json`
- `results/val_percase_20260917/val_*.json`
- Checkpoints: school `runs/school/ujepa-*`; 40901 `runs/c_ladder_40901/{C3,C4}`
- Scripts: `eval_official_test.py`, `eval_per_case_val.py`, `deep_dive_*.py`, `school_sbatch_*`, `train_c_ladder.py` (A0D)

## Jobs

| Arm | Host | Job |
|---|---|---|
| C1 train/test | school | 381886 / 382955 |
| C2 train/test | school | 382006 / 383257 |
| C3/C4 train/test | 40901 | c_ladder_40901 GPU0/1 |
| A0D train/test | school | 383177 / 383493 |
| A0–A3/A2-L/LU test | 40901 + school | 382534–382537 |
