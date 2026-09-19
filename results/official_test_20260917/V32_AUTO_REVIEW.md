# V3.2 Aligned Global Innovation — auto-review, not launched

User brief after `948ed7f`. **Training not started** until user review.

## Design locked

- **I = P(q, Z_G) − P(q, Z_PE)** with **shared** head P (causal patient innovation).
- Hard **GridSample** at p_i (patient voxel frame; FOV audit → not raw DICOM origin).
- Fusion: **A(I) only**, RMS-normalized; **α0=0.04** (≈4% RMS, not raw 0.05).
- H0/H1 same params/init; B0 = **A0DA_GLP** isolated-loader control.
- Closed: old G2 / L_GL; no story claiming stable R1 +0.0056.

## Auto-review (2026-09-19)

| | |
|---|---|
| Design checklist | **19 PASS** |
| Unit tests 40901 | **ALL_V32_TESTS_PASS** |
| H0 S_pre | **0.0** (PE-only both sides) |
| H1 S_pre | **~0.66** (distinct global content) |
| S_fuse at α=0.04 | **~0.04** |
| WARN | R1 trajectory-sensitive; no launch; shuffle=real WORD pairs |

## Proposed launch (after user OK)

| Host | Arm |
|---|---|
| 40901 | B0 A0DA_GLP s42 |
| 40902 | H1 s42 |
| school | H0 s43 / H1 s43 / A0DA_GLP s43 |

Gate: **H1−H0 val** ≳+0.003 both seeds + real-case shuffle evidence.

## Files

- `ujepa/aligned_global_innovation.py`
- `scripts/train_v32.py`
- `tests/test_v32_innovation.py`
- `scripts/auto_review_v32.py`
