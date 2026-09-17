# SLL20-30k mechanism diagnostics (per-organ + gradient conflict)

**2026-09-17 correction:** WORD organ names use **PLS4MIS order** (not the earlier wrong map).  
Numbers in `per_organ_dice.json` are unchanged; only labels were wrong.

## Corrected per-organ Dice (mean over 20 val)

| # | Organ | A0 | A1 | A2 | A3 | A2−A0 | A3−A0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Liver | 0.936 | 0.928 | 0.938 | 0.934 | +0.002 | −0.002 |
| 2 | Spleen | 0.905 | 0.906 | 0.910 | 0.901 | +0.005 | −0.004 |
| 3 | Kidney(L) | 0.935 | 0.931 | 0.923 | 0.918 | −0.012 | −0.017 |
| 4 | Kidney(R) | 0.926 | 0.931 | 0.929 | 0.928 | +0.003 | +0.003 |
| 5 | Stomach | 0.824 | 0.806 | 0.816 | 0.813 | −0.008 | −0.011 |
| 6 | Gallbladder | 0.540 | 0.520 | 0.547 | 0.508 | +0.008 | **−0.032** |
| 7 | Esophagus | 0.610 | 0.579 | 0.579 | 0.626 | −0.030 | +0.017 |
| 8 | Pancreas | 0.742 | 0.722 | 0.729 | 0.720 | −0.013 | −0.022 |
| 9 | Duodenum | 0.524 | 0.517 | 0.510 | 0.519 | −0.015 | −0.006 |
| 10 | Colon | 0.718 | 0.732 | 0.730 | 0.725 | +0.011 | +0.007 |
| 11 | Intestine | 0.766 | 0.768 | 0.772 | 0.765 | +0.006 | −0.001 |
| 12 | Adrenal | 0.593 | 0.543 | 0.550 | 0.593 | **−0.043** | −0.000 |
| 13 | **Rectum** | 0.567 | 0.567 | **0.475** | **0.443** | **−0.092** | **−0.124** |
| 14 | Bladder | 0.864 | 0.872 | 0.877 | 0.857 | +0.013 | −0.008 |
| 15 | **Femur(L)** | 0.840 | 0.858 | 0.857 | **0.878** | +0.018 | **+0.038** |
| 16 | Femur(R) | 0.899 | 0.893 | 0.896 | 0.896 | −0.003 | −0.004 |
| | **ALL** | **0.762** | 0.754 | 0.752 | 0.751 | −0.009 | −0.010 |

### Corrected reading

- **Dominant negative transfer: Rectum** (soft tissue, low contrast, variable morphology).
- **Dominant positive transfer: Femur(L)** (high-contrast bone, stable shape/location).
- Gallbladder / Adrenal / Pancreas also hurt on JEPA arms (variable soft tissue / small).
- Large stable organs (Liver, Spleen, Kidney-R) ~neutral.

This **supports** `predictability ≠ discriminability`:

| Structure type | Predictability | ΔDice (JEPA arms) |
|---|---|---|
| High-contrast stable bone (Femur L) | high | **positive** |
| Variable soft tissue (Rectum) | low | **strongly negative** |

A1 does **not** show Rectum collapse (0.567 = A0), so Rectum damage is **JEPA-linked**.

## Gradient conflict (unchanged numbers; interpretation tightened)

| Arm | cos mean | frac cos&lt;0 | typical ‖g_J‖/‖g_seg‖ |
|---|---:|---:|---|
| A2 | −0.034 | 62.5% | ~0.1–0.3 (median; **do not use ratio_mean 9.45**) |
| A3 | ~0.000 | 50% | noisier |

With λ=0.3, effective ρ ≈ 0.03–0.09 on typical batches — **frequent but weak** conflict.

**B3 (pretrain→FT, g_J=0 during FT) still < A0** → conflict is **not** the main explanation.

## Core hypothesis (to verify, not concluded)

$$
Z = Z_P + Z_D,\quad
\text{JEPA rewards } Z_P,\quad
\text{seg needs } Z_D.
$$

Predictor optimal ≈ $E[Z_{tar}|Z_{ctx}]$, which **erases hard-to-predict residual** that may carry boundaries / patient-specific morphology.

## Priority next (user-locked)

1. ~~Fix organ mapping~~ **DONE**
2. **A2-L vs A2-LU** — does 80U help at all?
3. B3 representation health: variance / effective rank / linear probe (frozen E0–E2)
4. Per-organ **JEPA error / mask coverage**: test $E[L_J^{Rectum}] > E[L_J^{Femur}]$ correlating with ΔDice
5. Pause A3/dual-path
6. If continuing JEPA: structure-aware weights or predictable+residual split — **not** λ grid on A3

## Artifacts

- `per_organ_dice.json` (names corrected)
- `grad_conflict_{a2,a3}.json`
- `B3_ANALYSIS.md`
- remote: `/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/`
