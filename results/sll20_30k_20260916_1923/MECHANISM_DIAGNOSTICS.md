# SLL20-30k mechanism diagnostics (per-organ + gradient conflict)

Date: 2026-09-17  
Checkpoints: `best.pt` from `sll20_30k_20260916_1923`  
Val: official 20 `imagesVal` whole-volume

## 1. Per-organ Dice (mean over 20 val cases)

| # | Organ | A0 | A1 | A2 | A3 | A3−A0 |
|---|---|---:|---:|---:|---:|---:|
| 1 | Liver | 0.936 | 0.928 | 0.938 | 0.934 | −0.002 |
| 2 | Right Kidney | 0.905 | 0.906 | 0.910 | 0.901 | −0.004 |
| 3 | Spleen | 0.935 | 0.931 | 0.923 | 0.918 | **−0.017** |
| 4 | Pancreas | 0.926 | 0.931 | 0.929 | 0.928 | +0.003 |
| 5 | Aorta | 0.824 | 0.806 | 0.816 | 0.813 | −0.011 |
| 6 | IVC | 0.540 | 0.520 | 0.547 | 0.508 | **−0.032** |
| 7 | Right Adrenal | 0.610 | 0.579 | 0.579 | 0.626 | **+0.017** |
| 8 | Left Adrenal | 0.742 | 0.722 | 0.729 | 0.720 | **−0.022** |
| 9 | Gallbladder | 0.524 | 0.517 | 0.510 | 0.519 | −0.006 |
| 10 | Esophagus | 0.718 | 0.732 | 0.730 | 0.725 | +0.007 |
| 11 | Stomach | 0.766 | 0.768 | 0.772 | 0.765 | −0.001 |
| 12 | Duodenum | 0.593 | 0.543 | 0.550 | 0.593 | −0.000 |
| 13 | **Left Kidney** | 0.567 | 0.567 | **0.475** | **0.443** | **−0.124** |
| 14 | Bladder | 0.864 | 0.872 | 0.877 | 0.857 | −0.008 |
| 15 | Rectum | 0.840 | 0.858 | 0.857 | **0.878** | **+0.038** |
| 16 | Left Femoral Head | 0.899 | 0.893 | 0.896 | 0.896 | −0.004 |
| | **ALL** | **0.762** | 0.754 | 0.752 | 0.751 | −0.010 |

### Reading

- **Not a clean “large↑ / small↓” story.** Liver/Kidney-R ~unchanged; small organs mixed.
- Dominant loss: **Left Kidney −0.124 (A3), −0.092 (A2)** — single-organ collapse, not a global smooth bias.
- Gains exist: Rectum +0.038, Right Adrenal +0.017 (A3).
- IVC / Left Adrenal / Spleen regress moderately.
- A1 Left Kidney is **unchanged vs A0** (0.567), so Left Kidney collapse is **JEPA-linked**, not dual-path alone.

Hypothesis update: JEPA is not uniformly “smoothing” all organs; it is **hurting at least one mid-size kidney** while slightly helping rectum/adrenal. Need to check whether Left Kidney is systematically under-attended in the 8×8×6 token grid or in masking frequency.

## 2. Gradient conflict (shared online encoder, A2/A3 best.pt, 16 batches)

| Arm | cos mean | frac cos&lt;0 | notes |
|---|---:|---:|---|
| **A2** | **−0.034** | **62.5%** | ratio ~0.1–0.3 typical; two outlier batches ratio≫1 |
| **A3** | ~0.000 | 50.0% | mixed; not as clean a conflict signal as A2 |

Interpretation:

- A2 (the “cleaner” JEPA arm) shows **mild but frequent gradient opposition**.
- This supports the joint-training interference hypothesis, but is not a smoking gun of huge cosine (−0.3).
- Effective weight `ρ = λ‖g_J‖/‖g_seg‖` is often ≲0.3·0.2≈0.06–0.1, so conflict is real but modest.
- A3 dual-path makes the picture noisier (capacity + low-freq residual + JEPA).

## 3. Implications for next experiments

1. **Do not λ-grid A3.** A1 already negative; A3 adds JEPA on a harmful path.
2. **A2 remains the only salvageable joint-training arm**, but needs either:
   - gradient gating / PCGrad (diagnostic first), or
   - **B3 pretrain→FT** (JEPA exits after stage 1) — currently running.
3. **Left Kidney** must be explained before claiming mechanism:
   - mask coverage / token support for left kidney region
   - case-level failure (is it 1–2 cases?)
4. If B3 ≥ A0, story becomes “JEPA useful as init, harmful as joint aux”.
5. If B3 ≈ A2 &lt; A0, representation itself is not helping 20L segmentation.

## Artifacts

- `per_organ_dice.json`
- `grad_conflict_a2.json`, `grad_conflict_a3.json`
- B3 run: `/data/hyc/U-JEPANet/runs/b3_pretrain_ft/` (in progress)
