# Official test deep-dive: per-case / per-organ

**date:** 2026-09-17  
**data:** 30-case `imagesTs` whole-volume Dice  
**val:** 20-case `imagesVal` (per-organ means archived; per-case re-eval in flight on school)

## 1. Who is low — test cases (A0)

Worst 10 of 30 by A0 mean-fg Dice:

| case | A0 | A1 | A2 | A3 | A2-L | A2-LU | worst organ (A0) |
|---|---:|---:|---:|---:|---:|---:|---|
| word_0103 | **0.603** | 0.613 | 0.597 | 0.611 | 0.577 | 0.599 | Rectum |
| word_0088 | **0.659** | 0.683 | 0.720 | 0.672 | 0.689 | 0.681 | Bladder |
| word_0021 | **0.682** | 0.681 | 0.669 | 0.624 | 0.679 | 0.665 | Adrenal |
| word_0124 | **0.690** | 0.670 | 0.666 | 0.692 | 0.665 | 0.700 | Esophagus |
| word_0074 | **0.704** | 0.692 | 0.708 | 0.688 | 0.693 | 0.692 | Gallbladder |
| word_0052 | 0.723 | 0.719 | 0.718 | 0.736 | 0.730 | 0.738 | Duodenum |
| word_0129 | 0.726 | 0.744 | 0.722 | 0.725 | 0.729 | 0.725 | Esophagus |
| word_0017 | 0.736 | 0.744 | 0.720 | 0.717 | 0.726 | 0.727 | Pancreas |
| word_0092 | 0.746 | 0.772 | 0.759 | 0.750 | 0.762 | 0.765 | Gallbladder |
| word_0097 | 0.747 | 0.778 | 0.760 | 0.763 | 0.761 | 0.769 | Rectum |

**Low cases are not “hard volume” uniformly** — each is dragged by **one soft-tissue organ**, not by liver/spleen/kidney.

## 2. Which organ is each case's bottleneck (A0, 30 cases)

| organ | #cases where it's the worst |
|---|---:|
| **Rectum** | **7/30** |
| Gallbladder | 6/30 |
| Esophagus | 6/30 |
| Adrenal | 4/30 |
| Duodenum | 4/30 |
| Bladder | 2/30 |
| Pancreas | 1/30 |
| Liver/Spleen/Kidney/Femur | **0** |

→ ALL Dice is dominated by a **small-organ / soft-tissue tail**, not by big stable organs.

## 3. Organ difficulty & variance (A0 test, n=30)

| organ | mean | min | max | std |
|---|---:|---:|---:|---:|
| Liver | 0.937 | 0.864 | 0.967 | 0.026 |
| Spleen | 0.920 | 0.721 | 0.963 | 0.048 |
| Kidney(L) | 0.900 | **0.539** | 0.963 | 0.098 |
| Kidney(R) | 0.921 | 0.806 | 0.959 | 0.039 |
| Stomach | 0.839 | 0.466 | 0.944 | 0.090 |
| Gallbladder | 0.620 | **0.105** | 0.904 | **0.223** |
| Esophagus | 0.631 | **0.143** | 0.818 | 0.176 |
| Pancreas | 0.711 | 0.329 | 0.863 | 0.161 |
| Duodenum | 0.543 | **0.053** | 0.804 | 0.167 |
| Colon | 0.750 | 0.429 | 0.897 | 0.104 |
| Intestine | 0.782 | 0.627 | 0.915 | 0.060 |
| Adrenal | 0.575 | 0.326 | 0.736 | 0.110 |
| **Rectum** | 0.588 | **0.008** | 0.831 | 0.162 |
| Bladder | 0.827 | **0.194** | 0.957 | **0.215** |
| Femur(L) | 0.846 | 0.723 | 0.934 | 0.051 |
| Femur(R) | 0.891 | 0.793 | 0.940 | 0.035 |

Near-zero (<0.10) counts:

| organ | A0 | A1 | A2 | A3 | A2-L | A2-LU |
|---|---:|---:|---:|---:|---:|---:|
| Gallbladder | 0 | 0 | 1 | 1 | 1 | 1 |
| Esophagus | 0 | 0 | 1 | 0 | 1 | 0 |
| Duodenum | 1 | 0 | 1 | 0 | 1 | 1 |
| **Rectum** | 1 | 0 | 2 | **3** | 1 | 1 |
| Bladder | 0 | 0 | 0 | 0 | 1 | 0 |

**A3 makes Rectum near-fail cases worse (3/30 <0.10).**

## 4. Arm × organ deltas vs A0 (test)

| organ | A1 | A2 | A3 | A2-L | A2-LU | reading |
|---|---:|---:|---:|---:|---:|---|
| Femur(L) | +.018 | +.029 | **+.042** | +.034 | +.029 | bone always helps |
| Femur(R) | +.010 | +.011 | +.004 | +.011 | +.009 | same, smaller |
| Esophagus | −.026 | −.017 | −.010 | +.017 | **+.024** | A2-LU fixes esophagus |
| Duodenum | +.011 | +.007 | +.002 | +.018 | +.007 | mild gain |
| **Rectum** | +.012 | **−.049** | **−.092** | −.033 | +.009 | JEPA on 100 (A2/A3) kills Rectum; **U80 pool recovers it** |
| Adrenal | −.026 | **−.049** | −.002 | −.010 | −.002 | A2 worst |
| Gallbladder | +.016 | +.006 | −.003 | −.009 | −.005 | mixed |
| Pancreas | −.009 | −.011 | **−.029** | −.021 | −.015 | A3 worst |
| Kidney(L) | −.005 | +.005 | **−.029** | +.003 | +.002 | A3 outlier |
| Spleen | −.023 | −.014 | −.026 | −.025 | −.023 | all arms slightly worse |
| Liver | −.006 | +.003 | −.004 | +.002 | +.002 | ~neutral |

### Mechanism summary

1. **Predictable bone (Femur L/R)** — every JEPA/dual-path arm improves.
2. **Variable soft tissue (Rectum, Adrenal, Pancreas)** — A2/A3 often hurt; A2-LU is the exception that recovers Rectum/Esophagus.
3. **Spleen** is the one large organ that consistently dips slightly on every modified arm — not catastrophic, but systematic.

## 5. Case-level swings (test)

### A2-LU vs A0 (the only arm above A0)

| direction | case | A0 | A2-LU | Δ |
|---|---|---:|---:|---:|
| worst | word_0054 | 0.828 | 0.808 | −0.020 |
| worst | word_0021 | 0.682 | 0.665 | −0.017 |
| best | word_0077 | 0.768 | 0.808 | **+0.040** |
| best | word_0088 | 0.659 | 0.681 | +0.023 |
| best | word_0097 | 0.747 | 0.769 | +0.022 |

Gains concentrate on already-mid cases (0088/0097 = Gallbladder/Rectum-limited). Losses are on relatively easy cases (0054).

### A3 vs A0 (worst arm)

| direction | case | A0 | A3 | Δ |
|---|---|---:|---:|---:|
| worst | word_0021 | 0.682 | 0.624 | **−0.058** |
| worst | word_0054 | 0.828 | 0.790 | −0.038 |
| best | word_0097 | 0.747 | 0.763 | +0.016 |
| best | word_0110 | 0.766 | 0.782 | +0.016 |

**word_0021 (Adrenal-limited)** is the single biggest dual-path casualty.

### A2 vs A0

| direction | case | A0 | A2 | Δ |
|---|---|---:|---:|---:|
| worst | word_0019 | 0.799 | 0.765 | −0.034 |
| best | word_0088 | 0.659 | 0.720 | **+0.062** |

A2's ALL deficit is **not broad**: it loses on easy cases and can win big on one hard Gallbladder case — high variance, not uniform damage.

## 6. Val vs test (now with per-case val re-eval)

Val per-case re-eval completed on school 4090d (jobs 382797–382802); ALL matches screening exactly
(A0 0.7617 / A1 0.7544 / A2 0.7524 / A3 0.7514 / A2-L 0.7641 / A2-LU 0.7577).

### Val worst cases (A0, n=20)

| case | A0 | worst organ |
|---|---:|---|
| word_0083 | **0.612** | Gallbladder |
| word_0048 | 0.695 | Rectum |
| word_0141 | 0.717 | Gallbladder |
| word_0080 | 0.719 | Esophagus |
| word_0085 | 0.731 | Gallbladder |

Val worst-organ histogram: Gallbladder **6/20**, Rectum 5/20, Duodenum 5/20, Esophagus 3/20.

Gallbladder on val: mean 0.540, min **0.000**, std **0.277**, 2 cases <0.10.
On test Gallbladder is much easier: mean 0.620, min 0.105.

### Val vs test organ mean (A0)

| organ | val | test | test−val |
|---|---:|---:|---:|
| **Gallbladder** | 0.540 | 0.620 | **+0.080** |
| Colon | 0.718 | 0.750 | +0.032 |
| Esophagus | 0.609 | 0.631 | +0.021 |
| Rectum | 0.567 | 0.588 | +0.021 |
| **Bladder** | 0.864 | 0.827 | **−0.038** |
| **Kidney(L)** | 0.935 | 0.900 | **−0.035** |
| Pancreas | 0.742 | 0.711 | −0.031 |

**Val is Gallbladder-heavy hard; test is more balanced.** This is a concrete mechanism for the A2-L vs A2-LU rank flip.

### Where A2-L and A2-LU disagree

**Val** — A2-L wins on Esophagus-limited cases:

| case | d(A2-L) | d(A2-LU) | LU−L | worstA0 |
|---|---:|---:|---:|---|
| word_0149 | +0.030 | +0.007 | **−0.023** | Esophagus |
| word_0080 | +0.039 | +0.020 | −0.018 | Esophagus |
| word_0083 | +0.018 | −0.001 | −0.018 | Gallbladder |

**Test** — A2-LU wins on Esophagus/Rectum-limited cases:

| case | d(A2-L) | d(A2-LU) | LU−L | worstA0 |
|---|---:|---:|---:|---|
| word_0124 | −0.025 | +0.011 | **+0.035** | Esophagus |
| word_0103 | −0.025 | −0.003 | +0.022 | Rectum |
| word_0099 | −0.007 | +0.007 | +0.014 | Duodenum |
| word_0069 | −0.006 | +0.007 | +0.013 | Bladder |

→ Same Esophagus family: val prefers L, test prefers LU. Not noise in ALL — it's **organ-mix**.

### A3 damage is case-concentrated

Val worst A3: word_0007 Gallbladder −0.037, word_0137 Duodenum −0.033, word_0035 Rectum −0.031.
Test worst A3: word_0021 Adrenal −0.058, word_0054 −0.038.

A3 can also win on the hardest Gallbladder val case (word_0083 +0.036) — high variance, negative mean.

## 7. Implications for C-series

1. Dynamic crop / CT-med should be read primarily on **Rectum / Adrenal / Pancreas / Esophagus / Gallbladder**, not on ALL alone.
2. **Do not kill U80 JEPA** based on the earlier val Δ_U — test reversed it; val's Gallbladder-heavy mix is the likely cause.
3. A3 dual-path remains closed; damage is concentrated and reproducible (Rectum −0.09 on test; Adrenal case word_0021 −0.058).
4. Next diagnostic worth running: per-case **JEPA error** on the low cases (test 0103/0088/0021/0124; val 0083/0048/0141) — does L_J spike exactly on the organ that collapses?

## Artifacts

- Scripts: `scripts/deep_dive_test_val.py`, `scripts/deep_dive_val_vs_test.py`, `scripts/eval_per_case_val.py`
- Test JSONs: `results/official_test_20260917/test_*.json` (includes `per_case`)
- Val JSONs: `results/val_percase_20260917/val_*.json`
- School jobs: test 382534–382537; val per-case 382797–382802
