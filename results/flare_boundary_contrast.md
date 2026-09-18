# FLARE2023 · inner_1 vs outer shell boundary contrast

**Status:** COMPLETE · 997 labeled imagesTr · labelsTr 13 organs  
**Source:** school `/public/share/td20230405/FLARE2023`  
**Naming:** `FLARE23_XXXX_0000.nii.gz` → label `FLARE23_XXXX.nii.gz`

## Table

| organ | μ_in | μ_out1 | **c1** | **CNR1** | μ_out3 | c3 | CNR3 | \|g\| | dir_rng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Liver | 82.4 | 13.5 | **+68.9** | 0.63 | −73.7 | +156 | 0.65 | 53.2 | 80.6 |
| Kidney(R) | 71.4 | −12.2 | **+83.5** | **1.03** | −27.7 | +99.0 | 1.12 | 88.6 | 46.3 |
| Spleen | 69.3 | −36.1 | **+105.5** | 0.87 | −116.1 | +185 | 0.83 | 81.4 | 113.6 |
| Pancreas | 55.4 | 30.2 | +25.2 | **0.28** | 21.8 | +33.7 | 0.33 | 47.3 | 43.6 |
| Aorta | 82.1 | 20.3 | **+61.7** | 0.54 | −26.4 | +108 | 0.60 | 72.3 | 118.2 |
| IVC | 80.4 | 51.9 | +28.5 | 0.36 | 31.6 | +48.8 | 0.43 | 39.1 | 63.4 |
| Adrenal(R) | 12.3 | −2.3 | +14.6 | **0.22** | 1.3 | +11.1 | 0.23 | 42.0 | 87.7 |
| Adrenal(L) | 12.9 | −12.8 | +25.7 | 0.42 | −22.2 | +35.1 | 0.52 | 46.0 | 60.4 |
| **Gallbladder** | 16.0 | 4.1 | **+11.9** | **0.14** | 5.0 | +11.0 | **0.13** | 37.2 | 86.6 |
| Esophagus | 24.1 | −1.8 | +25.8 | 0.21 | −25.0 | +49.1 | 0.25 | 49.0 | 80.9 |
| Stomach | 4.7 | −3.9 | +8.6 | 0.17 | −20.2 | +24.9 | 0.25 | 54.8 | 67.6 |
| Duodenum | 35.4 | 20.6 | +14.8 | 0.23 | 11.0 | +24.4 | 0.30 | 45.2 | 49.4 |
| Kidney(L) | 64.9 | −21.1 | **+86.0** | **1.10** | −39.4 | +104 | **1.24** | 90.7 | 37.5 |

## Cross-dataset CNR (r1) comparison

| organ | WORD | AMOS | FLARE |
|---|---:|---:|---:|
| Kidney L/R | 0.64 / 0.60 | 1.15 / 1.04 | **1.10 / 1.03** |
| Liver | 0.47 | 0.66 | 0.63 |
| Spleen | 0.59 | 0.83 | 0.87 |
| Gallbladder | 0.19 | 0.33 | **0.14** |
| Pancreas | 0.24 | 0.55 | **0.28** |
| Duodenum | 0.19 | 0.27 | 0.23 |
| Stomach | 0.26 | 0.32 | **0.17** |
| Aorta/Femur-like | Femur 0.47 | Aorta 0.58 | Aorta 0.54 |

## Conclusion (three datasets)

1. **True boundary separability (inner vs outer)** is high for kidneys/spleen/liver/vessels across all three sources; CNR kidneys consistently ≈1.0+ on AMOS/FLARE.
2. **Weak-boundary organs:** gallbladder, duodenum, stomach, esophagus, adrenal(R) — low CNR everywhere.
3. **Absolute HU means differ by dataset** (Liver: WORD 125 / AMOS 51 / FLARE 82) — do not mix raw HU without normalization checks.
4. Outer shells often sit in **air/fat (negative HU)**; signed contrast is the correct “separability” signal vs the old internal-only edge−interior.

Artifacts:
- `results/flare_boundary_contrast.json`
- `results/word_boundary_contrast.json`
- `results/amos_boundary_contrast.json`
