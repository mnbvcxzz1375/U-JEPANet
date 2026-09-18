# Organ boundary contrast: inner_1 vs outer shells

**Correction of prior analysis:** edge−interior *inside* the organ is only a radial  
change. Real separability needs **inner band vs outer tissue shell**.

## Metrics

| name | definition |
|---|---|
| inner_1 | organ voxels with ≤1 voxel to exterior (`mask − erode1`) |
| outer_rk | voxels outside organ within k (`dilate_k − mask`), k=1,2,3 |
| contrast_rk | `μ(inner_1) − μ(outer_rk)` (signed HU) |
| CNR_rk | `\|contrast\| / sqrt(σ_in² + σ_out²)` |
| \|g\| | mean \|∇HU\| on inner_1 |
| dir_rng | max−min of 6-direction outer means |

## WORD (100 train · labelsTr_All) — **DONE**

| organ | μ_in | μ_out1 | c1 | CNR1 | μ_out3 | c3 | CNR3 | \|g\| | dir_rng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Liver | 124.8 | 94.2 | **+30.6** | 0.47 | 89.5 | +35.3 | 0.49 | 34.3 | 41.2 |
| Spleen | 122.0 | 83.6 | **+38.4** | 0.59 | 79.8 | +42.1 | 0.58 | 44.4 | 51.3 |
| Kidney(L) | 125.3 | 88.9 | **+36.4** | **0.64** | 83.5 | +41.8 | **0.72** | 48.1 | 19.1 |
| Kidney(R) | 128.5 | 93.9 | **+34.6** | 0.60 | 89.1 | +39.4 | 0.67 | 45.6 | 23.3 |
| Stomach | 107.4 | 93.3 | +14.1 | 0.26 | 91.6 | +15.8 | 0.27 | 28.3 | 26.8 |
| Gallbladder | 116.6 | 109.5 | +7.1 | 0.19 | 110.8 | +5.7 | 0.18 | 20.8 | 45.1 |
| Esophagus | 115.3 | 108.5 | +6.8 | 0.16 | 114.6 | +0.8 | 0.17 | 26.0 | 37.3 |
| Pancreas | 131.2 | 118.3 | +12.9 | 0.24 | 114.8 | +16.4 | 0.30 | 25.6 | 29.4 |
| Duodenum | 120.6 | 112.0 | +8.6 | 0.19 | 111.3 | +9.3 | 0.19 | 24.1 | 25.8 |
| Colon | 78.1 | 75.9 | +2.2 | 0.12 | 80.5 | **−2.5** | 0.13 | 26.8 | 10.8 |
| Intestine | 92.9 | 79.1 | +13.8 | 0.32 | 79.3 | +13.6 | 0.29 | 26.9 | 9.5 |
| Adrenal | 107.1 | 95.8 | +11.3 | 0.27 | 101.6 | +5.5 | 0.20 | 27.5 | 31.9 |
| **Rectum** | 102.3 | 95.7 | **+6.6** | **0.17** | 93.7 | +8.7 | **0.21** | **19.5** | 19.3 |
| Bladder | 104.4 | 99.3 | +5.0 | 0.17 | 103.7 | +0.6 | 0.13 | 20.3 | 30.0 |
| Femur(L) | 201.7 | 165.7 | **+36.0** | 0.47 | 156.0 | +45.6 | 0.58 | 46.7 | 33.0 |
| Femur(R) | 201.2 | 165.4 | **+35.8** | 0.47 | 155.9 | +45.3 | 0.59 | 46.4 | 32.7 |

### WORD reading (outer-shell version)

- Solid high-contrast organs (Spleen/Kidney/Femur/Liver): **outer shell is 30–45 HU darker** than inner band; CNR ~0.5–0.7; \|g\| high.
- **Rectum**: contrast only **+6.6 / CNR 0.17**, \|g\| lowest among listed — weak true separability, matches JEPA ΔDice collapse better than the old internal edge−interior gap (−2.7).
- **Colon** at r=3 even goes slightly **negative** (outer brighter) — lumen/fat adjacency.
- Directional range is large for Spleen/Liver/Gallbladder (anisotropy).

## AMOS 22 CT — running (login node CPU, 50/200 at check)

## FLARE2023 labeled — running (75/997 at check)

Artifacts: `word_boundary_contrast.json` · school logs under `runs/school/`
