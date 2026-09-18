# AMOS 22 CT · inner_1 vs outer shell boundary contrast

**Status:** COMPLETE (200 CT train cases, amos_id < 500)  
**Source:** school `/public/share/td20230405/AMOS 22/amos22`  
**Log:** school `runs/school/amos_boundary_nohup.log`

## Table (case-averaged)

| organ | μ_in | μ_out1 | **c1** | **CNR1** | μ_out3 | c3 | CNR3 | \|g\| | dir_rng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spleen | 50.6 | −79.8 | **+130.3** | 0.83 | −147.0 | +197.5 | 0.73 | 86.1 | 167.7 |
| Kidney(R) | 62.3 | −9.7 | **+72.0** | **1.04** | −18.8 | +81.1 | 1.00 | 59.4 | 34.9 |
| Kidney(L) | 58.3 | −19.4 | **+77.8** | **1.15** | −32.2 | +90.5 | 1.12 | 62.2 | 30.0 |
| Gallbladder | 26.1 | 0.5 | +25.6 | 0.33 | −14.9 | +41.0 | 0.30 | 35.3 | 78.9 |
| Esophagus | 24.9 | −2.8 | +27.7 | 0.30 | −25.6 | +50.5 | 0.32 | 52.0 | 65.5 |
| Liver | 51.1 | −75.6 | **+126.7** | 0.66 | −151.1 | +202.3 | 0.65 | 79.0 | 169.9 |
| Stomach | 7.7 | −24.2 | +31.9 | 0.32 | −58.3 | +66.1 | 0.36 | 75.7 | 57.1 |
| Aorta | 84.4 | 5.2 | **+79.2** | 0.58 | −32.7 | +117.0 | 0.52 | 75.1 | 81.2 |
| Postcava | 55.9 | 23.3 | +32.6 | 0.31 | −0.9 | +56.7 | 0.33 | 38.1 | 52.4 |
| Pancreas | 51.4 | 16.0 | +35.4 | 0.55 | 3.8 | +47.6 | 0.58 | 36.3 | 34.6 |
| Adrenal(R) | 22.0 | −5.1 | +27.1 | 0.44 | −2.2 | +24.2 | 0.32 | 40.2 | 68.7 |
| Adrenal(L) | 22.0 | −19.1 | **+41.0** | 0.79 | −20.4 | +42.4 | 0.67 | 41.1 | 41.4 |
| Duodenum | 24.5 | 11.3 | +13.1 | **0.27** | 2.8 | +21.7 | 0.31 | 50.2 | 32.8 |
| Bladder | 25.6 | 6.7 | +18.9 | 0.36 | 14.0 | +11.6 | 0.21 | 34.0 | 51.9 |
| Prostate/Uterus | 42.9 | 26.9 | +16.0 | 0.38 | 16.1 | +26.8 | 0.43 | 21.9 | 24.1 |

## Notes

1. **Absolute HU scale ≠ WORD** (Liver AMOS inner ~51 vs WORD ~125). Cross-dataset pooling needs care.
2. **Outer shells often negative HU** (air/fat outside spleen/liver/kidney) → very large signed contrast (100+). This is real tissue separability in CT, not an internal radial artifact.
3. **Highest CNR:** Kidney L/R (~1.0–1.15); solid organs + aorta strong.
4. **Weakest CNR:** Duodenum 0.27, Gallbladder/Esophagus/Bladder/Prostate ~0.3–0.4 — harder boundaries.
5. **Directional range large** for Spleen/Liver (~170) — strong anisotropy of surrounding tissue.

## Cross-dataset (qualitative)

| pattern | WORD | AMOS |
|---|---|---|
| Kidney high CNR | yes (~0.6–0.7) | yes (~1.0–1.15) |
| Weak soft-tissue boundary | Rectum | Duodenum / GB |
| Bone-like high contrast | Femur | (no bone label) |

FLARE2023 same protocol still running at last check (~600/997).

Artifacts: `results/amos_boundary_contrast.json`
