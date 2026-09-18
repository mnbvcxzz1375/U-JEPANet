# AMOS 22 organ HU statistics (CT train)

**Dataset:** AMOS 22 at `/public/share/td20230405/AMOS 22`  
**Source zip:** `amos22.zip` · job `385351` on school `hpc_gpu`  
**Scope:** `imagesTr` + `labelsTr`, **CT only** (amos_id &lt; 500), **200 cases**  
**Edge:** mask minus 1× 6-connectivity erosion  
**Labels:** official AMOS `dataset.json`

## CT-only table (voxel-pooled HU)

| Organ | mean HU | edge HU | interior HU | edge−interior | #voxels | #cases |
|---|---:|---:|---:|---:|---:|---:|
| Spleen | 73.3 | 46.1 | 79.8 | **−33.7** | 22.6M | 198 |
| Kidney(R) | 77.9 | 52.9 | 84.7 | **−31.8** | 17.0M | 200 |
| Kidney(L) | 73.6 | 48.6 | 80.4 | **−31.8** | 17.6M | 199 |
| Gallbladder | 25.8 | 24.0 | 26.8 | −2.8 | 3.3M | 189 |
| Esophagus | −6.1 | 25.7 | −29.7 | **+55.4** | 1.8M | 199 |
| Liver | 68.2 | 47.9 | 71.3 | **−23.4** | 152.4M | 200 |
| Stomach | **−103.2** | 5.0 | −130.1 | **+135.1** | 39.7M | 198 |
| Aorta | 157.7 | 95.2 | 173.8 | **−78.7** | 13.2M | 200 |
| Postcava | 68.5 | 52.9 | 73.3 | −20.4 | 7.8M | 200 |
| Pancreas | 59.9 | 47.7 | 67.7 | −20.0 | 8.3M | 200 |
| Adrenal(R) | 31.5 | 20.9 | 56.1 | **−35.3** | 0.39M | 199 |
| Adrenal(L) | 32.5 | 20.6 | 57.4 | **−36.8** | 0.46M | 200 |
| Duodenum | −1.0 | 21.0 | −13.7 | **+34.7** | 6.6M | 200 |
| Bladder | 18.6 | 22.6 | 17.1 | +5.5 | 14.5M | 195 |
| Prostate/Uterus | 49.7 | 41.7 | 52.9 | −11.1 | 6.1M | 192 |

## Notes vs WORD

1. **Absolute HU scale differs a lot** from WORD (e.g. Liver AMOS 68 vs WORD 152; Spleen 73 vs 162). Do not pool WORD and AMOS HU without checking per-dataset intensity conventions / rescaling.
2. **Air-containing organs** (Stomach mean −103, Duodenum ~0, Esophagus mean −6) dominate AMOS averages; edge vs interior flips positive when lumen/air sits inside the mask.
3. **Solid organs + vessels** show the same qualitative pattern as WORD: interior brighter than edge by ~20–80 HU (Aorta largest gap).
4. **Gallbladder / Bladder** again have small edge−interior gaps (~3–6), similar to WORD’s weak-boundary pattern.

## Permanent extract (avoid re-unzip)

- Target: `/public/share/td20230405/AMOS 22/amos22/`
- Zip stays at `.../AMOS 22/amos22.zip`
- Future jobs should use `AMOS_ROOT=/public/share/td20230405/AMOS 22/amos22` (sbatch already updated)
- Extract size grows toward full train+val+test (~tens of GB)

## Artifacts

- `results/amos22_organ_hu_stats.json`
- School log: `runs/school/amos_hu_385351/`
- Script: `scripts/amos_organ_hu_stats.py`
