# WORD organ HU statistics (100 train cases)

**Source:** `imagesTr` + `labelsTr_All` on server-40902  
**Cases:** 100 train volumes  
**Edge:** binary mask minus 1-iteration 6-connectivity erosion (organ boundary voxels)  
**Interior:** eroded organ core (non-edge)  
**Label order:** PLS4MIS WORD

## Table (voxel-pooled HU)

| Organ | mean HU | edge HU | interior HU | edge−interior | #voxels |
|---|---:|---:|---:|---:|---:|
| Liver | 151.5 | 124.0 | 154.8 | **−30.8** | 52.6M |
| Spleen | 162.1 | 122.0 | 169.1 | **−47.2** | 9.2M |
| Kidney(L) | 162.5 | 124.0 | 169.7 | **−45.7** | 6.7M |
| Kidney(R) | 162.9 | 126.9 | 169.7 | **−42.7** | 6.4M |
| Stomach | 101.0 | 106.2 | 100.0 | +6.2 | 17.6M |
| Gallbladder | 115.8 | 113.9 | 116.8 | −2.9 | 0.61M |
| Esophagus | 109.0 | 115.8 | 104.5 | **+11.3** | 0.61M |
| Pancreas | 145.9 | 129.9 | 153.7 | **−23.8** | 3.6M |
| Duodenum | 122.7 | 118.1 | 124.8 | −6.7 | 3.1M |
| Colon | 69.2 | 76.3 | 67.3 | **+9.0** | 31.1M |
| Intestine | 108.5 | 90.9 | 114.0 | **−23.2** | 41.1M |
| Adrenal | 114.7 | 104.0 | 129.5 | **−25.6** | 0.43M |
| **Rectum** | 103.4 | 101.4 | 104.1 | **−2.7** | 2.6M |
| Bladder | 105.7 | 103.7 | 106.1 | −2.3 | 10.0M |
| Femur(L) | 208.1 | 200.8 | 209.6 | −8.8 | 5.9M |
| Femur(R) | 208.4 | 200.3 | 210.1 | −9.8 | 6.0M |

## Reading

1. **Bone (Femur)** has by far the highest HU (~208) and a sharp interior vs edge drop (~9 HU) — high contrast, easy to predict.
2. **Solid soft organs** (Liver/Spleen/Kidney) sit ~150–163 HU interior; **edge is 30–47 HU darker** — large boundary gradient.
3. **Rectum / Bladder / Gallbladder** are mid-low HU (~100–116) with **very small edge−interior gap** (2–3 HU) — weak local contrast at the boundary.
4. **Esophagus / Colon / Stomach** show **positive** edge−interior (edge brighter) — likely fat/air/lumen adjacency, not a clean tissue boundary.

## Relation to JEPA screening

Rectum (largest negative Dice transfer under JEPA) has **low absolute HU and almost no edge contrast** — hard to discriminate locally, and not a “high-contrast predictable bone-like” structure. Femur(L) (largest positive transfer) has **high HU + strong interior-edge separation**. This is consistent with, but does not prove, the predictability vs discriminability hypothesis.

Artifacts: `results/word_organ_hu_stats.json`
