# A2-L vs A2-LU: does 80 unlabeled help JEPA?

**Protocol:** same SLL20 dual-loader 30k as A2; only the **JEPA image pool** differs.  
**Crops:** fixed cache (not dynamic) — L uses 240 labeled patches; LU uses 720 (20L+80U).  
**Val:** whole-volume 20 `imagesVal`. seed 42. Host 40901.

## Results

| Arm | JEPA pool | best Dice | @step | final@30k |
|---|---|---:|---:|---:|
| **A2-L** | 20L only (240 patches) | **0.7641** | 16000 | 0.7470 |
| **A2-LU** | 20L+80U (720 patches) | 0.7577 | 20000 | 0.7495 |
| A0 scratch | — | 0.7617 | 8000 | 0.7480 |
| A2 (old, LU-style) | 100 train | 0.7524 | 16000 | 0.7452 |

$$
\boxed{\Delta_U = \mathrm{Dice}_{LU}-\mathrm{Dice}_{L} = -0.0064}
$$

## Interpretation

1. **80 unlabeled + current JEPA hurts, not helps.**  
   Adding U images to the JEPA pool **lowers** best Dice by ~0.006.

2. **A2-L is the first JEPA arm to match / slightly exceed A0** (0.7641 vs 0.7617, +0.0024).  
   This is small and single-seed — **not a claim of JEPA superiority**, but it reverses the “JEPA always worse than A0” screening result.

3. Combined with B3 and organ diagnostics:
   - Joint JEPA on **labeled** data is not harmful and may be mildly useful as an aux regularizer.
   - Joint JEPA on **unlabeled** data (different appearance distribution / less task-aligned content) is **negative transfer** at this scale.
   - Dual-path (A1/A3) remains negative; do not revive A3 from this result.

4. Caveats:
   - Fixed crop cache still applies; C1/C2 on school test dynamic crop.
   - L vs LU differ in JEPA batch diversity (240 vs 720 patches), not only “U exists”.
   - One seed, one λ=0.3.

## What this rules out / opens

| Claim | Status |
|---|---|
| “Unlabeled CT via JEPA improves 20L seg” | **Not supported** (Δ_U &lt; 0) |
| “JEPA aux on labeled data is always harmful” | **Weakened** (A2-L ≥ A0) |
| “Need more U data to make JEPA work” | Possible but untested (only 80U) |
| “Task-aligned U loss (consistency/mix) better than latent JEPA” | Now a **priority baseline** |

## Next

1. Keep C1/C2 (dynamic crop) running on school.
2. If A2-L is real under dynamic crop too, consider **A2-L + CT-window** as the only JEPA arm worth continuing.
3. Compare **80U + pseudo-label / consistency** vs **80U + JEPA** before abandoning unlabeled data.

## Artifacts

- `a2L_summary.json`, `a2LU_summary.json`
- Remote: `/data/hyc/U-JEPANet/runs/a2l_lu/{L,LU}/`
