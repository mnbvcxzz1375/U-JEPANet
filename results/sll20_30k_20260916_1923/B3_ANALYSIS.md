# B3 result: JEPA pretrain → seg fine-tune

**Protocol:** A2-style encoder, JEPA-only pretrain 8k steps on 100 train images (20L+80U), then segmentation fine-tune 15k steps on 20 labeled with **JEPA fully off**.  
**Val:** whole-volume 20 `imagesVal` every 2k FT steps.  
**Code:** `scripts/b3_pretrain_finetune.py` · remote `runs/b3_pretrain_ft/`

## Result

| Setting | best mean fg Dice |
|---|---:|
| **A0 scratch U-Net** | **0.7617** |
| A2 joint (JEPA aux) | 0.7524 |
| A3 joint dual-path+JEPA | 0.7514 |
| **B3 pretrain→FT** | **0.7507** |
| A1 dual-path no JEPA | 0.7544 |

B3 val curve (FT steps): 2k 0.354 → 6k 0.739 → **8k 0.751** → 15k 0.747.

## Interpretation

1. **Gradient conflict is not the whole story.**  
   B3 removes joint JEPA gradients during segmentation. Still **below A0 by −0.011**.

2. **JEPA init is not a useful prior for 20L WORD segmentation** under this setup.  
   Pretraining 8k steps did not transfer into a better encoder for Dice.

3. Combined with earlier diagnostics:
   - joint training: mild negative cos, not catastrophic
   - per-organ: Left Kidney collapse on JEPA arms
   - dual-path alone (A1) also slightly worse than A0

4. Cleanest screening conclusion:

$$
\boxed{\text{On 20L WORD / 30k budget, scratch U-Net } \ge \text{ JEPA-pretrained or JEPA-joint variants}}
$$

## What B3 does *not* prove

- Not that medical SSL never helps (80 unlabeled is small; 8k pretrain is short).
- Not that a stronger backbone / longer pretrain / better masking cannot work.
- B3 FT budget (15k) is shorter than A0’s 30k — A0 still wins, so the gap is not explained by “B3 undertrained more than A0”.

## Recommended next (if continuing JEPA at all)

Priority order unchanged and now better justified:

1. **Stop A3 / dual-path line.** A1 and B3 both fail to beat A0.
2. If one more JEPA probe: **A2-L vs A2-LU** (does 80U help at all?) or **structure-aware masking + longer pretrain + cosine LR**, still vs A0.
3. Otherwise switch main effort to **PLL / sparq-seg**, keep JEPA as a negative/supplementary study with these diagnostics.

## Artifacts

- `b3_summary.json`, `b3_run.log` (this folder)
- Checkpoints: `/data/hyc/U-JEPANet/runs/b3_pretrain_ft/{pretrain_last,ft_best}.pt`
