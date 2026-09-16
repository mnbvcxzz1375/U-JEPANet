# WORD-SLL-20% 30k · A0 interim

**run:** `sll20_30k_20260916_1923/a0`  
**host:** 40901 GPU1 · vllmenv · commit `5ac7f87`  
**protocol:** 20L+80U dual-loader, 30k supervised updates, whole-volume val on 20 `imagesVal`  
**status:** COMPLETE

## Result

- steps: 30000
- elapsed: 5316 s (~89 min)
- **best mean_fg_dice: 0.7617** (@ step 8000)
- final val @30000: 0.7480

## Val trajectory (mean fg Dice)

| step | dice |
|---:|---:|
| 2000 | 0.354 |
| 4000 | 0.608 |
| 6000 | 0.748 |
| **8000** | **0.762** |
| 10000 | 0.757 |
| 12000 | 0.752 |
| 14000 | 0.746 |
| 16000 | 0.756 |
| 18000 | 0.750 |
| 20000 | 0.749 |
| 22000 | 0.751 |
| 24000 | 0.749 |
| 26000 | 0.742 |
| 28000 | 0.747 |
| 30000 | 0.748 |

## Notes

- Healthy baseline: rises quickly then plateaus ~0.75; best early-mid (8k).
- Whole-volume eval on full 20 val cases (not centre crop).
- Checkpoints: `best.pt`, `last.pt`, `step_0{5..30}000.pt` on server.
- Not compared to A1/A2/A3 yet.

## Remote path

`/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/a0/`
