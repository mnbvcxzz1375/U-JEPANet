# WORD-SLL-20% 30k · A0–A3 Analysis

**run_id:** `sll20_30k_20260916_1923`  
**protocol:** 20 labeled + 80 unlabeled · dual loader · 30k supervised updates · whole-volume val (20 `imagesVal`) · seed 42  
**code:** `5ac7f87` · host 40901 · vllmenv  
**remote:** `/data/hyc/U-JEPANet/runs/sll20_30k_20260916_1923/`

## Headline

**A0（原始 U-Net）在本协议下最好；A3（双路+JEPA）最差。**  
机制筛查结论：当前实现的 dual-path residual + I-JEPA 辅助目标，在 20L SLL / 30k / 整卷 val 下**没有带来增益**。

## Results (best whole-volume mean fg Dice)

| Arm | 结构 | JEPA | best Dice | @step | final @30k | 用时 |
|---|---|---|---:|---:|---:|---:|
| **A0** | U-Net | 无 | **0.7617** | 8000 | 0.7480 | 5317s |
| A1 | + dual-path | 无 | 0.7544 | 10000 | 0.7378 | 6187s |
| A2 | U-Net + JEPA head | 有 | 0.7524 | 16000 | 0.7452 | 6851s |
| A3 | dual-path + JEPA | 有 | 0.7514 | 10000 | 0.7342 | 8673s |

关键对照：

- **A3 − A1 = −0.0030**（JEPA 在双路结构上略负）
- **A1 − A0 = −0.0073**（双路结构本身略负）
- **A2 − A0 = −0.0093**（仅 JEPA 也略负）
- A3 比 A0 慢约 **63%**，且更差

## Val curves (mean fg Dice)

| step | A0 | A1 | A2 | A3 |
|---:|---:|---:|---:|---:|
| 2000 | 0.354 | 0.348 | 0.348 | 0.353 |
| 4000 | 0.608 | 0.600 | 0.600 | 0.602 |
| 6000 | 0.748 | 0.744 | 0.738 | 0.717 |
| **8000** | **0.762** | 0.749 | 0.750 | 0.736 |
| 10000 | 0.757 | 0.754 | 0.748 | 0.751 |
| 16000 | 0.756 | 0.748 | 0.752 | 0.751 |
| 30000 | 0.748 | 0.738 | 0.745 | 0.734 |

四臂都在 ~6–10k 达峰后缓慢回落/平台；继续训到 30k **没有**拉开 A3 优势。

## Health (not a failure mode)

- 四臂均 30k 跑完，无 NaN/OOM
- A2/A3 的 JEPA loss 在 λ=0.3 后保持有限（A2≈0.039，A3≈0.088 @30k）
- A3 的 α 从 0.10 升到 ~0.20（分支在用）
- 整卷 val 每 2000 步正常执行

因此这不是“没训起来”，而是**训起来了但没有增益**。

## Interpretation (screening only)

按预注册解释表：

| 观察 | 应得结论 |
|---|---|
| A1 不高于 A0 | 当前 dual-path 半替代**未证明**有帮助 |
| A3 不高于 A1 | **不支持** JEPA 在此设定下的额外增益 |
| A2 ≈ A3 ≈ A0 附近且略低 | 简单 U-Net 在 20L 下已接近该预算上限 |

这与 3k pilot（A3 略高）相反——pilot 全 labeled、小 patch、短预算，不能外推。

## What this does *not* show

1. 不是“JEPA 永远无效”：仅单一 seed、单一 λ=0.3、单一 stage、自定义小 U-Net。
2. 不是“无标签无用”的直接证明：A2/A3 的 JEPA 用的是 100 train，但 20L 分割监督可能已饱和。
3. 未做 A3-L vs A3-LU（无标签增量贡献）消融。
4. 未与 nnU-Net ResEnc-M / PL-Seg 对照。

## Recommended next (if continuing this line)

1. **先接受筛查结论**：不要在未改设计前把 A3 推成主方法。
2. 若继续：  
   - 降低 dual-path 分支负担或改融合方式（A1 已偏负）  
   - λ_J 网格 {0.1, 0.3, 0.5} 或 A2-only 辅助目标  
   - A3-L vs A3-LU 定量无标签贡献  
   - 换更强 backbone 再测，排除“baseline 已够强”
3. 更干净的下一条主线可能仍是 **PLL / 结构化缺失**（sparq-seg），而不是继续堆 JEPA 半替代。

## Artifacts

- `summary.json`（本目录）
- 远端 checkpoints / 全量 logs：`.../a0` … `a3`
- 本地 code：https://github.com/mnbvcxzz1375/U-JEPANet
