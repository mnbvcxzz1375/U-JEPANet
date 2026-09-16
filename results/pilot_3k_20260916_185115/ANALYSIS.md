# A0–A3 3k Pilot Analysis

**run_id:** `pilot_3k_20260916_185115`  
**日期:** 2026-09-16  
**主机:** `ubuntu@10.126.25.5:40901` GPU1  
**环境:** `/home/ubuntu/anaconda3/envs/vllmenv` · torch 2.10.0+cu128  
**代码:** 本地 `ab57558`（训练中补丁：A1 `target is None` 修复，后提交为 `17872fa`）  
**远端目录:** `/data/hyc/U-JEPANet/runs/pilot_3k_20260916_185115/`

## 协议（筛查级）

- 数据：WORD npy 缓存 `data/word_pilot_cache`
  - train：20 × `imagesTr` 病例 × 4 random crop（96×96×64）= 80 samples
  - val：8 × `imagesVal` 中心 crop
  - 标签：`labelsTr_All` / `labelsVal`；**未读 imagesTs**
- seed 42，batch 2，3000 steps，val every 500
- 四臂同一划分/crop 缓存/seed

## 结果

| Arm | best mean_fg_dice | 末步 l_seg | 末步 l_jepa | α 末值 | elapsed |
|---|---:|---:|---:|---:|---:|
| A0 | **0.1227** | 0.273 | — | — | 170s |
| A1 | 0.1163 | 0.587 | — | 0.1445 | 217s |
| A2 | 0.1311 | 0.014 | 0.117 | — | 254s |
| A3 | **0.1315** | 0.370 | 0.069 | 0.1480 | 363s |

A3 − A1 ≈ **+0.015**（方向信号，不是方法结论）。

## 健康检查

- 四臂均跑满 3000 steps，loss 有限
- 各臂 val Dice 随 checkpoint 上升
- A1/A3 的 `alpha` 离开 0.1 → 双路分支有梯度
- A2/A3 JEPA loss 在 ramp 后保持 > 0
- GPU1 峰值约 2.8 GB；进程 RSS 约 1.9 GB；无 OOM

## 解释边界

1. **不能**把 0.13 Dice 当作 WORD 正式指标：仅 20 例 / 8 val crop / 3k steps。
2. **不能**用 JEPA loss 下降证明分割增益；本分析只作健康门。
3. A3 略高于 A1/A0，但样本与预算都远不足以支持“JEPA 有效”。
4. 本仓定位为 **mechanism screening platform**（简单 U-Net），不是最终 SOTA backbone。

## 已知问题（过程中修复）

- 在线读完整 NIfTI 会卡住首 step → 改为预缓存 npy
- A1 在 `model.target is None` 时调用 `target.train(False)` 崩溃 → 已修

## 远端/本地证据

- 远端：`summary.json`、`{a0,a1,a2,a3}.log`、`best.pt`/`last.pt`、`provenance.txt`
- 本目录：`summary.json`、`provenance.txt`（自远端拷贝）

## 下一步（30k 前）

1. 确认正式数据协议（全量 imagesTr / SLL 12+88 / PLL）
2. val 与项目 locked evaluator 对齐（当前 8 例中心 crop 仅 pilot）
3. 等预算 + 同 seed 四臂齐开 30k
