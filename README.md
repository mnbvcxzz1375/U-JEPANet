# U-JEPANet

U-Net 深层双路 JEPA 半替代实验仓（A0–A3 对照）。

本仓库实现对话计划中的核心设计：

- 在 U-Net **中深层阶段**接入双路条件化深层模块（原图 tokens + 上一层特征 tokens）
- 采用 **I-JEPA 式**隐空间预测目标（遮挡上下文 → 预测目标隐表示，EMA 目标编码器）
- 第一轮做**半替代残差融合**，不删除原 CNN 深层分支
- 严格防双路信息泄漏：在线分支两路均来自同一遮挡输入

## 实验臂（第一轮）

| 编号 | 结构 | JEPA | 回答的问题 |
|---|---|---|---|
| **A0** | 原始 U-Net3D | 无 | 基线 |
| **A1** | U-Net + 双路深层模块 | 无 | 仅结构是否有帮助 |
| **A2** | 原始 U-Net + 对深层特征做 JEPA | 有 | 仅预测学习是否有效 |
| **A3** | 双路深层模块 + JEPA | 有 | 结构与预测目标结合 |

关键比较是 **A3 vs A1**（推理结构相同，差异主要在 JEPA 训练）。

## 目录

```
ujepa/           # 模型与训练组件
configs/         # A0–A3 / pilot 配置
train.py         # 训练入口
tests/           # 形状 / 防泄漏 / resume / val 测试
results/         # 可复现 run 的 summary + 分析（同步进 Git）
EXPERIMENT_PROTOCOL.md
```

**结果约定：** 每次筛选/正式运行结束后，把轻量 `summary.json`、`provenance.txt` 与 `ANALYSIS.md` 提交到 `results/<run_id>/`；大 checkpoint 与全量日志留在服务器，并在分析里写明路径。

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q
python train.py --config configs/a0_baseline.yaml --smoke
python train.py --config configs/a3_dualpath_jepa.yaml --smoke
```

正式训练需指定 WORD 风格数据目录（NIfTI 或已有 h5/npy cache）。默认配置面向
筛选预算（约 30k 更新），不是充分收敛预算。

## 设计边界（来自预注册计划）

1. 第一轮 decoder **不加** JEPA 损失，但参与分割训练。
2. 预测器仅训练期使用；推理使用在线编码器 + 分割 decoder。
3. 在线两路必须同源遮挡；目标分支用未遮挡图像 + EMA。
4. 不把 `JEPA loss 下降` 当作成功标准；以 **A3 相对 A1** 的分割改善为准。
5. 第一轮不加入 organ token / 多层 JEPA / decoder 预测 / 完全替换 CNN 深层。

## 关键实现约束（2026-09-16 修订）

| 约束 | 做法 |
|---|---|
| mask ratio | 在 **JEPA token grid** 上采样，再 nearest 上采样到 CT；保证 hidden≈0.4 |
| predictor PE | 始终加 3D sin-cos PE：`visible: z+pe`，`masked: q_mask+pe` |
| token 数 | `jepa_token_stride=16` → 128×128×96 得 **8×8×6=384** tokens |
| ImageEmbed | strided patch conv（kernel=stride），不再全分辨率 256 通道特征图 |
| JEPA forward | 只走 encoder，**不跑 decoder** |
| 半监督 | unlabeled 病例 fail-closed 不读 GT；`L_seg` 只算 labeled rows |
| EMA | `EMATarget` 为 `nn.Module`，进入 `state_dict` / checkpoint |
| Norm | 默认 InstanceNorm3d（3D batch=1~2 + clean/masked 混流） |
| class_num | **17**（WORD `labelsTr_All` = 0..16） |

详见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

## 与既有项目关系

- 研究设计归属 vault 项目 `word-jepa-plseg`
- 本仓是深层 JEPA 半替代的**独立实现仓**，不修改 PLS4MIS / JEPA_Pseudorgb 训练树
- PL-Seg 保留为部分标注协议下的独立对照，不在第一轮叠加
