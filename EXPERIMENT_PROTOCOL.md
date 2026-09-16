# U-JEPANet 实验协议（第一轮预注册）

来源：2026-09-16 MEDIC AD 器官分割可行性对话计划。  
本协议锁定 A0–A3 机制比较；正式论文结论需多 seed 复现。

## 1. 任务与数据边界

- 目标：WORD 风格腹部器官 CT 分割（或等价 16 类设定）。
- 训练图：允许使用的训练病例；验证/测试图像**不进入**无标签 JEPA 训练。
- 若存在部分标注（PLL）：不得把未标注器官体素当背景写入 CE。
- 预处理、patch、重采样、增强、推理设置在 A0–A3 间保持一致。

建议筛选划分（若无既有冻结划分）：

| 设置 | 全标注训练例 | 其余训练例 |
|---|---:|---:|
| 主实验 20% | 20 | 80 无标注 |
| 扩展 10% | 10 | 90 无标注 |
| 全监督参考 | 100 | — |

标注集合应嵌套；病例选择种子与训练种子分开保存。

## 2. 结构

### 2.1 基线 U-Net（A0）

4 或 5 级 3D U-Net；深监督可选但四臂一致。默认 4 级：
`feature_chns = [16, 32, 64, 128]` 或更大配置，所有臂相同。

### 2.2 双路深层模块（A1/A3）

选中深层阶段 `s`（默认 index=2，对应 1/4 分辨率；可配置）。

**JEPA 在独立低分辨率 token 网格上进行**（`jepa_token_stride`，默认 16）：
对 128×128×96 得到 8×8×6=384 tokens，落在协议要求的 256–1024。

```
T_F = Proj(F_{s-1}) → resample to JEPA grid
T_X = StridedPatchEmbed(X, stride=16)
Z_J = J_theta(T_F, T_X)      # 特征 tokens 作 Query，图像 tokens 作 K/V
F_s = F_s_CNN + alpha * A(Z_J)   # residual 始终 resize 到 CNN stage
```

- `alpha` 初始化 0.1（非零，保证分割梯度能进入新分支）
- 输出必须保留三维空间组织，不是全局向量
- 遮挡在 **token grid** 上采样，hidden 比例≈`mask_ratio`，再 nearest 上采样到 CT

默认模块超参：

| 参数 | 值 |
|---|---|
| token 通道 | 256 |
| 注意力头 | 4 |
| 交互 block | 4 |
| predictor block | 2 |
| jepa_token_stride | 16 |
| 位置编码 | 3D sin-cos（encoder 与 **predictor 均使用**） |
| class_num | 17（0=bg + 16 organs） |
| norm | InstanceNorm3d |

### 2.3 仅 JEPA、无双路结构（A2）

对原 U-Net 选中深层特征 `F_s` 做轻量投影到相同目标维度，用相近 predictor 做遮挡预测。  
推理结构与 A0 相同。

## 3. JEPA 目标（A2/A3）

对同一空间增强 crop `X`：

```
X_mask = M ⊙ X + (1-M) ⊙ c     # M=1 可见；c 固定填充
```

**在线分支（防泄漏）：**

```
Z_ctx = J_theta( Proj(E_{<s}(X_mask)), ImageEmbed(X_mask) )
```

两路都必须来自 `X_mask`，禁止：图像分支遮挡、特征分支用完整 `X`。
JEPA online/target 前向**只跑 encoder**，不跑分割 decoder。

**目标分支：**

```
Z_tar = B_{EMA}(X)             # 未遮挡 + EMA 参数
```

EMA 覆盖形成目标的整条编码路径（前段 + image embed + 深层模块），并写入 checkpoint。

**预测器：**

```
x_i = z_i + PE_i   (visible)
x_i = q_mask + PE_i (masked)   # 必须带 target position PE
```

**损失：**

```
L_J = mean_{i in Omega} SmoothL1( pred_i, sg[ LN(Z_tar_i) ] )
```

`Omega` 为被遮挡目标位置。第一轮：同 crop、同空间增强，仅在线增加遮挡。

遮挡：在 JEPA token grid 上的三维连续块，初始比例约 40%，**不用器官 GT 生成**。

## 4. 训练目标

```
L = L_seg(labeled only) + lambda_J(t) * L_J(all train images)
```

- `L_seg = Dice + CE`（可加多尺度深监督，四臂一致）
- Dataset 对 unlabeled 病例 **fail-closed 不加载 GT**，即使磁盘有 label
- 优化顺序：seg backward → JEPA backward → clip → step → EMA（避免同时持有两张大图）
- 节奏：前 3000 步仅分割；随后初始化 EMA=在线，再用 3000 步线性升 `lambda_J`；然后保持
- 优化器四臂统一：AdamW，lr 2e-4，wd 1e-4，grad clip 1.0
- checkpoint 含 model（含 EMA submodule）、optimizer、step、RNG
- 筛选预算：每组约 30,000 更新；正式实验再拉长到基线充分收敛

模块职责：

| 模块 | seg 更新 | JEPA 更新 | 推理保留 |
|---|---|---|---|
| U-Net 前段 | 是 | 是 | 是 |
| 图像分支 + 深层模块 | 是 | 是 | 是 |
| decoder | 是 | 否（第一轮） | 是 |
| predictor | 否 | 是 | 否 |
| EMA 目标编码器 | 不反传 | EMA | 否 |

## 5. 防假成功检查

1. **遮挡泄漏**：固定可见区，只改被遮挡区内容 → 在线预测应不变。
2. **表示健康**：跨病例特征变化、有效秩；打乱上下文预测误差应显著变差。
3. **分支是否被使用**：记录新模块梯度与 alpha；验证时临时禁用仅作诊断，不能代替 A1/A3 训练消融。

成功标准：**A3 相对 A1 的分割与表示质量改善**，不是 JEPA loss 变低。

## 6. 结果解释

| 观察 | 结论 |
|---|---|
| A1↑，A3 不高于 A1 | 支持结构改造，不支持 JEPA 增益 |
| A2 ≈ A3 | 简单辅助预测可能已够，双路未证明必要 |
| A3 稳定 > A1，探针也改善 | 有证据继续 JEPA 深层方向 |
| JEPA loss 低但分割不变、打乱上下文不变 | 排查塌缩 / 位置捷径 / 分支未用 |
| 完全替换差于半替代 | 保留混合结构 |

## 7. 暂不做

- 器官 token / prototype
- 多层编码器 JEPA
- decoder JEPA
- 与完整 PL-Seg 损失叠加
- 直接宣称 AlignJEPA（其为遥感视觉-语言对齐，冻结主干，与本设定不同）
