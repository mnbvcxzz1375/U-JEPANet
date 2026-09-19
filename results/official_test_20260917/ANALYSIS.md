# Official test complete C-series + attribution

**date:** 2026-09-17 (revised after code audit)  
**protocol:** WORD official `imagesTs` 30 cases, whole-volume 128×128×96 stride 64×64×48  
**selection:** locked `best.pt` from imagesVal screening; no test-based reselection  
**authorization:** user, 2026-09-17  
**code base:** `533c1ec` + post-audit fixes (z-strat, intensity_mode, A0DA, paired C4)

## Corrected framing (post-audit)

These results are **valuable screening evidence that the data pipeline is the main bottleneck**, not a finished causal attribution of JEPA. Three implementation issues affect interpretation:

1. **A0 → A0D is a bundled change**: dynamic resampling + 70% foreground-biased labeled crops + fixed HU window (L=40,W=400), versus fixed cache + uniform crops + per-volume min-max. Cannot claim “dynamic crop alone ≈82%”.
2. **C4 weak/strong views were unpaired** (two independent datasets/loaders, no shared case/crop). Historical C4 numbers **cannot** reject correctly paired asymmetric JEPA views.
3. **Train/eval intensity mismatch**: dynamic arms train under windowing; submitted `533c1ec` eval used min-max. Checkpoint selection and reported test scores mix model quality with preprocessing robustness. Window-mode re-eval jobs are in flight.

Also: z-stratified non-fg origins could exceed legal `d-pd` (last band overshot), adding air/padding more often on unlabeled JEPA rows than labeled seg rows. **Fixed** in `dynamic_dataset.py` after the audit; historical runs used the buggy sampler.

## ALL Dice (legacy min-max eval, as first reported)

| Arm | crop | CT-med seg | JEPA | val best | test ALL | Δ vs A0 |
|---|---|---|---|---:|---:|---:|
| A0 | fixed | no | no | 0.7617 | 0.7674 | — |
| A2 | fixed | no | yes | 0.7524 | 0.7636 | −0.0039 |
| A3 | fixed dual-path | no | yes | 0.7514 | 0.7580 | −0.0094 |
| A2-LU | fixed | no | yes 20L+80U | 0.7577 | 0.7700 | +0.0026 |
| **A0D** | dynamic+fg+window | no | **no** | 0.8042 | **0.8110** | **+0.0436** |
| C1 | dynamic+fg+window | no | yes | 0.8050 | 0.8118 | +0.0444 |
| C2 | + CT-med (incl. mild_affine) | yes | yes | 0.8105 | 0.8169 | +0.0495 |
| C4 | + unpaired weak/strong | yes | yes | 0.8177 | 0.8168 | +0.0494 |
| **C3** | + JEPA strong-context | yes | yes | 0.8081 | **0.8203** | **+0.0529** |

### Case-paired bootstrap (per audit, patient unit, 100k)

| Compare | mean ΔDice | 95% CI | cases better/worse |
|---|---:|---|---|
| C1 − A0D | +0.00080 | [−0.00403, +0.00562] | 14/16 |
| C3 − A0D | +0.00929 | [+0.00470, +0.01391] | 21/9 |
| C3 − C1 | +0.00848 | [+0.00475, +0.01226] | 22/8 |

Interpretation: **C1 shows no reliable mean gain over A0D**; C3’s lift is not a single-case artifact. Still single-seed, screening-only.

## Pre-registered gates — status after audit

| Gate | Status |
|---|---|
| C1 > C0 | **PASS** on val/test — data-pipeline bundle vs fixed-crop is real |
| C2 > C1 | **PASS** — but C2 also includes `mild_affine` on labeled seg, not pure CT intensity |
| C3 > C2 | **PASS** on test; C3 JEPA path uses `jepa_strong_context`, **not** identical to `seg_augment_hu` |
| C4 > C3 | **INVALID** — unpaired views; does not test weak/strong JEPA hypothesis |
| JEPA on dyn pipeline (C1 vs A0D) | **Not proven** (CI covers 0) |

## Critical missing 2×2 cell

|  | no seg aug | with seg aug |
|---|---|---|
| **no JEPA** | A0D | **A0DA — in flight (job 383499)** |
| **with JEPA** | C1 | C2 / C3 |

We do **not** yet know $M(J=1,A=1)-M(J=0,A=1)$ — whether JEPA still helps after dynamic crop + CT-med. Do not claim “JEPA has no value under good pipeline” until A0DA (and ideally multi-seed) completes under **window** eval.

## Complete 2×2 + multi-seed + paired C4 (window test, 2026-09-18)

| Arm | JEPA | CT-med | seed | val best (window) | **test ALL (window)** |
|---|---|---|---|---:|---:|
| A0 fixed | no | no | 42 | 0.7617* | 0.7519† |
| A0D | no | no | 42 | 0.8042* | **0.8135** |
| A0D | no | no | 43 | 0.8043 | 0.8128 |
| C1 | yes | no | 42 | 0.8050* | 0.8144 |
| **A0DA** | **no** | **yes** | **42** | **0.8183** | **0.8209** |
| **A0DA** | **no** | **yes** | **43** | 0.8141 | **0.8215** |
| C2 | yes | yes | 42 | 0.8105* | 0.8197 |
| C2 | yes | yes | 43 | 0.8198 | 0.8210 |
| C3R | yes | yes + strong-ctx | 42 | 0.8170 | 0.8205 |
| C3R | yes | yes + strong-ctx | 43 | 0.8144 | **0.8237** |
| **C4P** | yes **paired** ws | yes | 42 | 0.8194 | **0.8224** |

\* some historical vals were logged under min-max; window-trained rows are the preferred protocol.  
† A0 fixed was trained on min-max cache; window test is **not** train-consistent for it — use 0.7674 min-max for the historical fixed baseline.

### 2×2 JEPA increment (window test)

|  | A=0 | A=1 (CT-med) |
|---|---|---|
| **J=0** | A0D ≈ **0.813** | **A0DA ≈ 0.821** |
| **J=1** | C1 ≈ **0.814** | C2 ≈ **0.820** |

```
JEPA @ A=0:  C1 − A0D ≈ +0.001   (negligible)
JEPA @ A=1:  C2 − A0DA ≈ −0.001 / −0.001  (s42 / s43; slightly negative)
C3R − A0DA:  −0.000 (s42) / +0.002 (s43)
C4P − C3R:   +0.002 (s42; paired weak/strong now slightly above C3R)
```

## V1.2 R1/R2 complete (window test, code f40d754 / V1.2-corrected)

Graph: full-context seg `F3=E3(F2*)`; aux L_P on original F2 (context live, target sg, masked-only); 384 tokens; deterministic eval.

| Arm | seed | val best | **test window** | host |
|---|---|---:|---:|---|
| A0DA | 42 | 0.8183 | 0.8209 | school |
| A0DA | 43 | 0.8141 | 0.8215 | school |
| **R1** λp=0 | **42** | 0.8185 | **0.8262** | 40901 |
| **R1** λp=0 | **43** | 0.8219 | **0.8273** | 40902 |
| **R2** λp=0.3 | **42** | 0.8225 | **0.8273** | 40902 |
| **R2** λp=0.3 | **43** | 0.8213 | **0.8268** | A800 385928 |

### Pre-locked gates

| Gate | s42 | s43 | mean | 读法 |
|---|---:|---:|---:|---|
| **R2 − R1** (pred loss) | +0.0011 | −0.0005 | **+0.0003** | \|Δ\|<0.002 且方向不一致 → **关闭 local same-crop predictive objective** |
| **R1 − A0DA** (contextual bottleneck) | **+0.0053** | **+0.0058** | **+0.0056** | 两 seed 同向；患者级 CI 分别约 [−0.001,+0.012] / [+0.002,+0.010] → **强正信号，非充分统计证明** |
| R2 − A0DA | +0.0064 | +0.0053 | +0.0059 | 与 R1 相近 → 增益主要来自 **low-res contextual mixing / bottleneck**，不是 pred loss |

**措辞（重要）：** V1.2 证明的是 **deep contextual bottleneck 有正信号**，不是 “predictive learning 已经有效”。R1 的 all-visible predictor 在 λp=0 时更接近 `Transformer(Z+PE)` 的 non-local mixing。Bladder 单器官约解释整体 mean 增量的 ~40%。

`eval_det_maxdiff=0.0` on all four arms.

### 结论（预锁门）

1. **Local same-crop predictive learning 正式结束**（R2≈R1）。
2. **Deep-path contextual bottleneck** 相对 A0DA 双 seed +0.005~0.006（架构信号；统计仍单/双 seed）。
3. **下一主线 V3 GLP：** R1 为新 baseline → G0 coord atlas → G1 whole-CT global → G2 + GL pred loss。主门 **G1−G0**（patient-specific global vs atlas）。

## V3.1 GLP G0/G1 — val + official test (complete s42+s43)

Checkpoints trained on school; **test re-run on 40901** (ckpts pulled locally; queue jobs cancelled).

| Arm | seed | val best | **test ALL** | low-CNR | high-CNR | α_G |
|---|---|---:|---:|---:|---:|---:|
| G0 | 42 | 0.8103 | **0.8222** | 0.731 | 0.933 | −0.047 |
| G1 | 42 | **0.8174** | 0.8213 | 0.734 | 0.927 | −0.001 |
| G0 | 43 | 0.8074 | **0.8178** | 0.726 | **0.930** | ~0 |
| G1 | 43 | **0.8131** | 0.8110 | 0.720 | 0.920 | ~0 |

### Δ_G = G1 − G0

| | s42 | s43 | mean |
|---|---:|---:|---:|
| **val** | **+0.0071** | **+0.0057** | **+0.0064** |
| **official test** | −0.0009 | **−0.0067** | **−0.0038** |

**Val↔test 排序翻转：** val 双 seed 支持 whole-CT content；**官方 test 双 seed 均为 G0≥G1**。在 causal-control 架构下，这**不支持**「patient-specific global anatomy 已在 test 上带来稳定增益」。

- α_G 在 G1 上始终 ~0 → 与 test 无增益一致（global residual 未被有效利用，或 val 过拟合了融合路径）。
- low/high-CNR：test 上 G0 在 high-CNR 更稳；G1 未在 low-CNR 上拉开。
- **结论：** 不自动开 G2；先查 val 选择是否过拟合 / shuffled-global / 是否把 GLP val 与旧 R1 val 混比。Causal 匹配本身工作正常（val 方向与 test 相反是科学结果，不是配对 bug）。

## Research story revision (after R1_GLP / 948ed7f review)

**Do not write:** A0DA → R1 **+0.0056 stable** → V3.

**Write instead:**

> Strong pipeline (dyn+fg+HU window+CT-med) → local contextual bottleneck showed **trajectory-sensitive** gains under one DataLoader RNG protocol; independent RNG (GLP trainer) did **not** reproduce +0.005 on val; naive global fusion (V3.1) failed to establish patient-specific benefit; diagnosis: **α-gate starvation + non-identifiable fusion (Z_L residual) + voxel-normalized FOV**.

- R1_GLP test ≈ 0.818/0.820 vs hist. R1_V1.2 0.826/0.827 — protocol-sensitive.
- G1 val ≈ R1_GLP; G0 val < R1 → G1−G0 val lift is mostly “G1 doesn’t hurt like G0”, not global benefit.
- G1−G0 test: s42 −0.0008 (15/30), s43 −0.0067 (9/30) — no patient-global evidence on test.

## V3.2 Aligned Global Innovation (implemented, not launched)

| | |
|---|---|
| Fusion | **I = Ẑ_G − Ẑ_A** only; RMS-norm Δ; α0≈0.04 (~3–5% RMS) |
| Geometry | **GridSample** hard align at p_i (patient voxel frame; FOV audit warns vs raw DICOM origin) |
| Arms | **B0 A0DA_GLP** · **H0 PE-only** · **H1 patient-global** (same stack/init) |
| Gate | **H1−H0** val ≳ +0.003 both seeds **and** real-case shuffle drop |
| RNG | DataLoader generator seed+10000 |
| Closed | old G2; no L_GL until global content is shown used |

Files: `ujepa/aligned_global_innovation.py`, `scripts/train_v32.py`, `tests/test_v32_innovation.py`. Auto-review: PASS + warnings (real-case shuffle still required at diag time).

### R1 under GLP trainer + gate diagnostics (2026-09-19)

Same DataLoader RNG / GLP val / code as G0/G1. α_g=0 (R1 has no coord stack).

| Arm | val s42 | val s43 | **test s42** | **test s43** |
|---|---:|---:|---:|---:|
| **R1_GLP** | **0.8184** | **0.8127** | **0.8176** | **0.8196** |
| G0 | 0.8103 | 0.8074 | **0.8222** | 0.8178 |
| G1 | 0.8174 | 0.8131 | 0.8213 | 0.8110 |
| R1_V1.2 (hist.) | 0.8185 | 0.8219 | 0.8262 | 0.8273 |

### Gate G0 − R1_GLP

| | s42 | s43 | mean |
|---|---:|---:|---:|
| **val** | **−0.0081** | **−0.0053** | **−0.0067** |
| **test** | +0.0046 | −0.0018 | **+0.0014** |

- **Val:** G0 明显低于同协议 R1 → PE/coordinate stack 在 val 上有害。
- **Test:** 混杂（s42 G0 更高，s43 略低）→ **不能**在 test 上断言 stack 必然伤 R1。
- R1_GLP test 仍低于历史 R1_V1.2 (~0.007–0.009) → trainer/eval 细节差仍在。

### Gate diagnostics (existing ckpts)

| ckpt | α | S_pre | S_fuse | 读法 |
|---|---:|---:|---:|---|
| G0 s42 | −0.047 | 0.012* | 0.37 | gate 可开 |
| G1 s42 | −0.001 | **0.0013** | **0.0018** | content 弱区分 + gate 关死 |

WORD n=150: spacing 齐（z=2.5），**FOV 变化大** → 64³ 为体素 atlas。

**结论：** 当前 G 系列在 test 上未证明 patient-global；val 上 G0 伤 R1。α 饥饿 + 物理未对齐仍是主要假说。下一步 V3.2（aligned + α0=0.05）或继续 R1 线，**不开旧 G2**。

### Predictive V1 first results (window, 2026-09-18 night)

> **V1.1 code audit:** P1/P2/P3 val/test used **stochastic masked inference**
> (`train_mode` defaulted True; `model.eval()` did not force full-context).
> Checkpoint selection was noisy. Treat all P* rows as **dev artifacts**.
> Corrected R1/R2: full-context seg path + masked-only aux loss + ~384 tokens.
> See `V1_1_CORRECTION.md`. **Not launched** pending user GitHub verification.

| Arm | 机 | val (window) | **test window ALL** | vs A0DA s42 (0.8209) |
|---|---|---:|---:|---:|
| A0DA s42 | school | 0.8183 | 0.8209 | — |
| A0DA s43 | school | 0.8141 | 0.8215 | +0.0006 |
| **P1 s42** λp=0, pred **in graph** | **40902 GPU0** | **0.8188** | **0.8232** | **+0.0023** |
| **P2 s42** +pred loss | **40901 GPU1** | **0.8201** | **0.8225** | +0.0016 |
| C3R s43 | school | 0.8144 | 0.8237 | +0.0028 |
| C4P s42 | 40901 | 0.8194 | 0.8224 | +0.0015 |

### V1 gates

| Gate | 观察 | 读法 |
|---|---|---|
| **P1 − A0DA** | test **+0.0023**，val ≈ +0.0005 | predictor 进图的**结构**至少不差，test 上略好 |
| **P2 − P1** | test **−0.0007**，val +0.0013 | **pred loss 未再抬 test**；目标仍无可靠额外贡献 |
| 与旧 A2 对比 | A2 0.8144（无增强 JEPA）远低 | 失败主因不是「predictor 缺席」单一因素；**数据管线仍是主杠杆** |

**当前结论（筛查、单 seed）：**

1. 把 predictor 放进 inference graph **可以追平/略超 A0DA**，但幅度小（~0.002）。
2. **预测损失仍不增加 test Dice** — 与 C1−A0D、C2−A0DA 方向一致。
3. P1 vs P2 无法支持「继续调 λ_J / 加强 JEPA loss」。
4. 学校多 seed（P1/P2 s43、P3 s42）仍在跑；若 P1 优势不稳，应转 **V3 global-local**（信息增量）而不是再堆预测目标。

**资源：** 40901 仅 P2 单卡；40902 仅 P1 单卡；其余在学校 A800。

### Post-audit conclusion

1. **Dynamic sampling + foreground bias + HU window + CT-med is the real pipeline fix.** A0DA (no JEPA) reaches **0.821–0.822** window test — at or above C2.
2. **Current JEPA adds no reliable mean Dice** once the pipeline is good (C2−A0DA ≤ 0 across seeds; C1−A0D ≈ +0.001).
3. C3R/C4P land within ~0.000–0.002 of A0DA — not enough to justify JEPA complexity at this budget/seed count.
4. **Correctly paired C4 (C4P)** no longer looks worse than C3R; the old C4 failure was an implementation artifact.
5. Dual-path remains closed. Historical min-max ranking of A2-LU>A0 is orthogonal to this pipeline result.

**Practical default going forward:** U-Net + dynamic crop + fg-biased + HU window + CT-med, **no JEPA**, until multi-seed with paired C4P or other SSL shows a clear edge.

## Complete window table (2026-09-18)

All dynamic-pipeline arms re-evaluated with `intensity_mode=window` (canonical L=40 W=400).

| Arm | min-max test | **window test** | Δ window−minmax |
|---|---:|---:|---:|
| A0D | 0.8110 | **0.8135** | +0.0025 |
| C1 | 0.8118 | **0.8144** | +0.0026 |
| C2 | 0.8169 | **0.8197** | +0.0028 |
| C3 | 0.8203 | **0.8217** | +0.0014 |
| C4 | 0.8168 | **0.8190** | +0.0022 |

**Window-mode contrasts (preferred going forward):**

```
C1 − A0D = 0.8144 − 0.8135 = +0.0009   (JEPA without seg-aug; still ~0)
C2 − A0D = 0.8197 − 0.8135 = +0.0062   (seg-aug bundle; still confounded with JEPA)
C3 − A0D = 0.8217 − 0.8135 = +0.0082
C3 − C2  = 0.8217 − 0.8197 = +0.0020
C4 − C3  = 0.8190 − 0.8217 = −0.0027   (historical C4 still unpaired — invalid for weak/strong gate)
```

Intensity mismatch was **small (~0.001–0.003)** for these checkpoints — not the main confound, but protocol is now aligned for new runs.

**Still missing for causal JEPA claim:** A0DA = U-Net + dynamic + CT-med + **no JEPA** under window val (job `383499`, early: step ~3.6k, val@2k 0.447). Once complete:

```
JEPA value under good pipeline = C2_or_C3 − A0DA   (same crop/aug, JEPA on/off)
```

## In-flight corrections (2026-09-17 night)

| What | Where |
|---|---|
| A0DA train (U-Net + dyn + CT-med, no JEPA, window val) | school `383499` gpu02 |
| Window-intensity official test C3/C4 | 40901 GPU0/1 |
| Window-intensity official test A0D/C1/C2 | school `383500–383502` gpu_4090 |
| z-strat fix + `intensity_mode` + paired `PairedJEPADataset` | code on both hosts |

## Per-organ (JSON, legacy min-max test)

A0D vs C1 vs C3 (audit-corrected from JSON):

| Organ | A0D | C1 | C3 | C1−A0D | C3−A0D |
|---|---:|---:|---:|---:|---:|
| Rectum | 0.6970 | 0.6573 | 0.7045 | **−0.0396** | +0.0076 |
| Gallbladder | 0.6837 | 0.7187 | 0.7107 | +0.0350 | +0.0270 |
| Adrenal | 0.5895 | 0.6192 | 0.6317 | +0.0297 | +0.0422 |
| Esophagus | 0.7052 | 0.6895 | 0.7061 | −0.0156 | +0.0009 |

C1 Rectum drop is concentrated: `word_0021/0052/0124` explain ~75% of the mean Rectum loss. Prefer case-level failure diagnosis over a global “predictability ≠ discriminability” story for now.

## Conclusions to use going forward

> In single-seed WORD-SLL20 screening, a new data pipeline (dynamic sampling + foreground-biased crops + HU windowing) accounts for most of the performance lift over the fixed-crop baseline. Because A0→A0D bundles multiple factors, that lift cannot yet be attributed to dynamic cropping alone. Without seg augmentation, JEPA shows a small, statistically uncertain mean gain over the new pipeline baseline (C1 vs A0D). With augmentation, C3 reaches higher test Dice on already-trained checkpoints, but the matching no-JEPA control (A0DA) has not been evaluated under a train-consistent intensity contract. C4’s historical numbers do not test weak/strong views because context/target crops were unpaired. Dual-path remains closed.

**Next (user-directed):** run corrected experiments on 40901 dual GPU + school platform — **not** more JEPA architecture complexity.

## Artifacts

- Legacy min-max JSONs remain in this folder (`test_*.json`) — do not overwrite.
- Window re-eval JSONs → `results/official_test_20260917/window/` once complete.
- A0DA run → school `runs/school/ujepa-a0da_<jobid>/`.
- Post-audit code: `ujepa/dynamic_dataset.py`, `ujepa/whole_volume_eval.py`, `ujepa/paired_views.py`, `scripts/train_c_ladder.py` (`A0DA`, intensity_mode), `scripts/eval_official_test.py`.
