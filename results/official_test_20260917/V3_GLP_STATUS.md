# V3.1 causal control — G0/G1 matched architecture

**P0 (final before launch):** G1−G0 was confounded (G0 skipped CrossAttn blocks).

## Fix

- **Always** instantiate `GlobalStem` + full `gl_pred` stack for G0/G1/G2 (same RNG order).
- **G0:** `Z_G = GlobalStem.position_only(B)` = PE tokens only.
- **G1/G2:** `Z_G = GlobalStem(X_G)` = image features + same PE.
- Both go through **2× CrossAttn + FFN**. Sole delta: patient CT content.

## Contract tests (40901) — ALL PASS

```
PASS test_g0_g1_init_matching_same_seed   # bitwise equal shared state_dict
PASS test_content_ablation_g1_equals_g0   # alpha=0.1, PE-only G1 ≡ G0; content differs
PASS test_both_stacks_cross_attn
ALL_GLP_INTEGRATION_TESTS_PASS
```

## Launch layout (user-specified)

| Host | Arm |
|---|---|
| 40901 GPU0 | G0 s42 |
| 40902 GPU0 | G1 s42 |
| school A800 | G0 s43, G1 s43 |

Gate: **Δ_G = val(G1)−val(G0)** both seeds; then low-CNR / α_G / shuffled-global.
ImagesVal selection only. Per-job `--cache-dir` unique.
