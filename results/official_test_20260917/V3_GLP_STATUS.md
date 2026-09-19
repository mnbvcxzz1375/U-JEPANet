# V3.1 GLP — integration P0s fixed; not launched

After user review of `2829c2a` (Needs revision). This commit fixes Dataset→collate→CUDA→val boundaries.

## P0 fixes

| Issue | Fix |
|---|---|
| batch coords used `origin[0]` | `token_coord_features_batched` → `(B,N,6)` per-sample |
| `affine_theta` `(B,1,3,4)` | Dataset stores `(3,4)`; collate → `(B,3,4)` |
| `global_image` `(B,64,64,64)` | Dataset stores `(1,64,64,64)`; collate → `(B,1,64,64,64)` |
| CPU/CUDA coord mismatch | batched helper uses `crop_origin.device` |
| val ignored coords/global | **`ujepa/glp_eval.py`**: real patch origins + whole-CT global; θ=I |

## P1 (done)

- Global tokens + whole-volume PE (`GlobalStem.pe_mlp`)
- DataLoader `generator=seed+10000` isolated from model-init RNG
- shuffle_global: batch=1 roll is no-op — external case ids required for diagnostic (documented in tests)

## α_G=0 gradient contract (intentional)

- G0/G1 step-0: fusion path zeros → only `alpha_g` gets grad; atlas/global enter after α leaves 0
- G2: `L_GL` trains global stem + atlas even when α=0

## Tests (40901)

```
ALL_GLP_V3_TESTS_PASS
ALL_GLP_INTEGRATION_TESTS_PASS
```

Includes: batch coords differ, CUDA batch=2 G0/G1/G2 fwd+bwd, GLP evaluator origins ≠ 0.

## Launch (after user OK)

G0/G1 s42+s43 parallel; imagesVal selection only; primary gate **G1−G0**.
