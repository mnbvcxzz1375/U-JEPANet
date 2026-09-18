# V1.1 correction — not for launch until user verifies on GitHub

## Bug fixed

`model.eval()` previously did **not** disable stochastic masking because
`PredictiveUNet.forward(..., train_mode=True)` was a Python default, and
`whole_volume_eval` calls `model(patch)` without that kwarg.

**Effect:** val/test on P1/P2/P3 used random 40% token masks per patch
(stochastic inference + noisy checkpoint selection).

## New semantics (V1.1)

| Path | Behavior |
|---|---|
| Seg forward train | full-context `P(Z, 1)` → merge(Z, Ẑ, R) |
| Seg forward eval | **same** full-context (deterministic under `model.eval()`) |
| Pred loss | separate aux branch, **masked tokens only** |
| Token grid | pooled to **~384** tokens (default 8×8×6) before attention |

## Arms (ready, not launched)

- **R1** `train_predictive_v11.py --arm R1` λp=0
- **R2** `--arm R2` λp=0.3 masked-only
- G0/G1/G2 stubbed in `ujepa/global_local_predictive.py` (not implemented)

## Historical P* numbers

`test_P*_window.json` from the stochastic-mask era are **not** valid for
R1/R2 comparison. Keep them as a labeled development artifact only.

## Next after user OK

1. Launch R1/R2 on 40901 single GPU + school
2. If R2 ≈ A0DA (+0.001–0.002) → close local same-crop prediction
3. Implement G0/G1/G2 Global→Local
