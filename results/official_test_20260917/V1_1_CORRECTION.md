# V1.2 correction (after user review of 4b7971f)

Status: **Needs revision → fixed; not launched.** Awaiting GitHub verify.

## P0-1 (fixed): deep path

`backbone.encode()` had already computed `F3=E3(F2_old)` before bottleneck.

**Now:** stage-by-stage encode; `F3 = down3(F2*)` so predictive output is on
the deep encoder path, not only a decoder skip.

Test: `test_deep_path_f3_sees_f2_star` — zeroing merge must change F3.

## P0-2 (fixed): aux gradients + target

Old: `masked_pred_loss(feats[s].detach())` on post-bottleneck `F2*`
→ `∇θ_E L_P = 0`, self-referential target.

**Now:**
- Aux branch runs on **original F2**
- Context tokens **not** detached → encoder + proj get grad
- Target = `sg(Z)`
- `L_P` only on masked tokens

Test: `test_aux_grad_reaches_encoder_proj_predictor`
(down2 / proj / predictor all have grad after `L_P.backward()`).

## Other

- `target_grid_384((32,32,24)) == (8,8,6)` explicit + unit-tested
- Deterministic eval test passes
- Masked fraction ~0.4 test passes
- R1/R2 trainer `scripts/train_predictive_v11.py` unchanged API
  (`model.pred_loss`); G0–G2 still stubbed
- Historical P* JSONs remain **dev artifacts** (stochastic-mask era)

## Locked graph

```
X → E0 → E1 → F2
      ├─ B_full → F2* → E3 → F3* → Decoder(F0,F1,F2*,F3*)
      └─ B_mask(F2) → L_P   (context grad, target sg)
```

## After user OK

Launch R1 (λp=0) / R2 (λp=0.3) only — 40901 single GPU + school.
No V3 implementation until R1/R2 complete under this graph.
