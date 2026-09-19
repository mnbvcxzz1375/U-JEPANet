# V3 GLP-U-Net — implement, not launched

User authorized V3 after V1.2 gates (4f60193 + review). **Training not started.**

## Locked story

- V1.2 shows **deep contextual bottleneck** positive signal (R1−A0DA ≈ +0.0056 both seeds);
  **not** “predictive learning works” (R2−R1 ≈ 0).
- R1 is the **new architecture baseline**.
- Primary next gate: **G1 − G0** (patient-specific whole-CT vs coordinate atlas).

## Arms

| Arm | Local R1 bottleneck | Coord atlas | Whole CT Z_G | L_GL |
|---|---|---|---|---|
| R1 | ✓ | × | × | × |
| G0 | ✓ | ✓ | × | × |
| G1 | ✓ | ✓ | ✓ | × |
| G2 | ✓ | ✓ | ✓ | ✓ Huber(Z_hat, sg(Z_L_orig)) |

- `α_G` init **0** → G* starts ≡ R1
- G2 target from **original F2** tokens, not post-fusion
- Selection **imagesVal only** (test is development-only after many peeks)
- Mechanism metric (pre-reg): ΔDice_lowCNR vs highCNR
- Diagnostic: shuffled-global on G1/G2

## P0 contract (must pass before train)

`tests/test_glp_v3_alignment.py` — **ALL_GLP_V3_TESTS_PASS** on 40901:

- identity crop origin math
- **affine θ round-trip** vs align_corners=False (edge-based coords, no double +0.5)
- α_G=0 freezes fusion
- G2 L_GL grads into global stem
- dataset returns crop_origin / full_shape / affine_theta / global_image

## Files

- `ujepa/ct_augment.py` — mild_affine `return_theta`, token→global voxel helpers
- `ujepa/dynamic_dataset.py` — crop meta + global view
- `ujepa/global_local_predictive.py` — GLPUNet
- `scripts/train_glp_v3.py` — G0/G1/G2 (+R1)
- `tests/test_glp_v3_alignment.py`

## After user verify → launch

40901 one GPU + 40902 + school A800/4090d; seeds 42/43; no SIGReg/sparse/particle yet.
