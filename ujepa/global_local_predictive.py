"""Global→Local Predictive U-Net (V3 / GLP-U-Net) — design stub, not launched.

Contract for implementation (after V1.1 verification):
- X_G = downsample(whole CT) -> E_G -> Z_G (~8^3 tokens)
- Local crop X_L -> U-Net -> F2 -> Z_L (8x8x6 tokens)
- a = normalized crop origin + patch size in whole-volume coords
- Z_hat_L = P(Z_G, e(a), optional Z_L_visible)
- R_L = Z_L - Z_hat_L   # patient-specific residual vs global expectation
- F2* = F2 + A([Z_hat_L, R_L])  # Z_L already in F2
- Arms: G0 coord-only, G1 global+coord no pred, G2 + masked/GL pred loss
- Eval: imagesVal multi-seed for selection; imagesTs development-only after many peeks
- Mechanism: Low-CNR organs should gain more than High-CNR if global context is the lever
"""

from __future__ import annotations

# Intentionally not implemented — pending user GitHub verification of V1.1.
