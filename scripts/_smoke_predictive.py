import sys
sys.path.insert(0, "/data/hyc/U-JEPANet")
import torch
from ujepa.predictive_unet import PredictiveUNet, pred_loss_from_aux
m = PredictiveUNet()
x = torch.randn(1, 1, 64, 64, 48)
y = m(x, True)
print("train_out", tuple(y.shape))
print("lp", float(pred_loss_from_aux(m.last_aux)))
y2 = m(x, False)
print("infer_out", tuple(y2.shape))
m3 = PredictiveUNet(use_residual=False)
m3(x)
print("no_res_ok")
print("params_M", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
# grad flows through predictor when lambda_p=0
loss = y.sum()
loss.backward()
g = m.bottleneck.predictor.mask_token.grad
print("pred_grad_finite", g is not None and torch.isfinite(g).all().item() if g is not None else False)
print("SMOKE_OK")
