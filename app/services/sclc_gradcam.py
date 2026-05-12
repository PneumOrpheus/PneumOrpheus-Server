"""GradCAM++ for SCLC MIL FPN models.

Supports two backbone families:
  • ResNet-50 FPN  (MILResNet50Classifier)   → hooks model.resnet.layer4
  • Swin-based FPN (MILSwinV2TinyClassifier,  → hooks model.fpn
                    MILSwinV2BaseClassifier,
                    MILSwinTinyClassifier)

For the Swin family, layer4 does not exist.  The FPN fused output
(B*N, fpn_channels, h, w) is the deepest spatial feature map before the
global-average-pool in the instance head, so it is the natural CAM target.

Returns (cam_np (N,H,W), att_np (N,)) in all cases.
Falls back to zeros / uniform weights on any error so inference never fails.
"""
from __future__ import annotations

import numpy as np


def _pick_hook_target(model):
    """Return the module to hook, and whether its output is a tuple."""
    if hasattr(model, "resnet") and hasattr(model.resnet, "layer4"):
        return model.resnet.layer4, False
    if hasattr(model, "fpn"):
        return model.fpn, True
    return None, False


def compute_gradcam_pp(
    model: object,
    input_tensor: object,
    class_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cam_np, att_np).

    cam_np : (N, H, W) float32 in [0, 1] — per-instance heatmaps
    att_np : (N,)      float32           — attention weights, sum to 1
    """
    import torch
    import torch.nn.functional as F

    def _fallback(tensor):
        try:
            N = tensor.shape[1]
            H = tensor.shape[3]
            W = tensor.shape[4]
        except Exception:
            N, H, W = 16, 256, 256
        return (
            np.zeros((N, H, W), dtype=np.float32),
            np.ones(N, dtype=np.float32) / N,
        )

    try:
        target_module, output_is_tuple = _pick_hook_target(model)
        if target_module is None:
            return _fallback(input_tensor)

        B, N, C, H, W = input_tensor.shape

        activations: list = []
        gradients: list = []

        def _fwd(module, inp, out):
            # For FPN the output is (fused, aux) and we only need fused
            activations.append(out[0] if output_is_tuple else out)

        def _bwd(module, grad_in, grad_out):
            # grad_out[0] is the gradient w.r.t. the first (fused) output
            gradients.append(grad_out[0])

        fwd_h = target_module.register_forward_hook(_fwd)
        bwd_h = target_module.register_full_backward_hook(_bwd)

        try:
            model.zero_grad()
            x = input_tensor.detach()
            cls_logits = model(x, return_segmentation=False)
            if isinstance(cls_logits, (tuple, list)):
                cls_logits = cls_logits[0]
            cls_logits[0, class_idx].backward()
        finally:
            fwd_h.remove()
            bwd_h.remove()
            model.zero_grad()

        if not activations or not gradients:
            return _fallback(input_tensor)

        A = activations[0].detach()
        G = gradients[0].detach()

        if G is None:
            return _fallback(input_tensor)

        # GradCAM++ alpha weights
        G2 = G ** 2
        G3 = G ** 3
        sum_A = A.sum(dim=(-2, -1), keepdim=True) # (B*N, C, 1, 1)
        alpha = G2 / (2.0 * G2 + sum_A * G3 + 1e-7)
        alpha = alpha * (G > 0).to(alpha.dtype)

        # Per-instance channel weights
        w = (alpha * F.relu(G)).sum(dim=(-2, -1)) # (B*N, C)

        # Spatial CAM
        cam = F.relu((w[:, :, None, None] * A).sum(dim=1)) # (B*N, h, w)
        cam = cam[:N] # (N, h, w) for B=1

        # Normalise each instance independently
        flat = cam.reshape(N, -1)
        cmin = flat.min(dim=1).values[:, None, None]
        cmax = flat.max(dim=1).values[:, None, None]
        cam = (cam - cmin) / (cmax - cmin + 1e-7)

        # Upsample to bag-slice resolution
        cam = F.interpolate(
            cam.unsqueeze(1).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1) # (N, H, W)

        # Attention weights
        att = model._last_attention
        if att is not None:
            att = att[0].detach()
            att = (att / (att.sum() + 1e-7)).cpu()
        else:
            att = torch.ones(N) / N

        return cam.detach().cpu().numpy().astype(np.float32), att.numpy().astype(np.float32)

    except Exception:
        return _fallback(input_tensor)