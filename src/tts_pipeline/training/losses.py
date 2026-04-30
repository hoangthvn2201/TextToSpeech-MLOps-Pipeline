from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(pred_vf: torch.Tensor, target_vf: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_vf, target_vf)


def duration_loss(pred_log_dur: torch.Tensor, gt_dur: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_gt = gt_dur.float().clamp(min=1).log()
    loss = F.mse_loss(pred_log_dur, log_gt, reduction="none")
    return loss.masked_fill(mask, 0.0).sum() / (~mask).sum().clamp(min=1)


def mel_reconstruction_loss(pred_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_mel, target_mel)


def feature_matching_loss(
    real_features: list[torch.Tensor], fake_features: list[torch.Tensor]
) -> torch.Tensor:
    loss = torch.zeros(1, device=real_features[0].device)
    for r, f in zip(real_features, fake_features):
        loss = loss + F.l1_loss(f, r.detach())
    return loss / max(len(real_features), 1)
