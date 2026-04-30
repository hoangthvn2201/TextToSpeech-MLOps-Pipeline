from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalFlowMatcher(nn.Module):
    """Optimal-transport conditional flow matching.

    Maps noise x_0 ~ N(0, I) to target mel x_1 via a learned vector field.
    At inference, integrates the learned ODE from t=0 to t=1.
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.sigma_min = sigma_min

    def sample_xt(
        self,
        x0: torch.Tensor,   # noise sample
        x1: torch.Tensor,   # target mel
        t: torch.Tensor,    # time ∈ [0, 1], shape (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t and the target vector field u_t at time t."""
        t_b = t[:, None, None]  # (B, 1, 1) for broadcasting over (B, n_mels, T)
        mu_t = t_b * x1 + (1 - t_b) * x0
        sigma_t = self.sigma_min + (1 - self.sigma_min) * t_b
        x_t = mu_t + sigma_t * torch.randn_like(x1)
        u_t = x1 - (1 - self.sigma_min) * x0  # target vector field (constant in OT-CFM)
        return x_t, u_t

    @torch.no_grad()
    def integrate(
        self,
        vector_field_fn: nn.Module,
        x0: torch.Tensor,
        condition: torch.Tensor,
        n_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        """Euler integration from t=0 to t=1."""
        dt = 1.0 / n_steps
        x = x0.clone()
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            vf = vector_field_fn(x, t, condition)
            x = x + dt * vf
        return x
