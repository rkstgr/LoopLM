"""Adam-atan2: scale-invariant Adam variant used in HRM.

Replaces m / (sqrt(v) + eps) with atan2(m, sqrt(v)), making the update
direction independent of the loss scale. See Everett et al. 2024.
"""

import math

import torch
from torch.optim import Optimizer


class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.lerp_(grad, 1 - beta1)
                v.lerp_(grad.square(), 1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                m_hat = m / bc1
                v_hat = v / bc2

                # atan2 update (scale-invariant)
                update = torch.atan2(m_hat, v_hat.sqrt())

                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss
