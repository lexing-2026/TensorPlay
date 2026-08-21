from .adam import Adam


class AdamW(Adam):
    """Adam with decoupled weight decay, matching ``torch.optim.AdamW``."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False,
                 fused=None):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group["decoupled_weight_decay"] = True
