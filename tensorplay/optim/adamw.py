import tensorplay as tp

from .adam import Adam, adam


__all__ = ["AdamW", "adamw"]


class AdamW(Adam):

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
def adamw(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        foreach=None, capturable=False, differentiable=False, fused=None,
        grad_scale=None, found_inf=None, has_complex=False, *, amsgrad,
        beta1, beta2, lr, weight_decay, eps, maximize):
    """Functional API that performs the AdamW algorithm computation."""
    adam(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        foreach=foreach,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=True,
    )
