import math

import tensorplay as tp

from ._utils import scalar_value, zeros_like
from .optimizer import Optimizer


def _sparse_adam(params, grads, exp_avgs, exp_avg_sqs, state_steps, *,
                 eps, beta1, beta2, lr, maximize):
    """Direct port of ``torch.optim._functional.sparse_adam``.

    The algorithm itself is Python in Torch; the sparse Tensor methods carry
    the indexed storage update.  Keeping this layer source-shaped means the
    optimizer becomes functional as soon as TensorPlay's sparse COO methods
    are available, instead of baking in a dense approximation.
    """

    for index, param in enumerate(params):
        grad = grads[index] if not maximize else -grads[index]
        grad = grad.coalesce()
        grad_indices = grad._indices()
        grad_values = grad._values()
        if grad_values.numel() == 0:
            continue
        size = grad.shape

        exp_avg = exp_avgs[index]
        exp_avg_sq = exp_avg_sqs[index]

        def make_sparse(values):
            constructor = getattr(grad, "new", None)
            if constructor is None:
                raise RuntimeError(
                    "TensorPlay sparse COO Tensor methods are unavailable"
                )
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = (
            grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        )
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(eps)
        bias_correction1 = 1.0 - beta1 ** state_steps[index]
        bias_correction2 = 1.0 - beta2 ** state_steps[index]
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        param.add_(make_sparse(-step_size * numer.div_(denom)))


class SparseAdam(Optimizer):
    """Masked Adam for dense parameters with sparse gradients."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 maximize=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "lr") <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if scalar_value(eps, "eps") <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= scalar_value(betas[0], "beta parameter at index 0") < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= scalar_value(betas[1], "beta parameter at index 1") < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps, maximize=maximize
        ))

        sparse_params = []
        complex_params = []
        for group_index, group in enumerate(self.param_groups):
            for param_index, param in enumerate(group["params"]):
                if param.is_sparse:
                    sparse_params.append([group_index, param_index])
                if param.is_complex():
                    complex_params.append([group_index, param_index])
        if sparse_params:
            raise ValueError(
                f"Sparse params at indices {sparse_params}: "
                "SparseAdam requires dense parameter tensors"
            )
        if complex_params:
            raise ValueError(
                f"Complex params at indices {complex_params}: "
                "SparseAdam does not support complex parameters"
            )

    @tp.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.grad.is_sparse:
                    raise RuntimeError(
                        "SparseAdam does not support dense gradients, "
                        "please consider Adam instead"
                    )
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = zeros_like(p)
                    state["exp_avg_sq"] = zeros_like(p)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state["step"] += 1
                state_steps.append(state["step"])

            if params_with_grad:
                try:
                    _sparse_adam(
                        params_with_grad, grads, exp_avgs, exp_avg_sqs,
                        state_steps, eps=scalar_value(group["eps"], "eps"),
                        beta1=scalar_value(beta1, "beta1"),
                        beta2=scalar_value(beta2, "beta2"),
                        lr=scalar_value(group["lr"], "lr"),
                        maximize=group.get("maximize", False),
                    )
                except AttributeError as exc:
                    raise RuntimeError(
                        "TensorPlay sparse COO Tensor methods are unavailable"
                    ) from exc
        return loss
