# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import cast

import tensorplay as tp
from tensorplay import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _get_scalar_dtype,
    _to_scalar,
    Optimizer,
)
from ._utils import scalar_value


__all__ = ["Adafactor", "adafactor"]



def _dtype_epsilon(dtype):
    if dtype == tp.float16:
        return 9.765625e-4
    if dtype == tp.bfloat16:
        return 7.8125e-3
    if dtype == tp.float64:
        return 2.220446049250313e-16
    return 1.1920928955078125e-7


class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr: float | Tensor = 1e-2,
        beta2_decay: float = -0.8,
        eps: tuple[float | None, float] = (None, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 >= beta2_decay:
            raise ValueError(f"beta2_decay should be <= 0 but is: {beta2_decay}")
        if eps[0] is not None and not 0.0 <= eps[0]:
            raise ValueError(f"epsilon1 should be >= 0 but is: {eps[0]}")
        if not 0.0 <= eps[1]:
            raise ValueError(f"epsilon2 should be >= 0 but is: {eps[1]}")
        if not 1.0 <= d:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")
        defaults = {
            "lr": lr,
            "beta2_decay": beta2_decay,
            "eps": eps,
            "d": d,
            "weight_decay": weight_decay,
            "foreach": foreach,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not tp.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = tp.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        row_vars,
        col_vars,
        variances,
        state_steps,
    ) -> bool:
        for p in group["params"]:
            if p.grad is None:
                continue
            if tp.is_complex(p):
                raise RuntimeError("Adafactor does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = tp.tensor(0.0, dtype=_get_scalar_dtype())

                if p.grad.dim() > 1:
                    row_shape = list(p.grad.shape)
                    row_shape[-1] = 1
                    # Row factor of variance, NOT the same shape as grads (will be reduced along last dim)
                    state["row_var"] = tp.zeros(
                        row_shape, dtype=p.grad.dtype, device=p.grad.device
                    )

                    col_shape = list(p.grad.shape)
                    col_shape[-2] = 1
                    # Col factor of variance, NOT the same shape as grads (will be reduced along penultimate dim)
                    state["col_var"] = tp.zeros(
                        col_shape, dtype=p.grad.dtype, device=p.grad.device
                    )
                else:
                    state["variance"] = tp.zeros_like(
                        p.grad
                    )

            row_vars.append(state.get("row_var", None))
            col_vars.append(state.get("col_var", None))
            variances.append(state.get("variance", None))
            state_steps.append(state["step"])
        return False  # has_complex

    @tp.no_grad()
    def step(self, closure=None):
        r"""Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            row_vars: list[Tensor | None] = []
            col_vars: list[Tensor | None] = []
            variances: list[Tensor | None] = []
            state_steps: list[Tensor] = []
            eps1, eps2 = group["eps"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
            )

            adafactor(
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
                d=group["d"],
                lr=group["lr"],
                beta2_decay=group["beta2_decay"],
                weight_decay=group["weight_decay"],
                eps1=eps1,
                eps2=eps2,
                foreach=group["foreach"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return loss


Adafactor.__doc__ = (
    r"""Implements Adafactor algorithm.

    .. math::
        \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \tau
                \text{(}\beta_2\text{ decay)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},    \\
            &\hspace{15mm}      \: \epsilon_1, \epsilon_2 \text{ (epsilons)}, \: d \text{(clipping threshold)}, \\
            &\hspace{15mm}      \: \lambda \text{(weight decay)},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : \: R_0 \leftarrow 0 \text{ (second moment row factor)},       \\
            &\hspace{23mm} \: C_0 \leftarrow 0 \text{ (second moment col factor)},               \\
            &\hspace{23mm} \: \widehat{V}_0 \leftarrow 0 \text{ (second moment for vectors)}     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}G_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}G_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\widehat{\beta}_{2_t} \leftarrow 1 - t^{\tau}                           \\
            &\hspace{5mm}\rho_t         \leftarrow min(\gamma, \frac{1}{\sqrt{t}})               \\
            &\hspace{5mm}\alpha_t       \leftarrow max(\epsilon_2,
                \text{RMS}(\theta_{t-1}))\rho_t                                                  \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}    \\
            &\hspace{5mm}\textbf{if} \: \text{dim}(G_t) > 1:                                     \\
            &\hspace{10mm}R_t           \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                (1-\widehat{\beta}_{2_t})(G_t \odot G_t) \cdot 1_m                               \\
            &\hspace{10mm}C_t           \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t)                         \\
            &\hspace{10mm}\widehat{V}_t \leftarrow
                \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{V}_t \leftarrow \widehat{\beta}_{2_t}\widehat{V}_{t-1}+
                (1-\widehat{\beta}_{2_t}) \cdot (G_t \odot G_t)                                  \\
            &\hspace{5mm}U_t            \leftarrow
                \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}                                \\
            &\hspace{5mm}\widehat{U}_t  \leftarrow \frac{U_t}{max(1, \frac{\text{RMS}(U_t)}{d})} \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \alpha_t \widehat{U}_t         \\

            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        \end{aligned}

    For further details regarding the algorithm we refer to `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`_.
    """
    + rf"""
    Args:
        params
        lr (float, Tensor, optional): unlike other optimizers, Adafactor does not require a
            learning rate, and Noam Shazeer and Mitchell Stern do not use lr at all.
            Deviating from the paper, this implementation uses lr for applying weight
            decay and as the maximum value for relative step size rho_t. Note that in
            the paper, a constant of 0.01 is used as the maximum value for relative
            step size, and so we set 0.01 as the default value. (default: 1e-2)
        beta2_decay (float, optional): the decay rate of beta2. beta2 standardly refers
            to the coefficient used for computing the running average of the gradient
            squared. (default: -0.8)
        eps (Tuple[float, float], optional): epsilon1 is the term added to the denominator
            of the update calculation to improve numerical stability. This use of epsilon1
            deviates from the algorithm written in the paper! See note below for more details.
            epsilon2 is the term used to avoid having too small a weight update when applying
            parameter scaling. (default: (None, 1e-3))
        d (float, optional): the clipping threshold, used to avoid larger-than-desired
            updates.
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        foreach (bool, optional): whether foreach implementation of optimizer is used. Note
            that the foreach implementation uses ~ sizeof(params) more peak memory than the
            for-loop version due to the intermediates being a tensorlist vs just one tensor.
            As Adafactor is commonly used when memory is prohibitive, Adafactor will default
            to the slower single tensor for-loop implementation unless this flag is explicitly
            True. This behavior is contrary to other optimizers, which will attempt defaulting
            to foreach on CUDA for faster runtime. (default: None)
        maximize"""
    + r"""
    .. Note::
        The implementation of Adafactor subtly differs from Noam Shazeer and Mitchell Stern
        and implementations in some other frameworks with its use of learning rate and
        :math:`\epsilon_1`.

        Regarding the learning rate hyperparameter: Noam Shazeer and Mitchell Stern do not
        use lr at all, as the stated algorithm uses :math:`\rho_t` and update clipping to
        affect the step size.

        This implementation allows `lr` to influence the maximum value for :math:`\rho_t`:

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(\gamma, \frac{1}{\sqrt{t}})
            \end{aligned}

        This differs from Noam Shazeer and Mitchell Stern, who use a constant of 0.01 as
        the maximum value of :math:`\rho_t`

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(0.01, \frac{1}{\sqrt{t}})
            \end{aligned}

        Noam Shazeer and Mitchell Stern do not enforce an opinion on how weight decay should
        be computed, and so we use the learning rate as a coefficient for decoupled weight
        decay, similar to what is suggested in `Decoupled Weight Decay Regularization`_.

        Regarding the use of :math:`\epsilon_1`: The implementation attempts to replicate the
        presumed intention of Noam Shazeer and Mitchell Stern to use :math:`\epsilon_1` as
        a stabilizing term when the squared gradient becomes small.

        This stabilization can be written as

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                    (1-\widehat{\beta}_{2_t})(G_t \odot G_t + 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                    (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow
                    \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}        \\
            \end{aligned}

        where the row and column factors of gradient squared :math:`R_t` and :math:`C_t`
        are left alone, and we apply :math:`\epsilon_1` at the final calculation of
        the variance estimate :math:`\widehat{V}_t` and for the update :math:`U_t`.

        This is in contrast to Noam Shazeer and Mitchell Stern and other frameworks which
        apply :math:`\epsilon_1` to both row and column factors of the squared gradient, but
        not in the calculations after:

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                            (1-\widehat{\beta}_{2_t})(G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                            (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow \frac{R_t \cdot C_t}{1^\top_n \cdot R_t}                          \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{\sqrt{\widehat{V}_t}}                                            \\
            \end{aligned}

        You may note that Noam Shazeer and Mitchell Stern describe using the sum of squared gradients,
        while this implementation uses the mean instead. This choice is mathematically equivalent and
        allows for greater numerical stability for large sums.

    .. _Adafactor\: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """
)


def _single_tensor_adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    # If grad is 1-dimensional (aka a vector), there is no factorization necessary
    # so row_var and col_var will be None while variance will be filled.
    # Contrarily, for a grad with multiple dimensions, we will factor along the last
    # 2 dimensions, and so row_var and col_var will be filled and variance will be None.
    row_vars: list[Tensor | None],
    col_vars: list[Tensor | None],
    variances: list[Tensor | None],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    d: float,
    lr: Tensor | float,
    beta2_decay: float,
    weight_decay: float,
    eps1: float | None,
    eps2: float,
    maximize: bool,
    has_complex: bool,
) -> None:
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Grad scaling should occur outside of optimizer.step()")

    if False:
        # this assert keeps the compiler from treating the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float because the compiled path receives floating-point inputs
        if not isinstance(lr, float):
            raise AssertionError(f"Expected lr to be a float, but got {type(lr)}")

    else:
        # Adafactor has no differentiable optimizer mode.  Materialize a
        # scalar Tensor lr exactly once so the Python min/norm calculations
        # and TensorPlay's alpha overloads receive a numeric scalar, matching
        lr = scalar_value(lr, "lr")

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        step_t = state_steps[i]
        row_var = row_vars[i]
        col_var = col_vars[i]
        variance = variances[i]
        if eps1 is None:
            eps1 = _dtype_epsilon(param.dtype)

        # update step
        step_t += 1
        step_float = step_t.item()

        one_minus_beta2_t = step_float**beta2_decay
        rho_t = min(lr, 1 / (step_float**0.5))
        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

        # Perform step weight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if grad.dim() > 1:
            if row_var is None or col_var is None:
                raise AssertionError(
                    "row_var and col_var should be defined when grad is multidimensional"
                )
            # same as (g * g).mean(dim=-1) w/o materializing an intermediate size g
            row_mean = (
                grad.norm( dim=-1, keepdim=True)
                .square()
                .div_(grad.size(-1))
            )
            row_var.lerp_(row_mean, one_minus_beta2_t)
            # same as (g * g).mean(dim=-2) w/o materializing an intermediate size g
            col_mean = (
                grad.norm( dim=-2, keepdim=True)
                .square()
                .div_(grad.size(-2))
            )
            col_var.lerp_(col_mean, one_minus_beta2_t)
            var_estimate = row_var @ col_var
            var_estimate.div_(row_var.mean(dim=-2, keepdim=True).clamp_(min=eps1))
        else:
            if variance is None:
                raise AssertionError("variance should be defined when grad is a vector")
            grad_squared = grad * grad
            variance.lerp_(grad_squared, one_minus_beta2_t)
            # avoid writing into variance during update
            var_estimate = variance.clone()

        # square the eps1 as we sqrt after to keep eps1's magnitude
        update = var_estimate.clamp_(min=eps1 * eps1).rsqrt_()
        update.mul_(grad)
        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
        param.add_(update, alpha=-alpha / denom)  # type: ignore[arg-type]


def _group_tensors_by_device_dtype_and_is_multidim(
    tensorlists: list[list[Tensor | None]],
) -> dict[
    tuple[tp.device | None, tp.dtype | None, bool],
    list[list[Tensor | None]],
]:
    """Groups tensors by device, dtype, AND multidimensionality -- whether the tensor
    has multiple dims or just one dim (is a vector). This allows the foreach impl of
    Adafactor to assume that every group of params will either be factored or not."""
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(tensorlists)
    ultra_grouped_tensors: dict[
        tuple[tp.device | None, tp.dtype | None, bool],
        list[list[Tensor | None]],
    ] = {}
    for (device, dtype), (tensorlists, _) in grouped_tensors.items():
        matrix_key = (device, dtype, True)
        vector_key = (device, dtype, False)

        # assumes grad is the second tensorlist
        for j, tensor in enumerate(tensorlists[1]):
            if tensor is None:
                raise AssertionError("grad should not be None")
            if tensor.dim() > 1:
                if matrix_key not in ultra_grouped_tensors:
                    ultra_grouped_tensors[matrix_key] = [[] for _ in tensorlists]
                for i in range(len(tensorlists)):
                    ultra_grouped_tensors[matrix_key][i].append(tensorlists[i][j])
            else:
                if vector_key not in ultra_grouped_tensors:
                    ultra_grouped_tensors[vector_key] = [[] for _ in tensorlists]
                for i in range(len(tensorlists)):
                    ultra_grouped_tensors[vector_key][i].append(tensorlists[i][j])
    return ultra_grouped_tensors


def _multi_tensor_adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    # If grad is 1-dimensional (aka a vector), there is no factorization necessary
    # so row_var and col_var will be None while variance will be filled.
    # Contrarily, for a grad with multiple dimensions, we will factor along the last
    # 2 dimensions, and so row_var and col_var will be filled and variance will be None.
    row_vars: list[Tensor | None],
    col_vars: list[Tensor | None],
    variances: list[Tensor | None],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    d: float,
    lr: Tensor | float,
    beta2_decay: float,
    weight_decay: float,
    eps1: float | None,
    eps2: float,
    maximize: bool,
    has_complex: bool,
) -> None:
    if len(params) == 0:
        return

    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Grad scaling should occur outside of optimizer.step()")

    lr = scalar_value(lr, "lr")

    grouped_tensors = _group_tensors_by_device_dtype_and_is_multidim(
        [params, grads, row_vars, col_vars, variances, state_steps]  # type: ignore[list-item]
    )
    for (_, dtype, is_multidim), (
        (
            device_params_,
            device_grads_,
            device_row_vars_,
            device_col_vars_,
            device_variances_,
            device_state_steps_,
        )
    ) in grouped_tensors.items():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_state_steps = cast(list[Tensor], device_state_steps_)
        if eps1 is None:
            eps_dtype = dtype if dtype is not None else device_params[0].dtype
            eps1 = _dtype_epsilon(eps_dtype)

        if maximize:
            device_grads = tp._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not tp.compiler.is_compiling() and device_state_steps[0].device.type == "cpu":
            tp._foreach_add_(
                device_state_steps, tp.tensor(1.0, device=tp.device("cpu")), alpha=1.0
            )
        else:
            tp._foreach_add_(device_state_steps, 1.0)

        one_minus_beta2_ts = []
        beta2_ts = []
        rho_ts = []
        for s in device_state_steps:
            one_minus_beta2_ts.append(s.item() ** beta2_decay)
            beta2_ts.append(1 - s.item() ** beta2_decay)
            rho_ts.append(min(lr, 1 / (s.item() ** 0.5)))

        alphas = [
            max(eps2, p.norm(2).item() / (p.numel() ** 0.5)) * r
            for p, r in zip(device_params, rho_ts, strict=True)
        ]

        # Perform step weight decay
        if weight_decay != 0:
            tp._foreach_mul_(device_params, 1 - lr * weight_decay)

        if is_multidim:
            device_row_vars = cast(list[Tensor], device_row_vars_)
            device_col_vars = cast(list[Tensor], device_col_vars_)
            if device_row_vars[0] is None or device_col_vars[0] is None:
                raise AssertionError(
                    "row_var and col_var should be defined when grad is multidimensional"
                )
            # same as (g * g).mean(dim=-1) w/o materializing an intermediate size g
            row_means = [
                grad.norm( dim=-1, keepdim=True)
                for grad in device_grads
            ]
            tp._foreach_mul_(row_means, row_means)
            tp._foreach_div_(row_means, [grad.size(-1) for grad in device_grads])
            tp._foreach_lerp_(device_row_vars, row_means, one_minus_beta2_ts)
            del row_means

            # same as (g * g).mean(dim=-2) w/o materializing an intermediate size g
            col_means = [
                grad.norm( dim=-2, keepdim=True)
                for grad in device_grads
            ]
            tp._foreach_mul_(col_means, col_means)
            tp._foreach_div_(col_means, [grad.size(-2) for grad in device_grads])
            tp._foreach_lerp_(device_col_vars, col_means, one_minus_beta2_ts)
            del col_means

            var_estimates = [
                row_var @ col_var
                for row_var, col_var in zip(
                    device_row_vars, device_col_vars, strict=True
                )
            ]
            row_var_means = [
                row_var.mean(dim=-2, keepdim=True) for row_var in device_row_vars
            ]
            tp._foreach_clamp_min_(row_var_means, eps1)
            tp._foreach_div_(var_estimates, row_var_means)
            del row_var_means
        else:
            device_variances = cast(list[Tensor], device_variances_)
            if device_variances[0] is None:
                raise AssertionError("variance should be defined when grad is a vector")

            grads_squared = tp._foreach_mul(device_grads, device_grads)
            tp._foreach_lerp_(device_variances, grads_squared, one_minus_beta2_ts)
            del grads_squared

            # avoid writing into variance during update
            var_estimates = [v.clone() for v in device_variances]

        # square the eps1 as we sqrt after to keep eps1's magnitude
        tp._foreach_clamp_min_(var_estimates, eps1 * eps1)
        tp._foreach_rsqrt_(var_estimates)
        tp._foreach_mul_(var_estimates, device_grads)
        updates = var_estimates

        alphas = [
            -a / (max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d)))
            for a, update in zip(alphas, updates, strict=True)
        ]
        tp._foreach_mul_(updates, alphas)  # type: ignore[arg-type]
        tp._foreach_add_(device_params, updates)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    row_vars: list[Tensor | None],
    col_vars: list[Tensor | None],
    variances: list[Tensor | None],
    state_steps: list[Tensor],
    foreach: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    *,
    d: float,
    lr: float | Tensor,
    beta2_decay: float,
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs Adafactor algorithm computation.

    """
    if not tp.compiler.is_compiling() and not all(
        isinstance(t, tp.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "`state_steps` argument must contain a list of singleton tensors"
        )

    # Native Adafactor keeps the scalar step counters on the host, performs
    # the RMS/clipping reductions in the backend, and fuses the elementwise
    # state/parameter update.  Split vector and factored tensors because their
    # state layouts differ; CUDA's factored kernel currently targets the
    # common 2-D parameter case, while CPU also handles higher-rank tensors.
    native_groups = {}
    native = (
        not tp.compiler.is_compiling()
        and grad_scale is None
        and found_inf is None
        and not has_complex
        and not isinstance(lr, tp.Tensor)
        and bool(params)
        # the fused kernel's combined write rounds at a different point.
        and params[0].dtype in (tp.float32, tp.float64)
    )
    if native:
        for index, (param, grad, row_var, col_var, variance, step) in enumerate(
            zip(params, grads, row_vars, col_vars, variances, state_steps)
        ):
            factored = param.dim() > 1
            if param.device.type == "cuda" and factored and param.dim() != 2:
                native = False
                break
            if not (
                param.is_contiguous()
                and param.is_floating_point()
                and grad.device == param.device
                and grad.is_contiguous()
                and grad.dtype == param.dtype
                and grad.shape == param.shape
                and step.device.type == "cpu"
                and step.is_contiguous()
                and step.numel() == 1
                and step.dtype in (tp.float32, tp.float64)
            ):
                native = False
                break
            if factored:
                if row_var is None or col_var is None:
                    native = False
                    break
                states = (row_var, col_var)
            else:
                if variance is None:
                    native = False
                    break
                states = (variance,)
            if not all(
                state.device == param.device
                and state.is_contiguous()
                and state.dtype == param.dtype
                for state in states
            ):
                native = False
                break
            key = (param.device, param.dtype, factored)
            group = native_groups.setdefault(
                key, {"params": [], "grads": [], "rows": [], "cols": [],
                      "variances": [], "steps": []}
            )
            group["params"].append(param)
            group["grads"].append(grad)
            group["rows"].append(row_var)
            group["cols"].append(col_var)
            group["variances"].append(variance)
            group["steps"].append(step)
    if native:
        native_lr = _to_scalar(lr)
        for (_, dtype, factored), group in native_groups.items():
            native_eps1 = eps1 if eps1 is not None else _dtype_epsilon(dtype)
            if factored:
                tp._fused_adafactor_factored_(
                    group["params"], group["grads"], group["rows"],
                    group["cols"], group["steps"], lr=native_lr,
                    beta2_decay=beta2_decay, eps1=native_eps1, eps2=eps2,
                    d=d, weight_decay=weight_decay, maximize=maximize,
                )
            else:
                tp._fused_adafactor_(
                    group["params"], group["grads"], group["variances"],
                    group["steps"], lr=native_lr, beta2_decay=beta2_decay,
                    eps1=native_eps1, eps2=eps2, d=d,
                    weight_decay=weight_decay, maximize=maximize,
                )
        if native_groups:
            return

    if foreach:
        func = _multi_tensor_adafactor
    else:
        func = _single_tensor_adafactor

    func(
        params,
        grads,
        row_vars,
        col_vars,
        variances,
        state_steps,
        d=d,
        lr=lr,
        beta2_decay=beta2_decay,
        weight_decay=weight_decay,
        eps1=eps1,
        eps2=eps2,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
