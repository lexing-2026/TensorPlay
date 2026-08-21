import math
import tensorplay as tp
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import (
    _batchable,
    adam as _foreach_adam,
    adam_foreach as _adam_foreach,
)
from ._fused import adam as _fused_adam
from ._utils import (
    capturable_supported,
    elementwise_max,
    ensure_state_step,
    foreach_enabled,
    scalar_value,
    scalar_pow,
    state_step,
    zeros_like,
)


def _zeros_like_state(param):
    return zeros_like(param)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach=None,
                 maximize=False, capturable=False, differentiable=False,
                 fused=None, decoupled_weight_decay=False):
        if isinstance(lr, tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= scalar_value(lr, "lr"):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if scalar_value(eps, "eps") < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if isinstance(betas[0], tp.Tensor) and betas[0].numel() != 1:
            raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], tp.Tensor) and betas[1].numel() != 1:
            raise ValueError("Tensor betas[1] must be 1-element")
        if not ((isinstance(betas[0], tp.Tensor) and isinstance(betas[1], tp.Tensor)) or
                (not isinstance(betas[0], tp.Tensor) and
                 not isinstance(betas[1], tp.Tensor))):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
            )
        if isinstance(betas[1], tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
            )
        if not 0.0 <= scalar_value(betas[0], "beta parameter at index 0") < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= scalar_value(betas[1], "beta parameter at index 1") < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= scalar_value(weight_decay, "weight_decay"):
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize,
                        capturable=capturable, differentiable=differentiable,
                        fused=fused,
                        decoupled_weight_decay=decoupled_weight_decay)
        super(Adam, self).__init__(params, defaults)
        if fused and differentiable:
            raise RuntimeError("`fused` does not support `differentiable`")
        if fused and foreach:
            raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if p_state and not isinstance(p_state.get("step"), tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]), dtype=tp.float32,
                        device=p.device if (group["capturable"] or group["fused"])
                        else tp.device("cpu"),
                    )

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            active = [p for p in group['params'] if p.grad is not None]
            maximize = group.get('maximize', False)
            decoupled = group.get('decoupled_weight_decay', False)
            foreach = group.get('foreach', None)
            fused = group.get('fused', False)

            if active and fused:
                if any(p.grad.is_sparse or p.is_complex() or
                       not p.is_floating_point() for p in active):
                    raise RuntimeError(
                        "`fused` does not support sparse gradients or complex/non-floating parameters"
                    )
                if (isinstance(group['weight_decay'], tp.Tensor) and
                        group['weight_decay'].requires_grad):
                    raise RuntimeError(
                        "Adam with fused=True does not support differentiable weight_decay"
                    )
                params = active
                grads = [p.grad if p.grad.is_contiguous() else p.grad.clone()
                         for p in active]
                exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], []
                states_valid = True
                for p in active:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = tp.tensor(
                            0.0, dtype=tp.float32, device=p.device
                        )
                        state['exp_avg'] = _zeros_like_state(p)
                        state['exp_avg_sq'] = _zeros_like_state(p)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = _zeros_like_state(p)
                    step = ensure_state_step(state, param=p, capturable=True)
                    if step.device != p.device:
                        states_valid = False
                        break
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    state_steps.append(step)
                if states_valid:
                    fused_lr = lr
                    if isinstance(fused_lr, tp.Tensor) and fused_lr.device != params[0].device:
                        fused_lr = fused_lr.to(device=params[0].device)
                    if _fused_adam(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, state_steps,
                            lr=fused_lr,
                            beta1=scalar_value(group['betas'][0], 'beta1'),
                            beta2=scalar_value(group['betas'][1], 'beta2'),
                            eps=scalar_value(group['eps'], 'eps'),
                            weight_decay=scalar_value(group['weight_decay'], 'weight_decay'),
                            amsgrad=group['amsgrad'], maximize=maximize,
                            grad_scale=getattr(self, "grad_scale", None),
                            found_inf=getattr(self, "found_inf", None),
                            decoupled_weight_decay=decoupled):
                        continue

            beta1, beta2 = group['betas']
            if (active and not fused and not maximize and not decoupled and foreach_enabled(group, active)
                    and not group.get('differentiable', False)
                    and not group.get('capturable', False)
                    and not isinstance(group['lr'], tp.Tensor)
                    and not isinstance(beta1, tp.Tensor)
                    and not isinstance(beta2, tp.Tensor)
                    and getattr(tp._C, "_foreach_adam", None) is not None):
                params = active
                grads = [p.grad for p in active]
                amsgrad = group['amsgrad']
                lr = group['lr']

                if (_batchable(params, grads) and
                        not (isinstance(lr, tp.Tensor) and lr.requires_grad) and
                        not (isinstance(beta1, tp.Tensor) and beta1.requires_grad) and
                        not (isinstance(beta2, tp.Tensor) and beta2.requires_grad) and
                        not (isinstance(group['weight_decay'], tp.Tensor) and
                             group['weight_decay'].requires_grad)):
                    exp_avgs = []
                    exp_avg_sqs = []
                    max_exp_avg_sqs = []
                    steps = []
                    states_valid = True
                    for p in active:
                        state = self.state[p]
                        if len(state) == 0:
                            state['step'] = tp.tensor(
                                0.0, dtype=tp.float32, device=tp.device('cpu')
                            )
                            state['exp_avg'] = _zeros_like_state(p)
                            state['exp_avg_sq'] = _zeros_like_state(p)
                            if amsgrad:
                                state['max_exp_avg_sq'] = _zeros_like_state(p)
                        required = ('step', 'exp_avg', 'exp_avg_sq')
                        if amsgrad:
                            required += ('max_exp_avg_sq',)
                        if any(key not in state for key in required):
                            states_valid = False
                            break
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        max_exp_avg_sqs.append(
                            state['max_exp_avg_sq'] if amsgrad else tp.Tensor()
                        )
                        step = ensure_state_step(
                            state, param=p, capturable=False
                        )
                        steps.append(int(step.item()) + 1)

                    state_tensors = [exp_avgs, exp_avg_sqs]
                    if amsgrad:
                        state_tensors.append(max_exp_avg_sqs)
                    if (states_valid and _batchable(params, grads, state_tensors)):
                        if _foreach_adam(
                                params, grads, exp_avgs, exp_avg_sqs,
                                max_exp_avg_sqs, steps, lr=group['lr'],
                                beta1=beta1, beta2=beta2, eps=group['eps'],
                                weight_decay=group['weight_decay'],
                                amsgrad=amsgrad):
                            for p in active:
                                ensure_state_step(
                                    self.state[p], param=p, capturable=False
                                ).add_(1.0)
                            continue

            if (active and not fused and foreach_enabled(group, active)
                    and not group.get('differentiable', False)
                    and not (isinstance(group['weight_decay'], tp.Tensor) and
                             group['weight_decay'].requires_grad)):
                beta1, beta2 = group['betas']
                if (not group.get('capturable', False) and
                        (isinstance(group['lr'], tp.Tensor) or
                         isinstance(beta1, tp.Tensor) or
                         isinstance(beta2, tp.Tensor))):
                    raise ValueError(
                        "Tensor lr and Tensor betas are not supported for "
                        "capturable=False and foreach=True"
                    )
                if group.get('capturable', False):
                    for p in active:
                        capturable_supported(p)
                params = active
                grads = [p.grad if p.grad.is_contiguous() else p.grad.clone()
                         for p in active]
                amsgrad = group['amsgrad']
                exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = [], [], [], []
                for p in active:
                    state = self.state[p]
                    if not state:
                        state['step'] = tp.tensor(
                            0.0, dtype=tp.float32,
                            device=p.device if group.get('capturable', False)
                            else tp.device('cpu'),
                        )
                        state['exp_avg'] = _zeros_like_state(p)
                        state['exp_avg_sq'] = _zeros_like_state(p)
                        if amsgrad:
                            state['max_exp_avg_sq'] = _zeros_like_state(p)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    steps.append(ensure_state_step(
                        state, param=p,
                        capturable=group.get('capturable', False)
                    ))
                fused_lr = group['lr']
                if (isinstance(fused_lr, tp.Tensor) and active and
                        fused_lr.device != active[0].device):
                    fused_lr = fused_lr.to(device=active[0].device)
                if _adam_foreach(
                        params, grads, exp_avgs, exp_avg_sqs,
                        max_exp_avg_sqs, steps, lr=fused_lr,
                        beta1=group['betas'][0], beta2=group['betas'][1],
                        eps=scalar_value(group['eps'], 'eps'),
                        weight_decay=group['weight_decay'], amsgrad=amsgrad,
                        maximize=group.get('maximize', False),
                        capturable=group.get('capturable', False),
                        decoupled_weight_decay=decoupled):
                    continue

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = -p.grad if group.get('maximize', False) else p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = tp.tensor(
                        0.0,
                        dtype=tp.float32,
                        device=p.device if (group.get('capturable', False) or group.get('fused', False))
                        else tp.device('cpu'),
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = _zeros_like_state(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = _zeros_like_state(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = _zeros_like_state(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                capturable = group.get('capturable', False)
                differentiable = group.get('differentiable', False)
                if capturable:
                    capturable_supported(p)
                step_t = state_step(
                    state, param=p, capturable=(capturable or group.get('fused', False))
                )

                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                exp_avg = tp.view_as_real(exp_avg) if is_complex else exp_avg
                exp_avg_sq = tp.view_as_real(exp_avg_sq) if is_complex else exp_avg_sq
                if amsgrad:
                    max_exp_avg_sq = tp.view_as_real(max_exp_avg_sq) if is_complex else max_exp_avg_sq

                lr = group['lr']
                weight_decay = group['weight_decay']
                lr_value = scalar_value(lr, 'lr')
                if group.get('decoupled_weight_decay', False):
                    param.mul_(1.0 - lr_value * scalar_value(weight_decay, 'weight_decay'))
                elif group['weight_decay'] != 0:
                    if isinstance(weight_decay, tp.Tensor):
                        grad = grad + param * weight_decay
                    else:
                        grad = grad.add(param, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    if differentiable:
                        max_exp_avg_sq = elementwise_max(
                            max_exp_avg_sq.clone(), exp_avg_sq
                        )
                        state['max_exp_avg_sq'] = max_exp_avg_sq
                    else:
                        max_exp_avg_sq.copy_(
                            elementwise_max(max_exp_avg_sq, exp_avg_sq)
                        )

                if capturable or differentiable:
                    bias_correction1 = 1 - scalar_pow(beta1, step_t)
                    bias_correction2 = 1 - scalar_pow(beta2, step_t)
                    if amsgrad:
                        denom = (
                            max_exp_avg_sq.sqrt() / bias_correction2.sqrt()
                        ).add_(group['eps'])
                    else:
                        denom = (
                            exp_avg_sq.sqrt() / bias_correction2.sqrt()
                        ).add_(group['eps'])
                    step_size = lr / bias_correction1 if isinstance(lr, tp.Tensor) else lr_value / bias_correction1
                    update = exp_avg.clone() if differentiable else exp_avg
                    param.addcdiv_(update, denom, value=-step_size)
                else:
                    step = scalar_value(step_t, 'step')
                    bias_correction1 = 1 - scalar_pow(beta1, step)
                    bias_correction2 = 1 - scalar_pow(beta2, step)
                    if amsgrad:
                        denom = (
                            max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                        ).add_(group['eps'])
                    else:
                        denom = (
                            exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                        ).add_(group['eps'])
                    step_size = lr_value / bias_correction1
                    param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
