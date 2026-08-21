import tensorplay as tp
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import _batchable, sgd as _foreach_sgd, sgd_foreach as _sgd_foreach
from ._fused import sgd as _fused_sgd
from ._utils import foreach_enabled, scalar_value

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False,
                 foreach=None, differentiable=False, fused=None):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "lr") < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if scalar_value(weight_decay, "weight_decay") < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if dampening < 0.0:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        if fused and foreach:
            # Match torch.optim.SGD's mutually-exclusive dispatch options.
            raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, fused=fused)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        if fused and differentiable:
            raise RuntimeError("`fused` does not support `differentiable`")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            maximize = group.get('maximize', False)
            foreach = group.get('foreach', None)
            fused = group.get('fused', False)

            active = [p for p in group['params'] if p.grad is not None]
            if active and fused:
                if any(p.grad.is_sparse or p.is_complex() or
                       not p.is_floating_point() for p in active):
                    raise RuntimeError(
                        "`fused` does not support sparse gradients or complex/non-floating parameters"
                    )
                # The native fused kernels consume flat contiguous buffers.
                # Preserve Torch's logical-gradient behavior for strided views
                # by materializing only those gradients.
                fused_grads = [
                    p.grad if p.grad.is_contiguous() else p.grad.clone()
                    for p in active
                ]
                fused_buffers = []
                candidate_buffers = []
                fused_state_valid = True
                if momentum != 0:
                    states = [self.state.setdefault(p, {}) for p in active]
                    has_buffers = ["momentum_buffer" in state for state in states]
                    if all(has_buffers):
                        fused_buffers = [state["momentum_buffer"] for state in states]
                    elif not any(has_buffers):
                        candidate_buffers = [grad.clone().detach() for grad in fused_grads]
                        fused_buffers = candidate_buffers
                    else:
                        fused_state_valid = False
                if (fused_state_valid and
                        not isinstance(weight_decay, tp.Tensor) and
                        _fused_sgd(
                            active, fused_grads, fused_buffers,
                            lr=(lr.to(device=active[0].device)
                                if isinstance(lr, tp.Tensor) and
                                lr.device != active[0].device else lr),
                            momentum=momentum, dampening=dampening,
                            weight_decay=weight_decay, nesterov=nesterov,
                            maximize=maximize,
                            is_first_step=bool(momentum != 0 and candidate_buffers),
                            grad_scale=getattr(self, "grad_scale", None),
                            found_inf=getattr(self, "found_inf", None),
                        )):
                    if candidate_buffers:
                        for p, buffer in zip(active, candidate_buffers):
                            self.state[p]["momentum_buffer"] = buffer
                    continue

            if (active and not fused and foreach_enabled(group, active)
                    and not group.get('differentiable', False)):
                params = active
                # Torch's foreach optimizer accepts strided gradients and
                # consumes their logical values.  The native TensorPlay
                # kernel intentionally uses flat pointers, so materialize
                # only the uncommon non-contiguous gradients (ResNet's final
                # linear weight is one such view) instead of disabling the
                # whole horizontal-fusion path.
                grads = [
                    p.grad if p.grad.is_contiguous() else p.grad.clone()
                    for p in active
                ]
                first_momentum_step = False
                candidate_buffers = None
                can_foreach = True
                if momentum != 0:
                    states = [self.state.setdefault(p, {}) for p in active]
                    has_buffers = ["momentum_buffer" in state for state in states]
                    if all(has_buffers):
                        buffers = [state["momentum_buffer"] for state in states]
                    elif not any(has_buffers):
                        # Torch's foreach path initializes every missing
                        # momentum buffer from the effective gradient, then
                        # performs the parameter update as one list operation.
                        # The native kernel applies weight decay and the
                        # first-step (dampening-free) momentum formula into
                        # these candidates, so state is published only after
                        # the native eligibility check succeeds.
                        candidate_buffers = [grad.clone().detach() for grad in grads]
                        buffers = candidate_buffers
                        first_momentum_step = True
                    else:
                        can_foreach = False
                        buffers = []
                else:
                    buffers = [tp.Tensor() for _ in active]

                state_lists = ([buffers] if momentum != 0 else [])
                if (can_foreach and not maximize and
                        not (isinstance(lr, tp.Tensor) and lr.requires_grad) and
                        not (isinstance(weight_decay, tp.Tensor) and weight_decay.requires_grad) and
                        _batchable(params, grads, state_lists)):
                    fast_success = _foreach_sgd(
                        params, grads, buffers, lr=lr, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov,
                        first_momentum_step=first_momentum_step)
                    if not fast_success:
                        fast_success = _sgd_foreach(
                            params, grads, buffers, lr=lr, momentum=momentum,
                            dampening=dampening, weight_decay=weight_decay,
                            nesterov=nesterov, maximize=maximize,
                            first_momentum_step=first_momentum_step)
                    if fast_success:
                        if candidate_buffers is not None:
                            for p, buffer in zip(active, candidate_buffers):
                                self.state[p]["momentum_buffer"] = buffer
                        continue
                elif can_foreach and maximize:
                    # The legacy native _foreach_sgd schema predates Torch's
                    # maximize flag.  Route maximize through the generic
                    # Torch foreach implementation instead of silently
                    # applying the descent gradient.
                    fast_success = _sgd_foreach(
                        params, grads, buffers, lr=lr, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, maximize=True,
                        first_momentum_step=first_momentum_step)
                    if fast_success:
                        if candidate_buffers is not None:
                            for p, buffer in zip(active, candidate_buffers):
                                self.state[p]["momentum_buffer"] = buffer
                        continue

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = -p.grad if maximize else p.grad
                
                if weight_decay != 0:
                    if isinstance(weight_decay, tp.Tensor) and weight_decay.requires_grad:
                        d_p = d_p + p * weight_decay
                    else:
                        d_p = d_p.add(
                            p, alpha=scalar_value(weight_decay, "weight_decay")
                        )
                
                if momentum != 0:
                    param_state = self.state.setdefault(p, {})
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone().detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if isinstance(lr, tp.Tensor) and lr.requires_grad:
                    p.addcmul_(d_p, lr, value=-1)
                else:
                    p.add_(d_p, alpha=-scalar_value(lr, "lr"))

        return loss
