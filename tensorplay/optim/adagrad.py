import tensorplay as tp
from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value, eps=eps)
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = tp.full_like(p, initial_accumulator_value)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError("weight_decay option is not supported for sparse gradients")
                    grad = grad.add(p, alpha=group['weight_decay'])

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    raise RuntimeError("Adagrad does not support sparse gradients")
                
                # state['sum'].addcmul_(grad, grad, value=1)
                state['sum'].add_(grad * grad)
                std = state['sum'].sqrt().add_(group['eps'])
                # p.data.addcdiv_(grad, std, value=-clr)
                p.data.add_(grad / std, alpha=-clr)

        return loss
