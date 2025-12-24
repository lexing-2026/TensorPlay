import tensorplay as tp
from .optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = tp.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = tp.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = tp.zeros_like(p)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                square_avg.mul_(alpha).add_(grad * grad, alpha=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    # avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                    avg = square_avg.add(grad_avg * grad_avg, alpha=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    buf.mul_(group['momentum']).add_(grad / avg)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    # p.data.addcdiv_(grad, avg, value=-group['lr'])
                    p.data.add_(grad / avg, alpha=-group['lr'])

        return loss
