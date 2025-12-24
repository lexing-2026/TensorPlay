from .core import Optimizer
import numpy as np


class SGD(Optimizer):
    """随机梯度下降优化器"""

    def __init__(self, params=None, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def _step(self):
        for ten in self.params:
            ten.data -= self.lr * ten.grad.data
            ten.zero_grad()


class Momentum(Optimizer):
    """动量优化器"""

    def __init__(self, params=None, lr=0.001, gamma=0.8):
        super().__init__(params)
        self.momentum = [np.zeros(ten.shape) for ten in self.params]
        self.lr = lr
        self.gamma = gamma

    def _step(self):
        for i, tensor in enumerate(self.params):
            self.momentum[i] = self.gamma * self.momentum[i] + (1 - self.gamma) * tensor.grad.data
            tensor.data -= self.momentum[i] * self.lr
            tensor.zero_grad()


class Adam(Optimizer):
    """Adaptive Moment Estimation（自适应矩估计）"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.m = [np.zeros(ten.shape) for ten in self.params]  # 一阶动量
        self.s = [np.zeros(ten.shape) for ten in self.params]  # 二阶动量
        self.times = 1  # 时间步
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def _step(self):
        for i, tensor in enumerate(self.params):
            # 更新一阶动量
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            # 更新二阶动量
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad.data ** 2
            # 偏差修正
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            tensor.data -= (self.lr * cm) / (cs ** 0.5 + self.eps)
            tensor.zero_grad()
        self.times += 1


class AdamW(Optimizer):
    """AdamW优化器，在Adam基础上改进了权重衰减"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4):
        super().__init__(params)
        self.m = [np.zeros(ten.shape) for ten in self.params]
        self.s = [np.zeros(ten.shape) for ten in self.params]
        self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay  # 权重衰减系数

    def _step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad.data ** 2
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # 引入权重衰减
            tensor.data -= self.lr * (cm / (cs ** 0.5 + self.eps) + self.weight_decay * tensor.data)
            tensor.zero_grad()
        self.times += 1


class Nadam(Optimizer):
    """Nadam优化器，结合Nesterov动量和Adam"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.m = [np.zeros(ten.shape) for ten in self.params]
        self.s = [np.zeros(ten.shape) for ten in self.params]
        self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def _step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * (tensor.grad.data ** 2)
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # Nadam更新：融入Nesterov动量
            tensor.data -= (self.lr * (self.b1 * cm + (1 - self.b1) * tensor.grad.data / (1 - self.b1 ** self.times)) /
                    (cs ** 0.5 + self.eps))
            tensor.zero_grad()

        self.times += 1


class Lookahead(Optimizer):
    """Lookahead优化器，使用主优化器和慢更新策略"""

    def __init__(self, params=None, base_optimizer=Adam, k=5, alpha=0.5, **kwargs):
        super().__init__(params)
        # 初始化基础优化器（Adam）
        self.base_optimizer = base_optimizer(params, **kwargs)
        # 慢权重（初始化为参数的副本）
        self.slow_weights = [ten.data.copy() for ten in self.params]
        self.k = k  # 慢更新间隔
        self.alpha = alpha  # 插值系数
        self.step_counter = 0  # 步数计数器

    def _step(self):
        # 调用基础优化器的step方法（快更新）
        self.base_optimizer.step()
        self.step_counter += 1

        # 每k步执行慢更新
        if self.step_counter % self.k == 0:
            for i in range(len(self.params)):
                # 慢权重更新：slow = slow + alpha * (fast - slow)
                self.slow_weights[i] += self.alpha * (self.params[i].data - self.slow_weights[i])
                # 将慢权重复制回参数
                self.params[i].data = self.slow_weights[i]


class RMSprop(Optimizer):
    """Root Mean Square Propagation（RMSprop），基于梯度平方的移动平均"""

    def __init__(self, params=None, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.s = [np.zeros(ten.shape) for ten in self.params]  # 二阶动量
        self.alpha = alpha  # 衰减系数
        self.lr = lr
        self.eps = eps

    def _step(self):
        for i, tensor in enumerate(self.params):
            # 更新二阶动量：s = alpha*s + (1-alpha)*grad^2
            self.s[i] = self.alpha * self.s[i] + (1 - self.alpha) * (tensor.grad.data ** 2)
            # 参数更新：theta = theta - lr * grad / (sqrt(s) + eps)
            tensor.data -= self.lr * tensor.grad.data / (self.s[i] ** 0.5 + self.eps)
            tensor.zero_grad()