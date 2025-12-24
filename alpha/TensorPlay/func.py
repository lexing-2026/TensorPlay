from .core import Tensor
from .operator import MeanSquaredError, CrossEntropy

# =============================================================================
# 损失函数
# =============================================================================
def mse(out: Tensor, target: Tensor) -> Tensor:
    """均方误差（Mean Squared Error）：MSE = (1/n) * sum((a - b)²)"""
    return MeanSquaredError()(out, target)


def cross_entropy(out: Tensor, target: Tensor, axis: int = 1, activation: str = 'softmax') -> Tensor:
    """交叉熵损失函数"""
    return CrossEntropy(axis=axis, activation=activation)(out, target)


def sse(out: Tensor, target: Tensor) -> Tensor:
    """平方误差（Sum of Squared Error）：SSE = sum((a - b)²)"""
    if out.shape != target.shape:
        raise ValueError("SSE can only be calculated between tensors of the same shape")
    return ((out - target) ** 2).sum()


def nll(out: Tensor, target: Tensor) -> Tensor:
    """交叉熵误差（Negative Log Likelihood）：NLL = -sum(target * log(output))"""
    if out.shape != target.shape:
        raise ValueError("NLL can only be calculated between tensors of the same shape")
    return -(target * out.log()).sum()



# =============================================================================
# 优化函数
# =============================================================================
def sphere(x, y):
    return x ** 2 + y ** 2


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


def higher_optimizer(x, epoch, func, verbose=False):
    """
    基于牛顿法的二阶单变量函数优化器
    :param x: 初始值
    :param epoch: 迭代次数
    :param func: 目标函数
    :param verbose: 是否打印每次迭代的结果
    :return: 优化后的变量和函数值
    """
    for i in range(epoch):
        y = func(x)
        y.name = 'y'
        if verbose:
            print(f"第{i + 1}次迭代: x={x.data}, y={y.data}")
        y.backward(higher_grad=True)
        gx = x.grad
        gx.name = 'gx'
        x.zero_grad()
        gx.backward()
        gx2 = x.grad
        x.zero_grad()
        x.data -= gx.data / gx2.data
    return x, y

# =============================================================================
# 测试函数
# =============================================================================
def deriv(func, x: Tensor, eps=1e-4):
    """函数的数值微分计算，测试用"""
    x1 = Tensor(x.data.data + eps)
    x2 = Tensor(x.data.data - eps)
    y1 = func(x1)
    y2 = func(x2)
    return (y1.data.data - y2.data.data) / (2 * eps)