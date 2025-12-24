from typing import List, Union, Tuple
from .utils import im2col_array, col2im_array
from .core import Operator, Tensor, Config
import numpy as np

# =============================================================================
# 加载算子
# =============================================================================
def load_operator():
    """加载算子"""
    Tensor.__add__ = add
    Tensor.__radd__ = add
    Tensor.__neg__ = neg
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__mul__ = mul
    Tensor.__rmul__ = rmul
    Tensor.__matmul__ = matmul
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
    Tensor.__pow__ = ten_pow
    Tensor.__getitem__ = ten_slice
    Tensor.reslice = reslice
    Tensor.sum = ten_sum
    Tensor.max = ten_max
    Tensor.min = ten_min
    Tensor.exp = exp
    Tensor.log = log
    Tensor.mean = mean
    Tensor.relu = relu
    Tensor.leaky_relu = leaky_relu
    Tensor.gelu = gelu
    Tensor.tanh = tanh
    Tensor.clip = clip
    Tensor.dropout = dropout
    Tensor.expand = expand
    Tensor.reshape = reshape
    Tensor.flatten = flatten
    Tensor.sigmoid = sigmoid
    Tensor.softmax = softmax
    Tensor.log_softmax = log_softmax
    Tensor.broadcast = broadcast
    Tensor.rebroadcast = rebroadcast
    Tensor.transpose = transpose
    Tensor.T = T

# =============================================================================
# 四则运算算子
# =============================================================================
class Add(Operator):
    """加法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a + b)

    def _backward(self) -> List[Tensor]:
        ga, gb = self.out().grad, self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]

def add(a: Tensor, b: Tensor) -> Tensor:
    return Add()(a, b)

class Sub(Operator):
    """减法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a - b)

    def _backward(self) -> List[Tensor]:
        ga, gb = self.out().grad, -self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]

def sub(a: Tensor, b: Tensor) -> Tensor:
    return Sub()(a, b)

def rsub(a: Tensor, b: Tensor) -> Tensor:
    return Sub()(b, a)

class Neg(Operator):
    """取负算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(-a)

    def _backward(self) -> List[Tensor]:
        return [-self.out().grad]


def neg(a: Tensor) -> Tensor:
    return Neg()(a)


class Mul(Operator):
    """乘法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a * b)

    def _backward(self) -> List[Tensor]:
        ga = self.out().grad * self.inp[1]
        gb = self.out().grad * self.inp[0]
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]


def mul(a: Tensor, b: Tensor) -> Tensor:
    return Mul()(a, b)


def rmul(a: Tensor, b: Tensor) -> Tensor:
    return Mul()(b, a)


class Div(Operator):
    """除法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a / b)

    def _backward(self) -> List[Tensor]:
        ga, gb = (self.inp[1] ** -1) * self.out().grad, (-self.inp[0] * self.inp[1] ** -2) * self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]


def div(a: Tensor, b: Tensor) -> Tensor:
    return Div()(a, b)


def rdiv(a: Tensor, b: Tensor) -> Tensor:
    return Div()(b, a)


class Pow(Operator):
    """幂算子"""

    def __init__(self, power: float):
        super().__init__()
        self.power = power

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a ** self.power)

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += n * x ^ (n - 1) * 输出梯度
        return [self.power * self.inp[0] ** (self.power - 1) * self.out().grad]


def ten_pow(a: Tensor, b: float) -> Tensor:
    return Pow(b)(a)

# =============================================================================
# 超越算子
# =============================================================================
class Exp(Operator):
    """自然指数算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(np.exp(a))

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += e ^ x * 输出梯度 = 输出值 * 输出梯度
        return [self.out().data * self.out().grad]

def exp(x: Tensor) -> Tensor:
    return Exp()(x)


class Log(Operator):
    """自然对数算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        # 防止对数输入为非正数
        data = a.copy()
        data[data <= 0] = 1e-10
        return Tensor(np.log(data))

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += (1 / x) * 输出梯度 = 输出梯度 * (输入值的倒数)
        return [self.out().grad * self.inp[0] ** -1]


def log(x: Tensor) -> Tensor:
    return Log()(x)


class Relu(Operator):
    """ReLU修正线性单元"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(np.maximum(a, 0))

    def _backward(self) -> List[Tensor]:
        mask = Tensor(self.inp[0].data >= 0)
        return [self.out().grad * mask]


def relu(a: Tensor) -> Tensor:
    return Relu()(a)


class LeakyReLU(Operator):
    """LeakyReLU修正线性单元"""

    def __init__(self, slope: float = 0.2):
        super().__init__()
        self.slope = slope

    def _forward(self, x: np.ndarray) -> Tensor:
        return Tensor(np.where(x < 0, x * self.slope, x))

    def _backward(self) -> List[Tensor]:
        mask = np.where(self.inp[0].data < 0, self.slope, 1.0)
        return [self.out().grad * Tensor(mask)]


def leaky_relu(x: Tensor, slope: float = 0.2) -> Tensor:
    return LeakyReLU(slope)(x)


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def _forward(self, a: np.ndarray) -> Tensor:
        mask_large = a > 20
        mask_small = a < -20
        mask_mid = ~mask_large & ~mask_small
        result = np.empty_like(a, dtype=a.dtype)
        result[mask_large] = 1.0
        result[mask_small] = np.exp(a[mask_small]) / (1 + np.exp(a[mask_small]))
        result[mask_mid] = 1 / (1 + np.exp(-a[mask_mid]))
        return Tensor(result)

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += σ(x) * (1 - σ(x)) * 输出梯度
        return [self.out() * (1 - self.out()) * self.out().grad]


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid()(x)


class Gelu(Operator):
    """GeLU高斯线性单元"""

    def _forward(self, a: np.ndarray) -> Tensor:
        mask_large = a > 20
        mask_small = a < -20
        mask_mid = ~mask_large & ~mask_small
        result = np.empty_like(a, dtype=a.dtype)
        result[mask_large] = 1.0
        result[mask_small] = np.exp(a[mask_small] * 1.702) / (1 + np.exp(a[mask_small] * 1.702))
        result[mask_mid] = 1 / (1 + np.exp(-a[mask_mid] * 1.702))
        return Tensor(result)

    def _backward(self) -> List[Tensor]:
        z = (self.inp[0] * 1.702).sigmoid()
        return [(z + self.inp[0] * 1.702 * z * (1 - z)) * self.out().grad]


def gelu(x: Tensor) -> Tensor:
    return Gelu()(x)


class Softmax(Operator):
    """softmax激活函数"""

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def _forward(self, a: np.ndarray) -> Tensor:
        # 数值稳定，防止除零
        exp_tensor = np.exp(a - a.max(axis=self.axis, keepdims=True))
        sum_exp = exp_tensor.sum(axis=self.axis, keepdims=True)
        return Tensor(exp_tensor / (sum_exp + 1e-10))

    def _backward(self) -> List[Tensor]:
        gx = self.out() * self.out().grad
        return [gx- self.out() * gx.sum(axis=self.axis, dims=True)]


def softmax(x: Tensor, axis: int = 1) -> Tensor:
    return Softmax(axis)(x)


class LogSoftmax(Operator):
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def _forward(self, a: np.ndarray) -> Tensor:
        delta = a - a.max(axis=self.axis, keepdims=True)
        return Tensor(delta - np.log(1e-10 + np.sum(np.exp(delta), axis=self.axis, keepdims=True)))

    def _backward(self) -> List[Tensor]:
        return [self.out().grad - self.out().exp() * self.out().grad.sum(axis=self.axis, dims=True)]


def log_softmax(x: Tensor, axis: int = 1) -> Tensor:
    return LogSoftmax(axis)(x)


class Tanh(Operator):
    """Tanh激活函数"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(np.tanh(a))

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += (1 - tanh²(x)) * 输出梯度
        return [(1 - self.out() ** 2) * self.out().grad]


def tanh(x: Tensor) -> Tensor:
    return Tanh()(x)

# =============================================================================
# 操作算子
# =============================================================================
class Broadcast(Operator):
    """广播算子"""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def _forward(self, x: np.ndarray) -> Tensor:
        return Tensor(np.broadcast_to(x, self.shape))

    def _backward(self) -> List[Tensor]:
        return [rebroadcast(self.out().grad, *self.inp[0].shape)]


def broadcast(x: Tensor, *shape: int) -> Tensor:
    if x.shape == shape:
        return x
    return Broadcast(*shape)(x)


class Rebroadcast(Operator):
    """逆广播算子"""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def _forward(self, x: np.ndarray) -> Tensor:
        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return Tensor(y)

    def _backward(self) -> List[Tensor]:
        return [broadcast(self.out().grad, *self.inp[0].shape)]


def rebroadcast(x: Tensor, *shape: int) -> Tensor:
    if x.shape == shape:
        return x
    return Rebroadcast(*shape)(x)


class Sum(Operator):
    """求和算子"""

    def __init__(self, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.sum(axis=self.axis, keepdims=self.dims))

    def _backward(self) -> List[Tensor]:
        # 扩展梯度以匹配原始形状
        g = self.out().grad
        if self.axis is not None and not self.dims:
            self.axis = self.axis if isinstance(self.axis, Tuple) else (self.axis,)
            old = list(self.inp[0].shape)
            for a in self.axis:
                old[a] = 1
            g = self.out().grad.reshape(*tuple(old))
        return [broadcast(g, *self.inp[0].shape)]


def ten_sum(x: Tensor, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False) -> Tensor:
    return Sum(axis, dims)(x)


def mean(x: Tensor, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False) -> Tensor:
    y = ten_sum(x, axis, dims)
    return y * (y.data.size / x.data.size)


class Reshape(Operator):
    """变形算子"""

    def __init__(self, *shape: int):
        super().__init__()
        self.re = shape

    def _forward(self, a: np.ndarray) -> Tensor:
        out = Tensor(a.reshape(*self.re))
        return out

    def _backward(self) -> List[Tensor]:
        return [self.out().grad.reshape(*self.inp[0].shape)]


def reshape(x: Tensor, *shape: int) -> Tensor:
    return Reshape(*shape)(x)


def expand(x: Tensor, axis: int) -> Tensor:
    """扩展张量的维度"""
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, *shape)


def flatten(x: Tensor) -> Tensor:
    """展平算子（保留批次维度）"""
    return reshape(x, *(x.shape[0], -1))


class Transpose(Operator):
    """转置算子"""

    def __init__(self, *axes: int):
        super().__init__()
        self.axes = axes

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.transpose(*self.axes))

    def _backward(self) -> List[Tensor]:
        reverse_axes = tuple(np.argsort(self.axes))  # 计算逆变换
        return [self.out().grad.transpose(*reverse_axes)]


def transpose(x: Tensor, *axes: int) -> Tensor:
    return Transpose(*axes)(x)


@property
def T(self: Tensor) -> Tensor:
    return Transpose(*range(self.ndim - 1, -1, -1))(self)


class Slice(Operator):
    """切片算子（不支持重复索引）"""

    def __init__(self, slices: Union[slice, Tuple[slice, ...]]):
        super().__init__()
        self.slices = slices if isinstance(slices, tuple) else (slices,)

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a[self.slices])

    def _backward(self) -> List[Tensor]:
        return [reslice(self.out().grad, self.slices, self.inp[0].shape)]


def ten_slice(x: Tensor, slices) -> Tensor:
    return Slice(slices)(x)


class Reslice(Operator):
    """逆切片算子（不支持重复索引）"""

    def __init__(self, slices: Union[slice, Tuple[slice, ...]], in_shape: Tuple[int, ...]):
        super().__init__()
        self.slices = slices if isinstance(slices, tuple) else (slices,)
        self.in_shape = in_shape

    def _forward(self, a: np.ndarray) -> Tensor:
        a_grad = np.zeros(self.in_shape)
        a_grad[self.slices] = a
        return Tensor(a_grad)

    def _backward(self) -> List[Tensor]:
        return [ten_slice(self.out().grad, self.slices)]


def reslice(x: Tensor, slices, shape) -> Tensor:
    return Reslice(slices, shape)(x)


class Concatenate(Operator):
    """拼接算子"""

    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def _forward(self, *tensors: np.ndarray) -> Tensor:
        if not tensors:
            raise ValueError("Input tensor list is empty!")
        out = Tensor(np.concatenate([t for t in tensors], axis=self.axis))
        return out

    def _backward(self) -> List[Tensor]:
        # 计算每个张量对应的梯度切片
        current = 0
        g = []
        for tensor in self.inp:
            size = tensor.shape[self.axis]
            slices = [slice(None)] * self.out().ndim
            slices[self.axis] = slice(current, current + size)
            g.append(self.out().grad[tuple(slices)])
            current += size
        return g


def concatenate(*tensors: Tensor, axis: int = 0) -> Tensor:
    return Concatenate(axis=axis)(*tensors)


class Max(Operator):
    """最大值算子"""

    def __init__(self, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.max(axis=self.axis, keepdims=self.dims))

    def _backward(self) -> List[Tensor]:
        # 扩展梯度以匹配原始形状
        mask = (self.inp[0].data == self.out().data)
        g = self.out().grad
        if self.axis is not None and not self.dims:
            self.axis = self.axis if isinstance(self.axis, Tuple) else (self.axis,)
            old = list(self.inp[0].shape)
            for a in self.axis:
                old[a] = 1
            g = self.out().grad.reshape(*tuple(old))
        return [g * mask.astype(g.dtype)]


def ten_max(x: Tensor, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False) -> Tensor:
    return Max(axis, dims)(x)


class Min(Operator):
    """最小值算子"""

    def __init__(self, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.min(axis=self.axis, keepdims=self.dims))

    def _backward(self) -> List[Tensor]:
        mask = (self.inp[0].data == self.out().data)
        g = self.out().grad
        if self.axis is not None and not self.dims:
            self.axis = self.axis if isinstance(self.axis, Tuple) else (self.axis,)
            old = list(self.inp[0].shape)
            for a in self.axis:
                old[a] = 1
            g = self.out().grad.reshape(*tuple(old))
        return [g * mask.astype(g.dtype)]


def ten_min(x: Tensor, axis: Union[int, Tuple[int, ...]] = None, dims: bool = False) -> Tensor:
    return Min(axis, dims)(x)


class Clip(Operator):
    """裁剪算子"""

    def __init__(self, x_min: float, x_max: float):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def _forward(self, x: np.ndarray) -> Tensor:
        return Tensor(x.clip(self.x_min, self.x_max))

    def _backward(self) -> List[Tensor]:
        mask = (self.inp[0].data >= self.x_min) * (self.inp[0].data <= self.x_max)
        return [self.out().grad * mask.astype(self.out().dtype)]


def clip(x: Tensor, x_min: float, x_max: float) -> Tensor:
    return Clip(x_min, x_max)(x)


def dropout(x: Tensor, ratio=0.5):
    if Config.training:
        mask = (np.random.rand(*x.shape) > ratio).astype(x.dtype)
        y = x * mask / ratio
        return y
    else:
        return x

# =============================================================================
# 损失算子
# =============================================================================
class MeanSquaredError(Operator):
    """均方误差算子"""
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        out = Tensor(np.square(a - b).sum(axis=self.axis, keepdims=True) / a.shape[-1])
        return out

    def _backward(self) -> List[Tensor]:
        g = self.out().grad * 2 * (self.inp[0] - self.inp[1]) / self.inp[0].shape[-1]
        return [g, -g]


class CrossEntropy(Operator):
    """交叉熵算子（必须从批次维度开始）"""
    def __init__(self, axis: int = 1, activation: str = 'softmax'):
        super().__init__()
        self.activation = activation
        self.axis = axis

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        if self.activation == 'softmax':
            if b.ndim == 1:
                # 转换为one-hot编码，提高后续计算效率
                batch_size, num_classes = a.shape
                one_hot = np.zeros((batch_size, num_classes), dtype=a.dtype)
                one_hot[np.arange(batch_size), b.astype(np.int32)] = 1.0
                b = one_hot
            delta = a - a.max(axis=self.axis, keepdims=True)
            y = delta - np.log(1e-10 + np.sum(np.exp(delta), axis=self.axis, keepdims=True))
            # 计算交叉熵损失：-sum(target * log_softmax) / batch_size
            loss = -np.sum(b * y, axis=self.axis, keepdims=True)
            loss = np.mean(loss, axis=0)
        elif self.activation == 'sigmoid':
            # 二分类：标签保持原格式（无需one-hot，通常为0或1）
            log_sigmoid = np.where(a >= 0, -np.log(1 + np.exp(-a)), a - np.log(1 + np.exp(a)))
            log_one_minus_sigmoid = np.where(a >= 0, -a - np.log(1 + np.exp(-a)), -np.log(1 + np.exp(a)))
            # 二元交叉熵公式：-mean(b*log(sigmoid) + (1-b)*log(1-sigmoid))
            loss = -np.mean(b * log_sigmoid + (1 - b) * log_one_minus_sigmoid, axis=self.axis, keepdims=True)
            loss = np.mean(loss, axis=0)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        return Tensor(loss)

    def _backward(self) -> List[Tensor]:
        batch_size = self.inp[0].shape[0]
        if self.activation == 'softmax':
            # 交叉熵梯度：(softmax - target) / batch_size
            g = (self.inp[0].softmax(axis=self.axis) - self.inp[1]) / batch_size
        elif self.activation == 'sigmoid':
            # 二分类梯度：(sigmoid - target) / batch_size
            g = (self.inp[0].sigmoid() - self.inp[1]) / batch_size
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        return [g, None]

# =============================================================================
# 线性层算子
# =============================================================================
class MatMul(Operator):
    """矩阵乘法运算符"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a @ b)

    def _backward(self) -> List[Tensor]:
        return [self.out().grad @ self.inp[1].transpose(-1, -2),
               self.inp[0].transpose(-1, -2) @ self.out().grad]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class DenseOp(Operator):
    """全连接层运算符"""

    def __init__(self, matrix: 'Dense'):
        super().__init__()
        self.w = matrix.w
        self.b = matrix.b
        self.bias = matrix.bias

    def _forward(self, x: np.ndarray) -> Tensor:
        if x.ndim == 1:
            x = x.reshape(1, -1)  # 单个样本转为 (1 × in_features)

        # 矩阵乘法：W @ x.T = (out_features × batch_size)
        out = self.w.data @ x.transpose(-1, -2)
        if self.bias is not None:
            out += self.b.data

        # 转置回 (batch_size × out_features)
        if x.ndim == 1:
            out = out.reshape(-1)
        return Tensor(out.transpose(-1, -2))

    def _backward(self):
        # 1. 处理输出梯度形状 (batch_size × out_features → out_features × batch_size)
        grad = self.out().grad
        a = self.inp[0]
        if self.inp[0].ndim == 1:
            grad = grad.reshape(1, -1)
            a = a.reshape(1, -1)
        grad = grad.transpose(-1, -2)
        # 2. 计算偏置梯度（对所有样本梯度求和）
        if self.bias is not None:
            if self.b.grad is None:
                self.b.grad = grad.sum(axis=1, dims=True)
            else:
                self.b.grad = self.b.grad + grad.sum(axis=1, dims=True)
        # 3. 矩阵乘法的反向传播（计算权重梯度）
        if self.w.grad is None:
            self.w.grad = grad @ a
        else:
            self.w.grad = self.w.grad + grad @ a
        b = (self.w.transpose(-1, -2) @ grad).transpose(-1, -2)
        if self.inp[0].ndim == 1:
            b = b.reshape(*self.inp[0].shape)
        return [b]


class BatchNormOp(Operator):
    """批量标准化处理层（Batch Normalization）"""

    def __init__(self, params: 'BatchNorm'):
        super().__init__()
        self.eps = params.eps
        self.decay = params.decay
        self.gamma = params.gamma
        self.beta = params.beta
        self.running_mean = params.running_mean
        self.running_var = params.running_var

    def _forward(self, x: np.ndarray) -> Tensor:
        x_ndim = x.ndim
        assert x_ndim in (2, 4), f"Only for 2D or 4D input, but got: {x_ndim}"
        if x_ndim == 4:
            batch_size, height, width, channels = x.shape
            # (N,H,W,C) -> (N*H*W, C)
            x = x.reshape(-1, channels)
        if Config.training:
            m = x.mean(axis=0)  # 按特征/通道计算均值
            var = x.var(axis=0)  # 按特征/通道计算方差
            inv_std = 1 / np.sqrt(var + self.eps)
            xc = (x - m) * inv_std
            n = x.size // self.gamma.size  # 单个特征的样本数量
            s = n - 1 if n - 1 > 1 else 1
            adjust = n / s
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * m
            self.running_var = self.decay * self.running_var + (1 - self.decay) * var * adjust
            self.inv_std = inv_std
        else:
            inv_std = 1 / np.sqrt(self.running_var + self.eps)
            xc = (x - self.running_mean) * inv_std
        y = xc * self.gamma.data + self.beta.data
        if x_ndim == 4:
            y = y.reshape(batch_size, height, width, channels)
        return Tensor(y)

    def _backward(self) -> List[Tensor]:
        if self.inp[0].ndim == 4:
            B, H, W, C = self.inp[0].shape
            gy = self.out().grad.reshape(-1, C)
            x = self.inp[0].reshape(-1, C)
        else:
            gy = self.out().grad
            x = self.inp[0]
        xc = (x - x.mean(axis=0, dims=True)) * self.inv_std
        if self.gamma.grad is None:
            self.gamma.grad = (xc * gy).sum(axis=0)
        else:
            self.gamma.grad = self.gamma.grad + (xc * gy).sum(axis=0)
        if self.beta.grad is None:
            self.beta.grad = gy.sum(axis=0)
        else:
            self.beta.grad = self.beta.grad + gy.sum(axis=0)
        gx = (gy - self.beta.grad / self.inp[0].shape[0] - xc * self.gamma.grad / self.inp[0].shape[0]) * self.gamma * self.inv_std
        if self.inp[0].ndim == 4:
            gx = gx.reshape(B, H, W, C)
        return [gx]


class LayerNormOp(Operator):
    """层标准化处理层（Layer Normalization）"""

    def __init__(self, params: 'LayerNorm'):
        super().__init__()
        self.eps = params.eps
        self.gamma = params.gamma
        self.beta = params.beta

    def _forward(self, x: np.ndarray) -> Tensor:
        x_ndim = x.ndim
        assert x_ndim in (2, 4), f"Only for 2D or 4D input, but got: {x_ndim}"
        if x_ndim == 4:
            # 均值形状：(N, 1, 1, 1)
            m = x.mean(axis=(1, 2, 3), keepdims=True)
            var = x.var(axis=(1, 2, 3), keepdims=True)
        else:
            m = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
        inv_std = 1 / np.sqrt(var + self.eps)
        xc = (x - m) * inv_std
        if Config.training:
            self.m = Tensor(m)
            self.var = Tensor(var)
            self.inv_std = Tensor(inv_std)
            self.xc = Tensor(xc)
        return Tensor(xc * self.gamma.data + self.beta.data)

    def _backward(self) -> List[Tensor]:
        if self.inp[0].ndim == 4:
            N, H, W, C = self.inp[0].shape
            n = H * W * C
            gamma_grad = (self.xc * self.out().grad).sum()
            beta_grad = self.out().grad.sum()
        else:
            N, features = self.inp[0].shape
            n = features
            gamma_grad = (self.xc * self.out().grad).sum()
            beta_grad = self.out().grad.sum()
        if self.gamma.grad is None:
            self.gamma.grad = gamma_grad
        else:
            self.gamma.grad = self.gamma.grad + gamma_grad
        if self.beta.grad is None:
            self.beta.grad = beta_grad
        else:
            self.beta.grad = self.beta.grad + beta_grad
        gx_normalized = self.out().grad * self.gamma
        gvar = (gx_normalized * (self.inp[0] - self.m) * (-0.5) * (self.var + self.eps) ** (-1.5)).sum(axis=1, dims=True)
        gmean = (gx_normalized * (-self.inv_std)).sum(axis=1, dims=True) + gvar * (-2 / n) * (self.inp[0] - self.m).sum(
            axis=1, dims=True)
        gx = gx_normalized * self.inv_std + gvar * (2 / n) * (self.inp[0] - self.m) + gmean / n
        return [gx]

# =============================================================================
# 卷积层算子
# =============================================================================
class Conv2DOp(Operator):
    def __init__(self, strides: Tuple[int, int], padding: Tuple[int, int]):
        super().__init__()
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray, w: np.ndarray, b: Union[np.ndarray, None]) -> Tensor:
        KH, KW, _, OC = w.shape
        B, H, W, C = x.shape
        # (B*OH*OW, C*KH*KW) * (C*KH*KW, OC) -> (B*OH*OW, OC)
        col = im2col_array(x, (KH, KW), self.strides, self.padding)
        OH, OW = col.shape[1:3]
        y = col.reshape(-1, C * KH * KW) @ w.reshape(-1, OC)
        y = y.reshape(B, OH, OW, OC)
        if b is not None:
            y += b
        return Tensor(y)

    def _backward(self) -> List[Tensor]:
        gx = ReConv2DOp(self.strides, self.padding)(self.out().grad, self.inp[1], None)
        gW = Conv2DGradW(self.inp[1].shape[0:2], self.strides, self.padding)(self.inp[0], self.out().grad)
        if self.inp[2] is not None:
            gb = self.out().grad.sum(axis=(0, 1, 2))
        else:
            gb = None
        return [gx, gW, gb]


class ReConv2DOp(Operator):
    def __init__(self, strides: Tuple[int, int], padding: Tuple[int, int]):
        super().__init__()
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray, w: np.ndarray, b: Union[np.ndarray, None]) -> Tensor:
        B, OH, OW, OC = x.shape
        KH, KW, C, _ = w.shape
        gcol = (x.reshape(-1, OC) @ w.reshape(OC, -1)).reshape(B, OH, OW, C, KH, KW)
        y = col2im_array(gcol, (B, OH, OW, OC), (KH, KW), self.strides, self.padding)
        # (B, H, W, C)
        if b is not None:
            y += b
        return Tensor(y)

    def _backward(self) -> List[Tensor]:
        gx = Conv2DOp(self.strides, self.padding)(self.out().grad, self.inp[1], None)
        gW = Conv2DGradW(self.inp[1].shape[0:2], self.strides, self.padding)(self.out().grad, self.inp[0])
        if self.inp[2] is not None:
            gb = self.out().grad.sum(axis=(0, 1, 2))
        else:
            gb = None
        return [gx, gW, gb]


class Conv2DGradW(Operator):
    def __init__(self, kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: Tuple[int, int]):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray, gy: np.ndarray) -> Tensor:
        OC = gy.shape[3]
        col = im2col_array(x, self.kernel_size, self.strides, self.padding)
        C, KH, KW = col.shape[3:]
        return Tensor((col.reshape(KH * KW * C, -1) @ gy.reshape(-1, OC)).reshape(KH, KW, C, OC))

    def _backward(self) -> List[Tensor]:
        gx = ReConv2DOp(self.strides, self.padding)(self.inp[1], self.out().grad, None)
        ggy = Conv2DOp(self.strides, self.padding)(self.inp[0], self.out().grad, None)
        return [gx, ggy]

# =============================================================================
# 池化层算子
# =============================================================================
class MaxPoolingOp(Operator):
    def __init__(self, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray) -> Tensor:
        col = im2col_array(x, self.kernel_size, self.strides, self.padding)
        B, OH, OW, C, KH, KW = col.shape
        col = col.reshape(B, KH * KW, OH, OW, C)
        self.indexes = col.argmax(axis=1)
        return Tensor(col.max(axis=1))

    def _backward(self) -> List[Tensor]:
        return [Pooling2DGrad(self.kernel_size, self.indexes, self.strides, self.padding)(self.out().grad, self.inp[0])]


class Pooling2DGrad(Operator):
    def __init__(self, kernel_size: Tuple[int, int], indexes: np.ndarray,
                 strides: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.indexes = indexes

    def _forward(self, gy: np.ndarray, x: np.ndarray) -> Tensor:
        B, OH, OW, C = gy.shape
        B, H, W, C = x.shape
        KH, KW = self.kernel_size
        # 绝对索引
        gcol = np.zeros((B * OH * OW * C * KH * KW), dtype=x.dtype)
        indexes = (self.indexes.ravel() + np.arange(0, self.indexes.size * KH * KW, KH * KW))
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(B, C, OH, OW, KH, KW).transpose(0, 1, 4, 5, 2, 3)
        # (B, C, KH, KW, OH, OW) -> (B, H, W, C)
        gx = col2im_array(gcol, (B, H, W, C), self.kernel_size, self.strides, self.padding)
        return Tensor(gx)

    def _backward(self) -> List[Tensor]:
        return [Pooling2DWithIndexes(self.mpool2d)(self.out().grad)]


class Pooling2DWithIndexes(Operator):
    def __init__(self, kernel_size: Tuple[int, int], indexes: np.ndarray,
                 strides: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.indexes = indexes

    def _forward(self, x: np.ndarray) -> Tensor:
        col = im2col_array(x, self.kernel_size, self.strides, self.padding)
        B, OH, OW, C, KH, KW = col.shape
        col = col.reshape(B, C, KH * KW, OH, OW).transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes].reshape(B, OH, OW, C)
        return Tensor(col)

    def _backward(self) -> List[Tensor]:
        pass


class AveragePoolingOp(Operator):
    def __init__(self, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_shape = None

    def _forward(self, x: np.ndarray) -> Tensor:
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.strides, self.padding)
        return Tensor(col.mean(axis=(2, 3)))

    def _backward(self) -> List[Tensor]:
        B, OH, OW, C = self.out().grad.shape
        KW, KH = self.kernel_size
        gcol = broadcast((self.out().grad / (KW * KH)).reshape(-1), *(KH, KW, B * C * OH * OW))
        gcol = gcol.reshape(KH, KW, B, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = Col2im(self.input_shape, self.kernel_size, self.strides, self.padding)(gcol)
        return [gx]


class Im2col(Operator):
    def __init__(self, kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: Tuple[int, int]):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray) -> Tensor:
        self.input_shape = x.shape
        return Tensor(im2col_array(x, self.kernel_size, self.strides, self.padding))

    def _backward(self) -> List[Tensor]:
        return [Col2im(self.input_shape, self.kernel_size, self.strides, self.padding)(self.out().grad)]


class Col2im(Operator):
    def __init__(self, input_shape, kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: Tuple[int, int]):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def _forward(self, x: np.ndarray) -> Tensor:
        return Tensor(col2im_array(x, self.input_shape, self.kernel_size, self.strides, self.padding))

    def _backward(self) -> List[Tensor]:
        return [Im2col(self.kernel_size, self.strides, self.padding)(self.out().grad)]
