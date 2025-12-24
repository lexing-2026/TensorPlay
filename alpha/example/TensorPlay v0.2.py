import json
import weakref
import numpy as np
from typing import List, Union, Tuple, Optional, Any, Callable


class Vector:
    """向量，默认使用float32"""
    precision = np.float32

    def __init__(self, data: Union[list, np.ndarray, int, float]):
        if isinstance(data, np.ndarray):
            self.data = data.astype(Vector.precision)
        else:
            self.data = np.array(data, dtype=Vector.precision)
        # 检查元素是否为数字类型
        if not np.issubdtype(self.data.dtype, np.number):
            raise TypeError(f"Vector elements must be numbers (not {self.data.dtype})")

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"

    def __str__(self) -> str:
        return str(self.data.tolist())

    def __neg__(self) -> 'Vector':
        return Vector(-self.data)

    def __add__(self, other: Union['Vector', int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            return Vector(self.data + other)
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Cannot add vectors with different shapes {self.shape} vs {other.shape}")
            return Vector(self.data + other.data)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")

    def __radd__(self, other: Union[int, float]) -> 'Vector':
        return self.__add__(other)

    def __iadd__(self, other: Union['Vector', int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            self.data += other
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Cannot add vectors with different shapes {self.shape} vs {other.shape}")
            self.data += other.data
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")
        return self

    def __sub__(self, other: Union['Vector', int, float]) -> 'Vector':
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> 'Vector':
        return (-self) + other

    def __mul__(self, other: Union['Vector', int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            return Vector(self.data * other)
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Cannot multiply vectors with different shapes {self.shape} vs {other.shape}")
            return Vector(self.data * other.data)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

    def __rmul__(self, other: Union[int, float]) -> 'Vector':
        return self.__mul__(other)

    def __imul__(self, other: Union['Vector', int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            self.data *= other
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Cannot multiply vectors with different shapes {self.shape} vs {other.shape}")
            self.data *= other.data
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")
        return self

    def __truediv__(self, other: Union['Vector', int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            return Vector(self.data / other)
        elif isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Cannot divide vectors with different shapes {self.shape} vs {other.shape}")
            return Vector(self.data / other.data)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector' and '{type(other).__name__}'")

    def __rtruediv__(self, other: Union[int, float]) -> 'Vector':
        if isinstance(other, (int, float)):
            return Vector(other / self.data)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'Vector'")

    def __pow__(self, power: Union[int, float]) -> 'Vector':
        if isinstance(power, (int, float)):
            return Vector(self.data ** power)
        else:
            raise TypeError(f"Cannot power Vector with {type(power).__name__}")

    def __rpow__(self, base: Union[int, float]) -> 'Vector':
        if isinstance(base, (int, float)):
            return Vector(base ** self.data)
        else:
            raise TypeError(f"Cannot power {type(base).__name__} with Vector")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @classmethod
    def set_precision(cls, pre: type):
        """设置向量的精度，参数为np.dtype"""
        if not np.issubdtype(pre, np.floating):
            raise TypeError(f"Precision must be a float type (not {pre})")
        cls.precision = pre

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> 'Vector':
        return Vector(np.zeros(shape, dtype=cls.precision))

    @classmethod
    def ones(cls, shape: Union[int, Tuple[int, ...]]) -> 'Vector':
        return Vector(np.ones(shape, dtype=cls.precision))


class Tensor:
    """张量：(B, H, W, C)"""

    def __init__(self, data: Union[list, np.ndarray, Vector, int, float], op=None, name: str = None):
        self.vector = Vector(data) if not isinstance(data, Vector) else data
        self.grad = Vector.zeros(self.vector.shape)
        self.op = op if Operator.enable_grad else None
        self.name = name
        self.rank = 0
        self.source_module = None  # 用于钩子机制

    def __repr__(self) -> str:
        return f"Tensor({self.vector.data}, grad={self.grad.data})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.vector.data.shape

    @property
    def ndim(self) -> int:
        return self.vector.data.ndim

    @property
    def size(self) -> int:
        return self.vector.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.vector.data.dtype

    def __len__(self):
        return len(self.vector.data)

    def __add__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other)
        op = Add()
        return op(self, other)

    def __sub__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other)
        op = Sub()
        return op(self, other)

    def __neg__(self) -> 'Tensor':
        op = Neg()
        return op(self)

    def __mul__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other)
        op = Mul()
        return op(self, other)

    def __truediv__(self, other: Union[int, float, 'Tensor']) -> 'Tensor':
        if isinstance(other, (int, float)):
            other = Tensor(other)
        op = Div()
        return op(self, other)

    def __pow__(self, power: float) -> 'Tensor':
        op = Pow(power)
        return op(self)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        op = MatMul()
        return op(self, other)

    def sum(self, axis: Optional[int] = None, dims: bool = False) -> 'Tensor':
        op = Sum(axis=axis, dims=dims)
        return op(self)

    def reshape(self, *shape: int) -> 'Tensor':
        op = Reshape(shape)
        return op(self)

    def transpose(self, *axes: int) -> 'Tensor':
        op = Transpose(axes)
        return op(self)

    def relu(self) -> 'Tensor':
        op = Relu()
        return op(self)

    def exp(self) -> 'Tensor':
        op = Exp()
        return op(self)

    def log(self) -> 'Tensor':
        op = Log()
        return op(self)

    def sigmoid(self) -> 'Tensor':
        op = Sigmoid()
        return op(self)

    def tanh(self) -> 'Tensor':
        op = Tanh()
        return op(self)

    def slice(self, slices: Union[slice, Tuple[slice, ...]]) -> 'Tensor':
        op = Slice(slices)
        return op(self)

    def mean(self, axis: Optional[int] = None, dims: bool = False) -> 'Tensor':
        """计算张量的均值"""
        op = Mean(axis=axis, dims=dims)
        return op(self)

    def repeat(self, repeats: Union[int, Tuple[int, ...]], axis: int = None) -> 'Tensor':
        """扩展张量维度"""
        op = Repeat(repeats, axis=axis)
        return op(self)

    @classmethod
    def concatenate(cls, tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """按维度连接多个张量"""
        op = Concatenate(axis=axis)
        return op(tensors)

    def clone(self) -> 'Tensor':
        """返回当前张量的副本，确保梯度独立"""
        cloned_tensor = Tensor(self.vector, op=self.op)
        cloned_tensor.grad = Vector(self.grad.data.copy())
        return cloned_tensor

    def detach(self) -> 'Tensor':
        """返回一个不追踪梯度的张量副本"""
        detached = Tensor(self.vector)
        return detached

    def softmax(self, axis: int = -1) -> 'Tensor':
        """softmax激活函数：softmax(x) = e^x / sum(e^x_j)"""
        exp_tensor = self.exp()
        sum_exp = exp_tensor.sum(axis=axis, dims=True).repeat(self.shape[axis], axis=axis)
        # 数值稳定，防止除零
        return exp_tensor / (sum_exp + Tensor(np.ones(self.shape) * 1e-10))

    def gelu(self) -> 'Tensor':
        """GELU激活函数：GELU(x) ≈ x * Sigmoid(1.702x)"""
        return self * (self * Tensor(1.702 * np.ones(self.shape))).sigmoid()

    @classmethod
    def mse(cls, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        """均方误差（Mean Squared Error）：MSE = (1/n) * sum((a - b)²)"""
        if a.shape != b.shape:
            raise ValueError("MSE can only be calculated between tensors of the same shape")
        return ((a - b) ** 2).sum() / Tensor(a.size)

    @classmethod
    def sse(cls, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        """平方误差（Sum of Squared Error）：SSE = sum((a - b)²)"""
        if a.shape != b.shape:
            raise ValueError("SSE can only be calculated between tensors of the same shape")
        return ((a - b) ** 2).sum()

    @classmethod
    def nll(cls, out: 'Tensor', target: 'Tensor') -> 'Tensor':
        """交叉熵误差（Negative Log Likelihood）：NLL = -sum(target * log(output))"""
        return -(target * out.log()).sum()

    def zero_grad(self) -> None:
        """清空梯度"""
        self.grad = Vector.zeros(self.vector.data.shape)

    def one_grad(self) -> None:
        """将梯度设为1"""
        self.grad = Vector.ones(self.vector.data.shape)

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> 'Tensor':
        """创建一个指定形状的全0张量"""
        return Tensor(Vector.zeros(shape))

    def backward(self, retain_grad: bool = True) -> None:
        """
        计算子图的反向传播
        :param retain_grad: bool 是否保留下游梯度
        """
        self.one_grad()
        # 使用集合避免重复处理同一运算符
        op_set = set()
        queue = []
        # 从输出张量开始，收集所有相关运算符
        if self.op is not None:
            queue.append(self.op)
            op_set.add(self.op)
        # 广度优先搜索收集所有相关运算符
        while queue:
            current_op = queue.pop(0)  # 取出队首元素
            if current_op.inp is None:
                continue
            # 处理输入为列表或单个张量的情况
            inputs = current_op.inp if isinstance(current_op.inp, list) else [current_op.inp]
            for inp_tensor in inputs:
                # 确保输入是张量且有运算符，排除起始张量
                if isinstance(inp_tensor, Tensor) and inp_tensor.op is not None and inp_tensor.op not in op_set:
                    op_set.add(inp_tensor.op)
                    queue.append(inp_tensor.op)
        # 按算符深度和计算顺序逆序处理
        op_list = sorted(op_set, key=lambda x: (x.rank, Operator.compute_list.index(x)), reverse=True)

        for op in op_list:
            if op.inp is None:
                continue
            op.propagate_grad()
            if not retain_grad:
                # 后续实现内存释放
                pass
        Operator.clean(specific_ops=op_set)


class Operator:
    """运算符基类，支持多维张量操作"""
    compute_list: List['Operator'] = []  # 记录计算顺序
    enable_grad: bool = True  # 控制是否追踪梯度

    def __init__(self):
        """子类根据需要重写，没有额外参数不写"""
        if self.enable_grad:
            self.compute_list.append(self)
        self.inp = None  # 输入张量
        self.out = None  # 输出张量
        self.rank = 0

    def __repr__(self) -> str:
        return f"TensorOperator.{self.__class__.__name__}"

    def __call__(self, *args):
        """前向调用接口"""
        out = self._forward(*args)
        if self.enable_grad:
            self.rank = max([inp.rank for inp in args])
            out.rank = self.rank + 1
        return out

    def propagate_grad(self) -> None:
        """后向调用接口，集成反向钩子调用"""
        if not self.enable_grad:
            raise RuntimeError(
                "Cannot call propagate_grad() when tracing gradient computation graph is disabled. "
                "Enable Operator() using set_grad_enabled(True)."
            )
        self._backward()
        # 调用反向钩子
        if self.out.source_module is not None:
            module = self.out.source_module
            module._call_backward_hooks(self.out, self.inp)

    def _forward(self, *args: Any) -> Tensor:
        """前向具体运算"""
        raise NotImplementedError

    def _backward(self) -> None:
        """后向具体计算"""
        raise NotImplementedError

    @classmethod
    def set_grad_enabled(cls, mode: bool) -> None:
        """控制是否开启梯度追踪"""
        cls.enable_grad = mode

    @classmethod
    def backward(cls, last1grad: bool = True) -> None:
        """全局反向传播"""
        if not cls.compute_list:  # 无运算记录时直接返回
            return
        reversed_ops = list(reversed(cls.compute_list))
        if last1grad:
            reversed_ops[0].out.one_grad()
        # 逆序遍历运算执行梯度反向传播
        for op in reversed_ops:
            if op.inp is None:
                continue
            op.propagate_grad()
        cls.clean()

    @classmethod
    def clean(cls, specific_ops: Optional[set['Operator']] = None) -> None:
        """清理计算图数据"""
        if specific_ops is not None:
            for op in specific_ops:
                if op in list(cls.compute_list):
                    cls.compute_list.remove(op)
                    del op
        else:
            while cls.compute_list:
                op = cls.compute_list.pop()
                del op
            cls.compute_list.clear()


class no_grad:
    """临时禁用梯度计算的上下文管理器"""

    def __enter__(self) -> 'no_grad':
        self.prev_mode = Operator.enable_grad
        Operator.set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Operator.set_grad_enabled(self.prev_mode)


class Add(Operator):
    """加法运算符"""

    def _forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(a.vector + b.vector, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        self.inp[0].grad += self.out.grad
        self.inp[1].grad += self.out.grad


class Sub(Operator):
    """减法运算符"""

    def _forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(a.vector - b.vector, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        self.inp[0].grad += self.out.grad
        self.inp[1].grad -= self.out.grad


class Neg(Operator):
    """取负运算符"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(-a.vector, op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        self.inp.grad.data -= self.out.grad.data


class Mul(Operator):
    """乘法运算符"""

    def _forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(a.vector * b.vector, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 乘法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += self.inp[1].vector * self.out.grad
        self.inp[1].grad += self.inp[0].vector * self.out.grad


class MatMul(Operator):
    """矩阵乘法运算符"""

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(a.vector.data @ b.vector.data, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        self.inp[0].grad.data += self.out.grad @ self.inp[1].vector.data.T
        self.inp[1].grad.data += self.inp[0].vector.data.T @ self.out.grad.data


class Div(Operator):
    """除法运算符"""

    def _forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(a.vector / b.vector, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 除法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += (self.inp[1].vector ** -1) * self.out.grad
        self.inp[1].grad += (-self.inp[0].vector * self.inp[1].vector ** -2) * self.out.grad


class Pow(Operator):
    """幂运算符"""

    def __init__(self, power: float):
        super().__init__()
        self.power = power

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector ** self.power, op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度 += n * x ^ (n - 1) * 输出梯度
        self.inp.grad += self.power * self.inp.vector ** (self.power - 1) * self.out.grad


class Exp(Operator):
    """自然指数函数"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.exp(a.vector.data), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度 += e ^ x * 输出梯度 = 输出值 * 输出梯度
        self.inp.grad += self.out.vector * self.out.grad


class Log(Operator):
    """自然对数函数"""

    def _forward(self, a: Tensor) -> Tensor:
        # 防止对数输入为非正数
        data = a.vector.data.copy()
        data[data <= 0] = 1e-10
        out = Tensor(np.log(data), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度 += (1 / x) * 输出梯度 = 输出梯度 * (输入值的倒数)
        self.inp.grad += self.out.grad * self.inp.vector ** -1


class Sum(Operator):
    """元素级求和运算符"""

    def __init__(self, axis: Optional[int] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector.data.sum(axis=self.axis, keepdims=self.dims), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 扩展梯度以匹配原始形状
        if self.axis is not None and not self.dims:
            old = list(self.inp.vector.data.shape)
            self.axis = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in self.axis:
                old[ax] = 1
            self.out.grad.data = self.out.grad.data.reshape(old)
        self.inp.grad.data += np.broadcast_to(self.out.grad.data, self.inp.shape)


class Reshape(Operator):
    """重塑形状运算符"""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.re = shape

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector.data.reshape(self.re), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        self.inp.grad.data += self.out.grad.data.reshape(self.inp.shape)


class Transpose(Operator):
    """转置运算符"""

    def __init__(self, axes: Tuple[int, ...]):
        super().__init__()
        self.axes = axes

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector.data.transpose(self.axes), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        # 梯度转置回原始形状
        reverse_axes = np.argsort(self.axes)  # 计算逆变换
        self.inp.grad.data += self.out.grad.data.transpose(reverse_axes)


class Slice(Operator):
    """切片运算符"""

    def __init__(self, slices: Union[slice, Tuple[slice, ...]]):
        super().__init__()
        self.slices = slices if isinstance(slices, tuple) else (slices,)

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector.data[self.slices], op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        # 创建与原始张量形状相同的零梯度
        a_grad = np.zeros_like(self.inp.vector.data)
        # 将梯度赋值到对应的切片位置
        a_grad[self.slices] = self.out.grad.data
        self.inp.grad.data += a_grad


class Mean(Operator):
    """平均值运算符"""

    def __init__(self, axis: Optional[int] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims
        self.normalizer = None  # 存储归一化系数，避免反向传播时重复计算

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.vector.data.mean(axis=self.axis, keepdims=self.dims), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
            if self.axis is None:
                self.normalizer = a.size
            else:
                self.axis = (self.axis,) if isinstance(self.axis, int) else self.axis
                self.normalizer = 1
                for ax in self.axis:
                    self.normalizer *= a.shape[ax]
        return out

    def _backward(self):
        # 扩展梯度以匹配原始形状
        if self.axis is not None and not self.dims:
            old = list(self.inp.vector.data.shape)
            self.axis = (self.axis,) if isinstance(self.axis, int) else self.axis
            for ax in self.axis:
                old[ax] = 1
            self.out.grad.data = self.out.grad.data.reshape(old)
        self.inp.grad.data += np.broadcast_to(self.out.grad.data, self.inp.shape) / self.normalizer


class Concatenate(Operator):
    """拼接运算符"""

    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def _forward(self, tensors: List[Tensor]) -> Tensor:
        if not tensors:
            raise ValueError("Input tensor list is empty!")
        # 检查拼接维度外的其他维度是否匹配
        shapes = [t.vector.data.shape for t in tensors]
        for i in range(len(shapes[0])):
            if i != self.axis:
                dims = {s[i] for s in shapes}
                if len(dims) > 1:
                    raise ValueError(f"All input tensors must have the same size in dimension {i}")
        out = Tensor(np.concatenate([t.vector.data for t in tensors], axis=self.axis), op=self)
        if self.enable_grad:
            self.inp = tensors
            self.out = weakref.proxy(out)
        return out

    def _backward(self) -> None:
        # 计算每个张量对应的梯度切片
        current = 0
        for tensor in self.inp:
            size = tensor.vector.data.shape[self.axis]
            slices = [slice(None)] * self.out.ndim
            slices[self.axis] = slice(current, current + size)
            tensor.grad.data += self.out.grad.data[tuple(slices)]
            current += size


class Repeat(Operator):
    """重复运算符"""

    def __init__(self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        super().__init__()
        self.repeats = repeats
        self.axis = axis
        self.repeat_indices = None  # 存储重复索引映射关系

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.repeat(a.vector.data, repeats=self.repeats, axis=self.axis), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
            if self.axis is None:
                n = a.vector.data.size
                if isinstance(self.repeats, tuple):
                    repeats = self.repeats
                else:
                    repeats = (self.repeats,) * n
                indices = []
                current = 0
                for rep in repeats:
                    indices.append((current, current + rep))
                    current += rep
                self.repeat_indices = indices
            else:
                axis = self.axis
                axis_size = a.vector.data.shape[axis]
                if isinstance(self.repeats, tuple):
                    repeats = self.repeats
                else:
                    repeats = (self.repeats,) * axis_size
                indices = []
                current = 0
                for rep in repeats:
                    indices.append((current, current + rep))
                    current += rep
                self.repeat_indices = (axis, indices)
        return out

    def _backward(self) -> None:
        grad = self.out.grad.data
        a_grad = np.zeros_like(self.inp.vector.data)
        if self.axis is None:
            for i, (start, end) in enumerate(self.repeat_indices):
                a_grad.flat[i] = np.sum(grad.flat[start:end])
        else:
            axis, indices = self.repeat_indices
            for i, (start, end) in enumerate(indices):
                slices = [slice(None)] * a_grad.ndim
                slices[axis] = slice(start, end)
                target_slices = [slice(None)] * a_grad.ndim
                target_slices[axis] = i
                a_grad[tuple(target_slices)] += np.sum(grad[tuple(slices)], axis=axis)
        self.inp.grad.data += a_grad


class Relu(Operator):
    """ReLU修正线性单元"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.maximum(a.vector.data, 0), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度+=输出梯度*输入函数（大于等于0）
        mask = (self.inp.vector.data >= 0).astype(Vector.precision)
        self.inp.grad.data += self.out.grad.data * mask


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-a.vector.data)), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度 += σ(x) * (1 - σ(x)) * 输出梯度
        self.inp.grad += self.out.vector * (1 - self.out.vector) * self.out.grad


class Tanh(Operator):
    """Tanh激活函数"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.tanh(a.vector.data), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 输入梯度 += (1 - tanh²(x)) * 输出梯度
        self.inp.grad += (1 - self.out.vector ** 2) * self.out.grad


class DenseOp(Operator):
    """线性层运算符"""

    def __init__(self, matrix: 'Dense'):
        super().__init__()
        self.w = matrix.w
        self.b = matrix.b
        self.bias = matrix.bias

    def _forward(self, x: Tensor) -> Tensor:
        a = x.vector.data
        if a.ndim == 1:
            a = a.reshape(1, -1)  # 单个样本转为 (1 × in_features)

        # 矩阵乘法：W @ x.T = (out_features × batch_size)
        out = self.w.vector.data @ a.T
        if self.bias is not None:
            out += self.b.vector.data.repeat(a.shape[0], axis=1)

        # 转置回 (batch_size × out_features)
        if x.ndim == 1:
            out = out.reshape(-1)
        out = Tensor(out.T, op=self)
        if self.enable_grad:
            self.inp = x
            self.out = weakref.proxy(out)
        return out

    def _backward(self):
        # 1. 处理输出梯度形状 (batch_size × out_features → out_features × batch_size)
        grad = self.out.grad.data
        a = self.inp.vector.data
        if self.inp.ndim == 1:
            grad = grad.reshape(1, -1)
            a = a.reshape(1, -1)
        grad = grad.T
        # 2. 计算偏置梯度（对所有样本梯度求和）
        if self.bias is not None:
            self.b.grad.data += np.sum(grad, axis=1, keepdims=True)
        # 3. 触发矩阵乘法的反向传播（计算权重梯度）
        self.w.grad.data += grad @ a
        b = (self.w.vector.data.T @ grad).T
        if self.inp.ndim == 1:
            b = b.reshape(self.inp.shape)
        self.inp.grad.data += b


class Layer:
    """
    基础参数层，实现钩子功能，所有参数层都需要继承此类
    save和load方法自定义格式，必须互认
    """
    layer_list: List['Layer'] = []  # 基础参数层全局记录，兼容最底层实现

    def __init__(self, *args):
        # 训练模式记录
        self.training = True
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_hooks = {}
        # 基础参数层只记录Layer类，Module以上不记录
        if isinstance(self, Module):
            return
        Layer.layer_list.append(self)

    def __repr__(self) -> str:
        prefix = '' if self.__class__.__name__ == 'Layer' else 'Layer.'
        return f"{prefix}{self.__class__.__name__}"

    def save(self, *args) -> str:
        """
        保存接口，所有继承了Layer的类需重写此方法
        :return: 自定义格式，与load方法互认
        """
        raise NotImplementedError

    def load(self, *args) -> None:
        """
        读取接口，所有继承了Layer的类需重写此方法
        :param args: str 自定义格式，与save方法互认
        """
        raise NotImplementedError

    def param(self) -> List[Tensor]:
        """
        参数接口，所有继承了Layer的类需重写此方法
        :return: list[Tensor]
        """
        raise NotImplementedError

    @classmethod
    def get_params(cls) -> List[Tensor]:
        """
        返回所有基础参数层参数，兼容优化器的默认设置
        :return: list[Tensor]
        """
        params = []
        for i in Layer.layer_list:
            # Param返回列表
            if i.param() is not None:
                params += i.param()
        return params

    def register_forward_pre_hook(self, hook: Callable) -> int:
        """注册前向传播前的钩子"""
        handle = id(hook)
        self._forward_pre_hooks[handle] = hook
        return handle

    def register_forward_hook(self, hook: Callable) -> int:
        """注册前向传播后的钩子"""
        handle = id(hook)
        self._forward_hooks[handle] = hook
        return handle

    def register_backward_hook(self, hook: Callable) -> int:
        """注册反向传播的钩子"""
        handle = id(hook)
        self._backward_hooks[handle] = hook
        return handle

    def remove_hook(self, handle: int) -> None:
        """移除指定钩子"""
        for hooks in [self._forward_pre_hooks, self._forward_hooks, self._backward_hooks]:
            if handle in hooks:
                del hooks[handle]
                return

    def _call_forward_pre_hooks(self, *args: Tensor, **kwargs) -> None:
        """调用前向传播前的钩子"""
        for hook in self._forward_pre_hooks.values():
            hook(self, args, kwargs)

    def _call_forward_hooks(self, *args: Tensor, **kwargs) -> None:
        """调用前向传播后的钩子"""
        for hook in self._forward_hooks.values():
            hook(self, args, kwargs, self._forward_result)

    def _call_backward_hooks(self, grad_outputs: Tensor, inputs: Tensor) -> None:
        """调用反向传播的钩子"""
        for hook in self._backward_hooks.values():
            if isinstance(inputs, Tensor):
                hook(self, grad_outputs, inputs)
            elif isinstance(inputs, list):
                for item in inputs:
                    if isinstance(item, Tensor):
                        hook(self, grad_outputs, [item for item in inputs])
            else:
                raise TypeError(f"input must be a Tensor or list of Tensors, got {type(inputs).__name__}")

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:
        """调用方法，集成钩子和张量-模块关联"""
        self._call_forward_pre_hooks(*args, **kwargs)
        self._forward_result = self.forward(*args, **kwargs)
        # 记录输出张量的来源模块（用于反向传播时触发钩子）
        if self._backward_hooks:
            if isinstance(self._forward_result, Tensor):
                self._forward_result._source_module = self
            elif isinstance(self._forward_result, list):
                for item in self._forward_result:
                    if isinstance(item, Tensor):
                        item._source_module = self
            else:
                raise TypeError(f"forward_result must be a Tensor or list, got {type(self._forward_result).__name__}")
        self._call_forward_hooks(*args, **kwargs)
        return self._forward_result

    def forward(self, *args: Tensor, **kwargs) -> Tensor:
        """前向传播方法，需要子类实现"""
        raise NotImplementedError(f"Module {self.__class__.__name__} has no forward method implemented")


class ConstantTensor(Tensor, Layer):
    """单张量参数层"""

    def __init__(self, data: Union[list, np.ndarray]):
        Tensor.__init__(self, data)
        Layer.__init__(self)

    def save(self) -> str:
        return json.dumps(self.vector.data.tolist())

    def load(self, text: str) -> None:
        self.vector = Vector(json.loads(text))

    def param(self):
        return [self]


class LayerNorm(Layer):
    """标准化处理层，带可学习参数（Layer Normalization）"""

    def __init__(self, shape: Union[int, Tuple[int, ...]], eps: float = 1e-4):
        self.w = Tensor(np.random.normal(0, 0.04, shape))
        self.b = Tensor(np.random.normal(0, 0.04, shape))
        self.eps = eps
        super().__init__()

    def forward(self, x: Tensor, eps=0.0001) -> Tensor:
        """前向传播：计算标准化并应用可学习参数"""
        batch_size = x.shape[0]

        # 计算均值 (沿批次维度)
        mean = x.sum(axis=0, dims=True) / Tensor([batch_size])
        mean_repeated = mean.repeat(batch_size, axis=0)

        # 计算方差
        x_minus_mean = x - mean_repeated
        var = (x_minus_mean ** 2).sum(axis=0, dims=True) / Tensor([batch_size])
        var_repeated = var.repeat(batch_size, axis=0)
        # 计算标准化结果
        std = (var_repeated + Tensor([eps])) ** 0.5
        x_normalized = x_minus_mean / std

        return self.w.repeat(batch_size, axis=0) * x_normalized + self.b.repeat(batch_size, axis=0)

    def save(self) -> str:
        text = {'w': self.w.vector.data.tolist(), 'b': self.b.vector.data.tolist()}
        return json.dumps(text)

    def load(self, text: str):
        parts = json.loads(text)
        self.w = Tensor(parts['w'])
        self.b = Tensor(parts['b'])

    def param(self) -> List[Tensor]:
        return [self.w, self.b]


class Dense(Layer):
    """全连接层"""

    def __init__(self, inp_size: int, out_size: int, bias=True):
        self.w = he_init(inp_size, out_size)
        self.bias = bias
        if bias:
            self.b = Tensor.zeros((out_size, 1))
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        op = DenseOp(self)
        out = op(a)
        return out

    def save(self) -> str:
        text = {'w': self.w.vector.data.tolist()}
        if self.bias:
            text['b'] = self.b.vector.data.tolist()
        return json.dumps(text)

    def load(self, text: str) -> None:
        parts = json.loads(text)
        self.w = Tensor(parts['w'])
        if self.bias:
            self.b = Tensor(parts['b'])

    def param(self) -> List[Tensor]:
        if self.bias:
            return [self.w, self.b]
        return [self.w]


class Conv2D(Layer):
    def __init__(self, width: int, height: int, stride_w=1, stride_h=1, pad=True, bias=True):
        """
        2d卷积层
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        self.width = width
        self.height = height
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad = pad
        self.kernel = my_init(width * height)
        self.bias = bias
        if bias:
            self.b = my_init(1)
        super().__init__()

    def padding(self, x):
        """
        填充
        :param x: list[Tensor(),Tensor()...]  2d的Ten，或者说列表包着的一列Ten
        :return: list[Tensor(),Tensor()...]
        """
        pad_x = (self.stride_w * (len(x[0]) - 1) - len(x[0]) + self.width) // 2
        pad_y = (self.stride_h * (len(x) - 1) - len(x) + self.height) // 2
        x2 = []
        for i in range(pad_y):
            x2.append(Tensor.zeros(len(x[0]) + pad_x * 2))
        for i in range(len(x)):
            x2.append(Tensor.connect([Tensor.zeros(pad_x), x[i], Tensor.zeros(pad_x)]))
        for i in range(pad_y):
            x2.append(Tensor.zeros(len(x[0]) + pad_x * 2))
        return x2

    def forward(self, x):
        """
        进行运算
        :param x: list[Tensor(),Tensor()...]  2d的Ten，或者说list包着的一列Ten
        :return: list[Tensor(),Tensor()...]
        """
        if self.pad:
            x = self.padding(x)
        x2 = []
        for y_pos in range(0, len(x) - self.height + 1, self.stride_h):
            x2line = []
            for x_pos in range(0, len(x[0]) - self.width + 1, self.stride_w):
                window = Tensor.connect([x[y_pos + i].cut(x_pos, x_pos + self.width) for i in range(self.height)])
                v = (window * self.kernel).sum()
                if self.bias:
                    v += self.b
                x2line.append(v)
            x2.append(Tensor.connect(x2line))
        return x2

    def save(self):
        t = f"{self.width}/{self.height}/{self.kernel.data}/{self.stride_w}/{self.stride_h}/{self.pad}"
        if self.bias:
            t += f"/{self.b.data}"
        return t

    def load(self, t):
        t = t.split("/")
        self.width = int(t[0])
        self.height = int(t[1])
        self.kernel = Tensor(eval(t[2]))
        self.stride_w = int(t[3])
        self.stride_h = int(t[4])
        self.pad = eval(t[5])
        if len(t) == 7:
            self.bias = True
            self.b = Tensor(eval(t[6]))
        else:
            self.bias = False

    def grad_descent_zero(self, lr):
        self.kernel.data -= self.kernel.grad * lr
        self.kernel.zero_grad()

    def param(self):
        return [self.kernel, self.b]


class Module(Layer):
    """模块基类，支持子模块管理"""

    def __init__(self):
        super().__setattr__('modules', {})  # 子模块字典
        super().__setattr__('parameters', {})  # 模块参数字典
        super().__setattr__('layers', {})  # 参数层字典
        super().__init__()

    def __setattr__(self, name: str, value):
        """属性设置，自动注册子模块和参数层"""
        # 基类最后注册，避免覆盖子类
        if isinstance(value, ConstantTensor):
            self.parameters[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Module):
            self.modules[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Layer):
            self.layers[name] = value
            super().__setattr__(name, value)
        else:
            # 管理参数不注册
            super().__setattr__(name, value)

    def __repr__(self):
        prefix = '' if self.__class__.__name__ == 'Module' else 'Module.'
        return f"{prefix}{self.__class__.__name__}"

    def params(self):
        """递归返回所有可训练参数"""
        param = []
        for p in self.parameters.values():
            param.extend(p.param())
        for l in self.layers.values():
            param.extend(l.param())
        for m in self.modules.values():
            param.extend(m.params())
        return param

    def named_params(self, prefix: str = ''):
        """递归返回带名称的参数"""
        named_param = []
        prefix = prefix + ('.' if prefix else '')
        for name, l in self.layers.items():
            named_param.append((prefix + name, l, sum([len(i) for i in l.param()])))
        for name, p in self.parameters.items():
            named_param.append((prefix + name, p, sum([len(i) for i in l.param()])))
        for name, m in self.modules.items():
            named_param.extend(m.named_params(prefix + name))
        return named_param

    def named_modules(self, prefix: str = ''):
        """返回带名称的子模块（包括自身）"""
        if prefix == '':
            prefix = self.__class__.__name__
        for name, module in self.modules.items():
            current_prefix = f"{prefix}.{name}" if prefix else name
            yield current_prefix, module
            yield from module.named_modules(current_prefix)

    def train(self, mode: bool = True):
        """设置训练模式"""
        Operator.set_grad_enabled(mode)
        self.training = mode
        for p in self.parameters.values():
            p.training = mode
        for l in self.layers.values():
            l.training = mode
        for module in self.modules.values():
            module.train(mode, )
        # 返回值方便链式调用
        return self

    def eval(self):
        """设置评估模式，复用train(False)"""
        Operator.set_grad_enabled(False)
        return self.train(False)

    def save(self, path: str):
        """将模块数据保存为JSON格式文件"""
        params = []
        layer = self.named_params()
        for param in layer:
            data = {"type": str(param[1]), "params": param[1].save()}
            params.append(data)
        datas = {"model": self.__class__.__name__, "layers_num": len(params), "parameters": params}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datas, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """从JSON格式文件加载模块数据"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found in: {path}!")
        if data.get("model") != self.__class__.__name__:
            raise ValueError(f"Model mismatched: expect {self.__class__.__name__}, got {data.get('model')}!")
        text = data.get("parameters")
        if text is None:
            raise ValueError(f"Got no parameters for {self.__class__.__name__}!")
        layer = self.named_params()
        l = len(layer)
        if data.get("layers_num") != l or len(text) != l:
            raise ValueError(f"Layers sizes mismatched: expect {l}, got {data.get('layers_num')}!")
        for i in range(l):
            if text[i]["type"] != str(layer[i][1]):
                raise ValueError(f"Layers type mismatched: expect {layer[i][1]}, got {text[i]['type']}!")
            layer[i][1].load(text[i]["params"])

    @staticmethod
    def _get_layer_info(layer):
        layer_type = str(layer).split(".")[-1]
        if layer_type == "Dense":
            return None, len(layer.w)
        elif layer_type == "Conv2D":
            return "Conv2D"
        elif layer_type == "AveragePooling2D":
            return "AveragePooling2D"
        elif layer_type == "Flatten":
            return "Flatten"
        else:
            return (None,)

    def summary(self):
        """打印模型摘要，包括各层类型、输出形状和参数"""
        # 收集层信息的列表
        layer = self.named_params()
        layers_info = []
        total_params = 0
        trainable_params = 0
        for p in layer:
            total_params += p[2]
            if p[1].training:
                trainable_params += p[2]
            out_shape = Module._get_layer_info(p[1])
            layers_info.append({
                "name": p[0],
                "type": str(p[1]).split(".")[-1],
                "output_shape": out_shape,
                "params": p[2]
            })

        max_name_len = max(len(f"{info['name']} ({info['type']})") for info in layers_info) + 4  # 增加4以留出边距
        max_shape_len = max(len(str(info["output_shape"])) for info in layers_info) + 4
        max_param_len = max(len(str(info["params"])) for info in layers_info) + 4

        # tf表头的奇怪配比（21、12、8）
        header_name_len = len("Layer (type)") + 21
        header_shape_len = len("Output Shape") + 12
        header_param_len = len("Param #") + 8

        max_name_len = max(max_name_len, header_name_len)
        max_shape_len = max(max_shape_len, header_shape_len)
        max_param_len = max(max_param_len, header_param_len)

        # 打印表头
        print(f"Model: \"{self.__class__.__name__}\"")
        print(f"┌{'─' * max_name_len}┬{'─' * max_shape_len}┬{'─' * max_param_len}┐")

        header = (f"│ {'Layer (type)':<{max_name_len - 2}} "
                  f"│ {'Output Shape':<{max_shape_len - 2}} "
                  f"│ {'Param #':>{max_param_len - 2}} │")
        print(header)

        # 打印分隔线
        print(f"├{'─' * max_name_len}┼{'─' * max_shape_len}┼{'─' * max_param_len}┤")

        # 打印各层信息
        for i, info in enumerate(layers_info):
            layer = f"{info['name']} ({info['type']})"
            shape = str(info['output_shape'])
            param = f"{info['params']:,}"  # 层参数添加千位分隔符

            row = (f"│ {layer:<{max_name_len - 2}} "
                   f"│ {shape:<{max_shape_len - 2}} "
                   f"│ {param:>{max_param_len - 2}} │")  # 参数右对齐更易读
            print(row)
            if i < len(layers_info) - 1:
                print(f"├{'─' * max_name_len}┼{'─' * max_shape_len}┼{'─' * max_param_len}┤")

        # 打印底部边框
        line = f"└{'─' * max_name_len}┴{'─' * max_shape_len}┴{'─' * max_param_len}┘"
        print(line)

        # 计算参数占用空间（float64）
        byte = 8

        def format_size(params):
            """ 将参数数量转换为最合适的存储单位"""
            bytes_size = params * byte
            units = ['B', 'KB', 'MB', 'GB']
            unit_index = 0
            size = bytes_size
            while size >= 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1
            return f"{size:.2f} {units[unit_index]}"

        # 打印总计信息，保持和表格对齐的缩进
        non_trainable = total_params - trainable_params
        print(f" Total params: {total_params:,} ({format_size(total_params)})")
        print(f" Trainable params: {trainable_params:,} ({format_size(trainable_params)})")
        print(f" Non-trainable params: {non_trainable:,} ({format_size(non_trainable)})")


class Sequential(Module):
    """顺序层容器，集中管理多个层"""

    def __init__(self, *args):
        super().__init__()
        self.container = []  # 记录模块顺序
        for i, module in enumerate(args):
            if not isinstance(module, (Layer, Module)):
                raise TypeError(f"Sequential need Layer or Module, not {type(module).__name__}")
            self.container.append(module)
            # 自动注册模块
            setattr(self, f"{self.__class__.__name__}_{i + 1}", module)

    def forward(self, x):
        """按顺序执行各层的前向传播"""
        for layer in self.container:
            x = layer(x)
        return x

    def __getitem__(self, index):
        """支持通过索引访问内部层"""
        return self.container[index]

    def __len__(self):
        """返回层数"""
        return len(self.container)


# 一些复杂结构的简单实现，用于实验，未更新到Module管理
class MiniDense:
    """低秩全连接层"""

    def __init__(self, inp_size, out_size, midsize=None, bias=True):
        if midsize is None:
            midsize = round(((inp_size + out_size) / 2) ** 0.5)
        self.f1 = Dense(inp_size, midsize, bias)
        self.f2 = Dense(midsize, out_size, bias)

    def __call__(self, x):
        x = self.f1(x)
        x = self.f2(x)
        return x

    def grad_descent_zero(self, lr):
        self.f1.grad_descent_zero(lr)
        self.f2.grad_descent_zero(lr)


class Attention:
    """单头自注意力模块"""

    def __init__(self, emb_size, qk_size=None, v_size=None):
        """
        :param emb_size: int 输入词向量维度
        :param qk_size: int q、k维度
        :param v_size: int 输出词向量维度，默认与输入相同
        """
        if qk_size is None:
            qk_size = emb_size // 2
        if v_size is None:
            v_size = emb_size
        self.q = Dense(emb_size, qk_size)
        self.k = Dense(emb_size, qk_size)
        self.v = Dense(emb_size, v_size)
        self.emb_size = emb_size
        self.qk_size = qk_size
        self.outsize = v_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        """
        :param x: list[Tensor,Tensor...]  装着词向量的列表
        :param mask_list: list[int,int...] 用于在softmax前盖住填充，输入中表中为1的位置会被替换为-inf
        :param tri_mask: bool 是否使用三角掩码（在计算注意力权重时只关注当前和之前的词）
        :return: list[Tensor,Tensor...]
        """
        q_list = []
        k_list = []
        v_list = []
        for w in x:
            q_list.append(self.q(w))
            k_list.append(self.k(w))
            v_list.append(self.v(w))
        att_list = []
        for i in range(len(q_list)):
            line = []
            for j in range(len(k_list)):
                if (mask_list is not None and (mask_list[i] == 1 or mask_list[j] == 1)) or (tri_mask and j > i):
                    line.append(Tensor([float("-inf")]))
                else:
                    line.append((q_list[i] * k_list[j]).sum() / Tensor([self.qk_size ** 0.5]))
            att_list.append(Tensor.connect(line).softmax())
        new_v_list = []
        for i in range(len(q_list)):
            line = Tensor.zeros(self.outsize)
            for j in range(len(q_list)):
                line += v_list[j] * (att_list[i].cut(j, j + 1).repeat(self.outsize))
            new_v_list.append(line)
        return new_v_list

    def grad_descent_zero(self, k):
        self.q.grad_descent_zero(k)
        self.k.grad_descent_zero(k)
        self.v.grad_descent_zero(k)


class LSTM:
    """长短期记忆网络"""

    def __init__(self, emb_size, out_size):
        self.for_gate = Dense(emb_size + out_size, out_size)
        self.inp_gate1 = Dense(emb_size + out_size, out_size)
        self.inp_gate2 = Dense(emb_size + out_size, out_size)
        self.out_gate = Dense(emb_size + out_size, out_size)
        self.h = Tensor.zeros(out_size)
        self.s = Tensor.zeros(out_size)

    def __call__(self, x):
        out = []
        for i in x:
            i = Tensor.connect([i, self.h])
            self.s *= self.for_gate(i).sigmoid()
            self.s += self.inp_gate1(i).sigmoid() * self.inp_gate2(i).tanh()
            self.h = self.out_gate(i).sigmoid() * self.s.tanh()
            out.append(self.h)
        return out

    def grad_descent_zero(self, lr):
        self.for_gate.grad_descent_zero(lr)
        self.inp_gate1.grad_descent_zero(lr)
        self.inp_gate2.grad_descent_zero(lr)
        self.out_gate.grad_descent_zero(lr)


class RNN:
    """线性循环神经网络"""

    def __init__(self, emb_size, out_size):
        """
        :param emb_size: int 输入的向量大小
        :param out_size: int 输出的向量大小
        """
        self.out_size = out_size
        self.f1 = Dense(emb_size + out_size, out_size)

    def __call__(self, x):
        """
        :param x: list[Tensor,Tensor...]
        :return: list[Tensor,Tensor...]
        """
        hidden = Tensor.zeros(self.out_size)
        out = []
        for i in x:
            hidden = self.f1(Tensor.connect([hidden, i]))
            out.append(hidden)
        return out

    def grad_descent_zero(self, lr):
        self.f1.grad_descent_zero(lr)


class Optimizer:
    """优化器类，储存参数需要重写接口函数save()和load()"""

    def __init__(self, params=None):
        """
        :param params:list[Tensor,Tensor...] 需要优化的参数的列表，为None时优化所有Layer中的参数
        """
        if params is None:
            self.params = Layer.get_params()
        else:
            self.params = params

    def step(self):
        """具体优化方法，必须重写"""
        raise NotImplementedError

    def zero_grad(self):
        """使参数的梯度归零"""
        for i in self.params:
            i.zero_grad()


class SGD(Optimizer):
    """随机梯度下降优化器"""

    def __init__(self, params=None, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for ten in self.params:
            ten.vector.data -= self.lr * ten.grad.data
            ten.zero_grad()


class Momentum(Optimizer):
    """动量优化器"""

    def __init__(self, params=None, lr=0.001, gamma=0.8):
        super().__init__(params)
        self.momentum = [np.zeros(ten.shape) for ten in self.params]
        self.lr = lr
        self.gamma = gamma

    def step(self):
        for i, tensor in enumerate(self.params):
            self.momentum[i] = self.gamma * self.momentum[i] + (1 - self.gamma) * tensor.grad.data
            tensor.vector.data -= self.momentum[i] * self.lr
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

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新一阶动量
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            # 更新二阶动量
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad.data ** 2
            # 偏差修正
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            tensor.vector.data -= (self.lr * cm) / (cs ** 0.5 + self.eps)
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

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad.data ** 2
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # 引入权重衰减
            tensor.vector.data -= self.lr * (cm / (cs ** 0.5 + self.eps) + self.weight_decay * tensor.vector.data)
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

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad.data
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * (tensor.grad.data ** 2)
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # Nadam更新：融入Nesterov动量
            tensor.vector.data -= (
                        self.lr * (self.b1 * cm + (1 - self.b1) * tensor.grad.data / (1 - self.b1 ** self.times)) /
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
        self.slow_weights = [ten.vector.data.copy() for ten in self.params]
        self.k = k  # 慢更新间隔
        self.alpha = alpha  # 插值系数
        self.step_counter = 0  # 步数计数器

    def step(self):
        # 调用基础优化器的step方法（快更新）
        self.base_optimizer.step()
        self.step_counter += 1

        # 每k步执行慢更新
        if self.step_counter % self.k == 0:
            for i in range(len(self.params)):
                # 慢权重更新：slow = slow + alpha * (fast - slow)
                self.slow_weights[i] += self.alpha * (self.params[i].vector.data - self.slow_weights[i])
                # 将慢权重复制回参数
                self.params[i].vector.data = self.slow_weights[i]


class RMSprop(Optimizer):
    """Root Mean Square Propagation（RMSprop），基于梯度平方的移动平均"""

    def __init__(self, params=None, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.s = [np.zeros(ten.shape) for ten in self.params]  # 二阶动量
        self.alpha = alpha  # 衰减系数
        self.lr = lr
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新二阶动量：s = alpha*s + (1-alpha)*grad^2
            self.s[i] = self.alpha * self.s[i] + (1 - self.alpha) * (tensor.grad.data ** 2)
            # 参数更新：theta = theta - lr * grad / (sqrt(s) + eps)
            tensor.vector.data -= self.lr * tensor.grad.data / (self.s[i] ** 0.5 + self.eps)
            tensor.zero_grad()


class LearningRateScheduler:
    """学习率调度器基类"""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.lr]  # 保存初始学习率
        self.step()

    def step(self):
        """更新学习率"""
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        """计算当前学习率，子类必须重写此方法"""
        raise NotImplementedError


class StepLR(LearningRateScheduler):
    """固定步长学习率调度器"""

    def __init__(self, optimizer, step_size=30, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 每过step_size个epoch，学习率乘以gamma
        return self.base_lrs[0] * (self.gamma ** (self.last_epoch // self.step_size))


class MultiStepLR(LearningRateScheduler):
    """多步学习率调度器"""

    def __init__(self, optimizer, milestones=None, gamma=0.1, last_epoch=-1):
        if milestones is None:
            milestones = [30, 60, 90]
        self.milestones = set(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 在指定milestones中的epoch，学习率乘以gamma
        if self.last_epoch in self.milestones:
            return self.optimizer.lr * self.gamma
        return self.optimizer.lr


class ExponentialLR(LearningRateScheduler):
    """指数衰减学习率调度器"""

    def __init__(self, optimizer, gamma=0.99, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 学习率按指数规律衰减: lr = lr * gamma^epoch
        return self.base_lrs[0] * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LearningRateScheduler):
    """余弦退火学习率调度器"""

    def __init__(self, optimizer, t_max=10, eta_min=0, last_epoch=-1):
        self.T_max = t_max  # 周期长度
        self.eta_min = eta_min  # 最小学习率
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 学习率按余弦曲线周期性变化
        return self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * \
            (1 + np.cos(np.pi * self.last_epoch / self.T_max))


class ReduceLROnPlateau:
    """自适应调度器，当指标停止改善时降低学习率"""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', min_lr=0):
        self.optimizer = optimizer
        self.mode = mode  # min表示指标越小越好，max表示指标越大越好
        self.factor = factor  # 学习率衰减因子
        self.patience = patience  # 容忍多少个epoch没有改善
        self.threshold = threshold  # 改善的阈值
        self.threshold_mode = threshold_mode  # rel表示相对变化，abs表示绝对变化
        self.min_lr = min_lr  # 最小学习率
        self.best = None
        self.num_bad_epochs = 0  # 记录连续没有改善的epoch数

    def step(self, metric):
        """
        根据监测指标更新学习率
        metric: 需要监测的指标值
        """
        if self.best is None:
            self.best = metric
            return

        if self.threshold_mode == 'rel':
            if self.mode == 'min':
                # 对于最小值指标，新值需要小于 best * (1 - threshold)才算改善
                improvement_threshold = self.best * (1 - self.threshold)
            else:
                # 对于最大值指标，新值需要大于 best * (1 + threshold)才算改善
                improvement_threshold = self.best * (1 + self.threshold)
        else:
            if self.mode == 'min':
                # 对于最小值指标，新值需要小于 best - threshold才算改善
                improvement_threshold = self.best - self.threshold
            else:
                # 对于最大值指标，新值需要大于 best + threshold才算改善
                improvement_threshold = self.best + self.threshold

        if (self.mode == 'min' and metric < improvement_threshold) or \
                (self.mode == 'max' and metric > improvement_threshold):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            # 如果超过容忍次数，则降低学习率
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0


class EarlyStopping:
    """早停机制，用于监测验证集指标，连续多轮无改善则触发早停"""

    def __init__(self, patience=10, delta=1e-4, mode='min', verbose=False, save=True,
                 path='best_model.tp'):
        self.patience = patience  # 容忍连续无改善的轮次
        self.delta = delta  # 指标改善的最小阈值（避免微小波动被判定为改善）
        self.mode = mode  # min：指标越小越好（如损失）；max：指标越大越好（如准确率）
        self.verbose = verbose  # 是否打印早停相关日志

        self.save = save  # 是否保存性能最优的模型
        self.path = path  # 最优模型保存路径

        self.best_score = None  # 记录历史最优指标值
        self.num_bad_epochs = 0  # 连续无改善的轮次计数
        self.early_stop = False  # 是否触发早停的标志

    def __call__(self, val_metric, model=None):
        """
        每轮验证后调用，判断是否触发早停
        val_metric: 当前轮次的验证集指标（如val_loss、val_acc）
        model: 当前训练的模型实例（需支持state_dict()方法，仅当save_best_model=True时需传入）
        """
        # 1. 计算当前指标对应的得分（统一转为最小化逻辑，方便比较）
        current_score = -val_metric if self.mode == 'min' else val_metric

        # 2. 初始化历史最优得分（第一轮调用时）
        if self.best_score is None:
            self.best_score = current_score
            return

        # 3. 判断当前指标是否有效改善：当前得分 > 历史最优得分 + delta（delta避免微小波动）
        if current_score > self.best_score + self.delta:
            self.best_score = current_score  # 更新历史最优得分
            self._save_best_model(val_metric, model)  # 保存新的最优模型
            self.num_bad_epochs = 0  # 重置连续无改善计数
        else:
            self.num_bad_epochs += 1  # 累加连续无改善计数
            if self.verbose:
                print(
                    f"EarlyStopping: consecutive {self.num_bad_epochs} epoches has no improvement(current: {val_metric:.6f})")

            if self.num_bad_epochs >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarlyStopping: consecutive {self.patience} epoches has no improvement, early stop!")
                    score = (-self.best_score) if self.mode == 'min' else self.best_score
                    print(f"EarlyStopping: best model's metric on validation set: {score:.6f}")

    def _save_best_model(self, val_metric, model):
        """保存最优模型参数（仅当开启保存功能且传入模型时）"""
        if self.save and model is not None:
            model.save(self.path)
            if self.verbose:
                print(f"EarlyStopping: find a better model(metric: {val_metric:.6f}), save to {self.path}")


class DataLoader:
    """数据加载器，支持批处理、打乱和自定义转换"""

    def __init__(self, data, batch_size: int = 64, shuffle: bool = True, transform=None):
        """
        :param data: 数据集，格式为[(输入特征, 标签), ...]
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据集
        :param transform: 数据转换函数，格式为func(input, label) -> (transformed_input, transformed_label)
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.indices = list(range(len(data)))
        self.cursor = 0  # 当前批次指针
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        """迭代器初始化"""
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """获取下一个批次"""
        if self.cursor >= len(self.data):
            raise StopIteration
        # 计算当前批次索引范围
        end = min(self.cursor + self.batch_size, len(self.data))
        batch_indices = self.indices[self.cursor:end]
        self.cursor = end

        batch_inputs = []
        batch_labels = []
        for idx in batch_indices:
            x, y = self.data[idx]
            if self.transform:
                x, y = self.transform(x, y)
            batch_inputs.append(Tensor(x))
            batch_labels.append(Tensor(y))
        return batch_inputs, batch_labels

    def __len__(self):
        """返回批次数量"""
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# 训练函数
def train_on_batch(model, batch, optimizer, loss_fn=Tensor.mse):
    """
    训练一个批次并返回损失和准确率
    :param model: 模型实例
    :param batch: 批次数据 (inputs, labels)
    :param optimizer: 优化器
    :param loss_fn: 损失函数 (output, label)
    :return: (loss, accuracy)
    """
    inputs, labels = batch
    outputs = [model(i) for i in inputs]

    sample_losses = [loss_fn(out, label) for out, label in zip(outputs, labels)]
    loss = Tensor.mean(sample_losses)

    correct = 0
    total = len(outputs)
    for out, label in zip(outputs, labels):
        # 二分类阈值判断
        pred = 1 if out.vector.data[0] > 0.5 else 0
        if pred == label.vector.data[0]:
            correct += 1
    accuracy = correct / total

    loss.backward()
    optimizer.step()

    return loss.vector.data[0], accuracy


def valid_on_batch(model, batch, loss_fn=Tensor.mse):
    """
    验证一个批次并返回损失和准确率
    :param model: 模型实例
    :param batch: 批次数据 (inputs, labels)
    :param loss_fn: 损失函数 (output, label)
    :return: (loss, accuracy)
    """
    inputs, labels = batch
    outputs = [model(i) for i in inputs]
    sample_losses = [loss_fn(out, label) for out, label in zip(outputs, labels)]
    loss = Tensor.mean(sample_losses)

    correct = 0
    total = len(outputs)
    for out, label in zip(outputs, labels):
        # 二分类阈值判断
        pred = 1 if out.vector.data[0] > 0.5 else 0
        if pred == label.vector.data[0]:
            correct += 1
    accuracy = correct / total

    return loss.vector.data[0], accuracy


# 各种初始化方法
def my_init(size) -> Tensor:
    """对单个张量初始化权重"""
    std = np.sqrt(2.0 / size)
    return Tensor(np.random.normal(0, std, size))


def xavier_init(inp_size: int, out_size: int) -> Tensor:
    """Xavier初始化 - 适用于tanh/sigmoid等激活函数"""
    std = np.sqrt(2.0 / (inp_size + out_size))
    return Tensor(np.random.normal(0, std, (out_size, inp_size)))


def he_init(inp_size: int, out_size: int) -> Tensor:
    """He初始化 - 适用于ReLU及其变体激活函数"""
    std = np.sqrt(2.0 / inp_size)
    return Tensor(np.random.normal(0, std, (out_size, inp_size)))


def uniform_init(size, a=-0.05, b=0.05) -> Tensor:
    """均匀分布初始化 - 适用于线性层"""
    return Tensor(np.random.uniform(a, b, size))


# 各种工具函数
def deriv(func, x: Tensor, eps=1e-4):
    """函数的数值微分计算，测试用"""
    x1 = Tensor(x.vector.data + eps)
    x2 = Tensor(x.vector.data - eps)
    y1 = func(x1)
    y2 = func(x2)
    return (y1.vector.data - y2.vector.data) / (2 * eps)