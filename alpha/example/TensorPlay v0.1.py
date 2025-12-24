import ast
import math
import json
import random


class Vector(list):
    """向量，可以进行按位加减乘等操作"""

    def __init__(self, data):
        # 检查每个元素是否为数字（int或float）
        for item in data:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Vector elements must be numbers (not {type(item).__name__})")
        super().__init__(data)

    def __repr__(self):
        return f"Vector({list(self)})"

    def __str__(self):
        return str(list(self))

    def __neg__(self):
        return Vector([-i for i in self])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            # 向量 + 标量：每个元素加标量
            return Vector([x + other for x in self])
        elif isinstance(other, Vector):
            # 向量 + 向量：对应元素相加
            if len(self) != len(other):
                raise ValueError(f"Cannot add vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i + j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            # 向量 + 标量：每个元素加标量
            for i in range(len(self)):
                self[i] += other
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"Cannot add vectors of different lengths ({len(self)} vs {len(other)})")
            for i in range(len(self)):
                self[i] += other[i]
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        # 向量×标量：每个元素乘以标量
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self])
        # 向量×向量：对应元素相乘
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"Cannot multiply vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i * j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            # 向量 * 标量：每个元素乘以标量
            for i in range(len(self)):
                self[i] *= other
        elif isinstance(other, Vector):
            # 向量 * 向量：对应元素相乘
            if len(self) != len(other):
                raise ValueError(f"Cannot multiply vectors of different lengths ({len(self)} vs {len(other)})")
            for i in range(len(self)):
                self[i] *= other[i]
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 向量 / 标量：每个元素除以标量
            return Vector([x / other for x in self])
        elif isinstance(other, Vector):
            # 向量 / 向量：对应元素相除
            if len(self) != len(other):
                raise ValueError(f"Cannot divide vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i / j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector' and '{type(other).__name__}'")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # 标量 / 向量：每个元素除以标量
            return Vector([other / x for x in self])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'Vector'")

    def __pow__(self, power: (int, float)):
        if isinstance(power, (int, float)):
            return Vector([i ** power for i in self])
        else:
            raise TypeError(f"Cannot power Vector with {type(power).__name__}")

    def __rpow__(self, base: (int, float)):
        if isinstance(base, (int, float)):
            return Vector([base ** x for x in self])
        else:
            raise TypeError(f"Cannot power {type(base).__name__} with Vector")

    @classmethod
    def zeros(cls, size):
        return Vector([0] * size)


class Tensor:
    """张量，带梯度的向量，梯度是另一个同维向量"""

    def __init__(self, data, op=None):
        self.data = Vector(data)
        self.grad = Vector.zeros(len(data))
        self.op = op

    def __repr__(self):
        return f"Tensor({list(self.data)}, grad={list(self.grad)})"

    def __len__(self):
        return len(self.data)

    # 重载运算方法，通过运算符类实现（关联计算图）
    def __add__(self, other):
        op = Add()
        return op.compute(self, other)

    def __sub__(self, other):
        op = Sub()
        return op.compute(self, other)

    def __neg__(self):
        op = Sub()
        return op.compute(Tensor.zeros(len(self)), self)

    def __mul__(self, other):
        op = Mul()
        return op.compute(self, other)

    def __truediv__(self, other):
        op = Div()
        return op.compute(self, other)

    def __pow__(self, power: (int, float)):
        op = Pow()
        return op.compute(self, power)

    def sum(self):
        op = Sum()
        return op.compute(self)

    def relu(self):
        op = Relu()
        return op.compute(self)

    def exp(self):
        op = Exp()
        return op.compute(self)

    def log(self):
        op = Log()
        return op.compute(self)

    def sigmoid(self):
        op = Sigmoid()
        return op.compute(self)

    def tanh(self):
        op = Tanh()
        return op.compute(self)

    def cut(self, start: int, end: int):
        op = Cut()
        return op.compute(self, start, end)

    @classmethod
    def mean(cls, x: list['Tensor']):
        op = Mean()
        return op.compute(x)

    @classmethod
    def connect(cls, x: list['Tensor']):
        op = Connect()
        return op.compute(x)

    def clone(self):
        """返回当前张量的副本，确保梯度独立"""
        cloned_tensor = Tensor(self.data, op=self.op)
        cloned_tensor.grad = Vector(self.grad)
        return cloned_tensor

    def detach(self):
        """返回一个不追踪梯度的张量副本"""
        detached = Tensor(self.data)
        return detached

    def repeat(self, times: int):
        """成倍扩展张量维度"""
        out = Tensor.connect([self.clone() for _ in range(times)])
        out.zero_grad()
        return out

    # 复用运算符实现繁琐的激活函数，自动构建计算图
    def softmax(self):
        """softmax激活函数：softmax(x) = e^x / sum(e^x_j)"""
        exp_tensor = self.exp()
        sum_exp = exp_tensor.sum()  # 返回单标量张量
        inv_sum_exp = sum_exp ** -1 if abs(sum_exp.data[0]) > 1e-5 else (sum_exp + Tensor([0.1])) ** -1
        return exp_tensor * inv_sum_exp.repeat(len(self))

    def gelu(self):
        """GELU激活函数：GELU(x) ≈ x * Sigmoid(1.702x)"""
        return self * (self * 1.702).sigmoid()

    @classmethod
    def mse(cls, a: 'Tensor', b: 'Tensor'):
        """均方误差（Mean Squared Error）：MSE = (1/n) * sum((a - b)²)"""
        if len(a) != len(b):
            raise ValueError("MSE can only be calculated between tensors of the same length")
        return ((a - b) ** 2).sum() / Tensor([len(a)])

    @classmethod
    def sse(cls, a: 'Tensor', b: 'Tensor'):
        """平方误差（Sum of Squared Error）：SSE = sum((a - b)²)"""
        if len(a) != len(b):
            raise ValueError("SSE can only be calculated between tensors of the same length")
        return ((a - b) ** 2).sum()

    @classmethod
    def nll(cls, out: 'Tensor', target: 'Tensor'):
        """交叉熵误差（Negative Log Likelihood）：NLL = -sum(target * log(output))"""
        return -(target * out.log()).sum()

    def zero_grad(self):
        """清空梯度"""
        self.grad = Vector.zeros(len(self))

    def one_grad(self):
        """将梯度设为1"""
        self.grad = Vector([1] * len(self))

    @classmethod
    def zeros(cls, size: int):
        """创建一个长度为size的全0张量"""
        return Tensor([0] * size)

    def backward(self, clean=True):
        """
        计算子图的反向传播
        :param clean: bool 是否清零计算图
        """
        self.one_grad()
        op_set = set()
        queue = []
        if self.op is not None:
            queue.append(self.op)
            op_set.add(self.op)

        while queue:
            current_op = queue.pop(0)  # 取出队首元素
            if current_op.inp is None:
                continue
            # 处理输入为列表或单个张量的情况
            inputs = current_op.inp if isinstance(current_op.inp, list) else [current_op.inp]
            for inp_tensor in inputs:
                if inp_tensor.op is not None and inp_tensor.op not in op_set:
                    op_set.add(inp_tensor.op)
                    queue.append(inp_tensor.op)

        # 按计算顺序排序搜索到的子图
        op_list = sorted(op_set, key=lambda x: Operator.compute_list.index(x), reverse=True)
        for op in op_list:
            if op.inp is None:
                continue
            op.propagate_grad()
        if clean:
            Operator.clean(specific_ops=op_set)


class Operator:
    """运算符类，所有对张量操作的运算符都需要继承此类，并重写compute和propagate_grad方法"""
    compute_list = []  # 记录计算顺序，用于反向传播
    enable_grad = True  # 控制是否追踪梯度

    def __init__(self):
        """子类根据需要重写，没有额外参数不写"""
        if self.enable_grad:
            self.compute_list.append(self)
        self.inp = None  # 输入张量
        self.out = None  # 输出张量

    def __repr__(self):
        return f"TensorOperator.{self.__class__.__name__}"

    def compute(self, *args):
        """
        子类需实现具体运算
        :return: Tensor
        """
        raise NotImplementedError

    def _compute_grad(self):
        """梯度具体计算，输入张量积累梯度"""
        raise NotImplementedError

    def propagate_grad(self):
        """梯度反向传播外部调用接口，集成反向钩子调用"""
        if not self.enable_grad:
            raise RuntimeError(
                "Cannot call propagate_grad() when tracing gradient computation graph is disabled. "
                "Enable Operator() using set_grad_enabled(True)."
            )
        self._compute_grad()
        if hasattr(self.out, '_source_module'):
            module = self.out._source_module
            module._call_backward_hooks(self.out, self.inp)

    @classmethod
    def set_grad_enabled(cls, mode):
        """控制是否开启梯度追踪"""
        cls.enable_grad = mode

    @classmethod
    def backward(cls, last1grad=True):
        """
        全局反向传播，计算所有相关张量的梯度
        :param last1grad: 是否将计算图中最后一个张量的梯度设为1（反向传播起点）
        :return: None
        """
        if not cls.compute_list:  # 无运算记录时直接返回
            return

        reversed_ops = list(reversed(cls.compute_list))
        if last1grad:
            reversed_ops[0].out.one_grad()

        # 逆序遍历运算执行梯度反向传播
        for op in reversed_ops:
            if op.inp is None:  # 跳过已清理的节点
                continue
            op.propagate_grad()

        cls.clean()

    @classmethod
    def clean(cls, specific_ops=None):
        """
        清理计算图数据
        :param specific_ops: 可选，指定需要清理的运算符集合，默认清理全部
        """
        if specific_ops is not None:
            for op in specific_ops:
                if op in cls.compute_list:
                    op.inp = None
                    op.out = None
                    cls.compute_list.remove(op)
        else:
            for op in cls.compute_list:
                op.inp = None
                op.out = None
            cls.compute_list.clear()


class no_grad:
    """运算符的上下文管理器，用于临时禁用梯度计算"""

    def __enter__(self):
        # 保存当前的梯度启用状态
        self.prev_mode = Operator.enable_grad
        # 禁用梯度计算
        Operator.set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复之前的梯度启用状态
        Operator.set_grad_enabled(self.prev_mode)
        # 不抑制任何异常
        return False


class Add(Operator):
    """加法运算符"""

    def compute(self, a: Tensor, b: Tensor):
        out = Tensor(a.data + b.data, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = out
        return out

    def _compute_grad(self):
        # 加法梯度反向传播：输入梯度+=输出梯度
        self.inp[0].grad += self.out.grad
        self.inp[1].grad += self.out.grad


class Sub(Operator):
    """减法运算符"""

    def compute(self, a: Tensor, b: Tensor):
        out = Tensor(a.data - b.data, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = out
        return out

    def _compute_grad(self):
        # 减法梯度反向传播：输入梯度+=输出梯度的正负数
        self.inp[0].grad += self.out.grad
        self.inp[1].grad -= self.out.grad


class Mul(Operator):
    """乘法运算符"""

    def compute(self, a: Tensor, b: Tensor):
        out = Tensor(a.data * b.data, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = out
        return out

    def _compute_grad(self):
        # 乘法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += self.inp[1].data * self.out.grad
        self.inp[1].grad += self.inp[0].data * self.out.grad


class Div(Operator):
    """除法运算符"""

    def compute(self, a: Tensor, b: Tensor):
        out = Tensor(a.data / b.data, op=self)
        if self.enable_grad:
            self.inp = [a, b]
            self.out = out
        return out

    def _compute_grad(self):
        # 除法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += (self.inp[1].data ** -1) * self.out.grad
        self.inp[1].grad += (-self.inp[0].data * self.inp[1].data ** -2) * self.out.grad


class Pow(Operator):
    """幂函数"""

    def __init__(self):
        super().__init__()
        self.power = None

    def compute(self, a: Tensor, power: float):
        out = Tensor(a.data ** power, op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
            self.power = power
        return out

    def _compute_grad(self):
        # 输入梯度 += n * x ^ (n - 1) * 输出梯度
        self.inp.grad += self.power * self.inp.data ** (self.power - 1) * self.out.grad


class Exp(Operator):
    """自然指数函数"""

    def compute(self, a: Tensor):
        out = Tensor(math.e ** a.data, op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入梯度 += e ^ x * 输出梯度 = 输出值 * 输出梯度
        self.inp.grad += self.out.data * self.out.grad


class Log(Operator):
    """自然对数函数"""

    def compute(self, a: Tensor):
        out = Tensor([math.log(i) if i > 0 else 1e-10 for i in a.data], op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入梯度 += (1 / x) * 输出梯度 = 输出梯度 * (输入值的倒数)
        self.inp.grad += self.out.grad * self.inp.data ** -1


class Sum(Operator):
    """元素级求和运算符"""

    def compute(self, a: Tensor):
        sum_value = sum(a.data)  # 返回标量
        out = Tensor([sum_value], op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入张量的每个分量梯度 += 输出张量的梯度（单元素）
        self.inp.grad += self.out.grad[0]


class Mean(Operator):
    """张量平均值算符"""

    def compute(self, x: list[Tensor]):
        if len(x) == 0:
            raise ValueError("Input tensor list is empty!")
        n = len(x[0].data)
        mean_data = Vector.zeros(n)
        for tensor in x:
            if len(tensor.data) != n:
                raise ValueError("All input tensors must have the same shape!")
            mean_data += tensor.data
        out = Tensor(mean_data / n, op=self)
        if self.enable_grad:
            self.inp = x
            self.out = out
        return out

    def _compute_grad(self):
        # 平均值梯度反向传播：每个输入张量的梯度 += 输出梯度 / 输入张量数量
        for tensor in self.inp:
            tensor.grad += self.out.grad / len(self.inp)


class Relu(Operator):
    """ReLU修正线性单元"""

    def compute(self, a: Tensor):
        out = Tensor([max(i, 0) for i in a.data], op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入梯度+=输出梯度*输入函数（大于等于0）
        for i in range(len(self.inp)):
            if self.inp.data[i] >= 0:
                self.inp.grad[i] += self.out.grad[i]


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def compute(self, a: Tensor):
        out = Tensor(1 / (1 + math.e ** -a.data), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入梯度 += σ(x) * (1 - σ(x)) * 输出梯度
        self.inp.grad += self.out.data * (1 - self.out.data) * self.out.grad


class Tanh(Operator):
    """Tanh激活函数"""

    def compute(self, a: Tensor):
        exp_p = math.e ** a.data
        exp_n = math.e ** -a.data
        out = Tensor((exp_p - exp_n) / (exp_p + exp_n), op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
        return out

    def _compute_grad(self):
        # 输入梯度 += (1 - tanh²(x)) * 输出梯度
        self.inp.grad += (1 - self.out.data ** 2) * self.out.grad


class Connect(Operator):
    """拼接张量"""

    def compute(self, x: list[Tensor]):
        out = Tensor([], op=self)
        for tensor in x:
            out.data.extend(tensor.data)
            out.grad.extend(tensor.grad)
        if self.enable_grad:
            self.inp = x
            self.out = out
        return out

    def _compute_grad(self):
        # 将输出梯度按原拼接顺序分配给每个输入张量
        seg = self.out.grad
        for tensor in self.inp:
            tensor.grad += Vector(seg[:len(tensor)])
            seg = seg[len(tensor):]


class Cut(Operator):
    """切分张量"""

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

    def compute(self, a: Tensor, start: int, end: int):
        out = Tensor(a.data[start:end], op=self)
        if self.enable_grad:
            self.inp = a
            self.out = out
            self.start = start
            self.end = end
        return out

    def _compute_grad(self):
        # 将输出张量的梯度分配到输入张量的对应索引位置
        for i in range(len(self.out)):
            self.inp.grad[self.start + i] += self.out.grad[i]


class DenseOp(Operator):
    """线性层运算符"""

    def __init__(self):
        super().__init__()
        self.matrix = None

    def compute(self, matrix, x: Tensor):
        if matrix.bias:
            out = [sum(matrix.w[i].data * x.data) + matrix.b[i].data[0] for i in range(len(matrix.w))]
        else:
            out = [sum(matrix.w[i].data * x.data) for i in range(len(matrix.w))]
        out = Tensor(out, op=self)
        if self.enable_grad:
            self.inp = x
            self.out = out
            self.matrix = matrix
        return out

    def _compute_grad(self):
        """
        线性层梯度反向传播：
        假设前向计算：out[i] = sum(w[i][j] * x[j]) + b[i]（j为输入维度）
        - w[i][j]的梯度：dLoss/dw[i][j] = dLoss/d_out[i] * x[j]
        - b[i]的梯度：dLoss/db[i] = dLoss/d_out[i]（b[i]仅影响out[i]）
        - x[j]的梯度：dLoss/dx[j] = sum(dLoss/d_out[i] * w[i][j])（i为输出维度）
        """
        for i in range(len(self.matrix.w)):
            self.matrix.w[i].grad += self.out.grad[i] * self.inp.data
            self.inp.grad += self.out.grad[i] * self.matrix.w[i].data
            if self.matrix.bias:
                self.matrix.b[i].grad += self.out.grad[i]


class Layer:
    """
    基础参数层，实现钩子功能，所有参数层都需要继承此类
    save和load方法自定义格式，必须互认
    """
    layer_list = []  # 基础参数层全局记录，兼容最底层实现

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

    def __repr__(self):
        prefix = '' if self.__class__.__name__ == 'Layer' else 'Layer.'
        return f"{prefix}{self.__class__.__name__}"

    def save(self, *args):
        """
        保存接口，所有继承了Layer的类需重写此方法
        :return: 自定义格式，与load方法互认
        """
        raise NotImplementedError

    def load(self, *args):
        """
        读取接口，所有继承了Layer的类需重写此方法
        :param args: str 自定义格式，与save方法互认
        """
        raise NotImplementedError

    def param(self):
        """
        参数接口，所有继承了Layer的类需重写此方法
        :return: list[Tensor]
        """
        raise NotImplementedError

    def grad_descent_zero(self, lr: float):
        """
        [DEPRECATED] 此方法已淘汰，建议使用优化器
        警告：该方法将在下次版本中移除，使用时需注意兼容性。
        """
        pass

    @classmethod
    def get_params(cls):
        """
        返回所有基础参数层所有参数，兼容优化器的默认设置
        :return: list[Tensor]
        """
        params = []
        for i in Layer.layer_list:
            # Param返回列表
            if i.param() is not None:
                params += i.param()
        return params

    def register_forward_pre_hook(self, hook):
        """注册前向传播前的钩子"""
        handle = id(hook)
        self._forward_pre_hooks[handle] = hook
        return handle

    def register_forward_hook(self, hook):
        """注册前向传播后的钩子"""
        handle = id(hook)
        self._forward_hooks[handle] = hook
        return handle

    def register_backward_hook(self, hook):
        """注册反向传播的钩子"""
        handle = id(hook)
        self._backward_hooks[handle] = hook
        return handle

    def remove_hook(self, handle):
        """移除指定钩子"""
        for hooks in [self._forward_pre_hooks, self._forward_hooks, self._backward_hooks]:
            if handle in hooks:
                del hooks[handle]
                return

    def _call_forward_pre_hooks(self, *args, **kwargs):
        """调用前向传播前的钩子"""
        for hook in self._forward_pre_hooks.values():
            hook(self, args, kwargs)

    def _call_forward_hooks(self, *args, **kwargs):
        """调用前向传播后的钩子"""
        for hook in self._forward_hooks.values():
            hook(self, args, kwargs, self._forward_result)

    def _call_backward_hooks(self, grad_outputs: Vector, inputs):
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

    def __call__(self, *args, **kwargs):
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

    def forward(self, *args, **kwargs):
        """前向传播方法，需要子类实现"""
        raise NotImplementedError(f"Module {self.__class__.__name__} has no forward method implemented")


class ConstantTensor(Tensor, Layer):
    """单张量参数层"""

    def __init__(self, data):
        Tensor.__init__(self, data)
        Layer.__init__(self)

    def save(self):
        text = str(self.data)
        return text

    def load(self, text: str):
        self.data = Vector(ast.literal_eval(text))

    def param(self):
        return [self]


class Norm(Layer):
    """标准化处理层，带可学习参数"""

    def __init__(self):
        self.w = Tensor([random.gauss(0, 0.04)])
        self.b = Tensor([random.gauss(0, 0.04)])
        super().__init__()

    def forward(self, x: Tensor, eps=0.0001):
        n = Tensor([len(x)])
        mean = (x.sum() / n).repeat(len(x))
        sigma = (((x - mean) ** 2).sum() / n) ** 0.5
        std = (x - mean) / (Tensor([eps]) + sigma).repeat(len(x))
        out = self.w.repeat(len(x)) * std + self.b.repeat(len(x))
        return out

    def grad_descent_zero(self, lr):
        self.w.data -= lr * self.w.grad
        self.b.data -= lr * self.b.grad
        self.w.zero_grad()
        self.b.zero_grad()

    def save(self):
        return f"{self.w.data}/{self.b.data}"

    def load(self, text: str):
        parts = text.split('/')
        self.w = Tensor(ast.literal_eval(parts[0]))
        self.b = Tensor(ast.literal_eval(parts[1]))

    def param(self):
        return [self.w, self.b]


class Dense(Layer):
    """全连接层"""

    def __init__(self, inp_size: int, out_size: int, bias=True):
        """
        :param bias: bool 是否加上偏置
        """
        # w: list[Tensor] 权重矩阵, bias: list[Tensor] 偏置向量
        self.w = he_init(inp_size, out_size)
        self.bias = bias
        if bias:
            self.b = [Tensor.zeros(1) for _ in range(out_size)]
        super().__init__()

    def forward(self, a: Tensor, with_op=True):
        """
        前向传播运算
        :param with_op: bool 是否使用单独的dense运算符
        """
        if with_op:
            op = DenseOp()
            out = op.compute(self, a)
            return out
        else:
            # 复用更基础的算符
            if self.bias:
                out = [(self.w[i] * a).sum() + self.b[i] for i in range(len(self.w))]
            else:
                out = [(self.w[i] * a).sum() for i in range(len(self.w))]
            out = Tensor.connect(out)
            return out

    def grad_descent_zero(self, lr: float):
        """
        单个层的梯度下降，并清空梯度
        :param lr: float 学习率
        """
        for i in range(len(self.w)):
            self.w[i].data -= lr * self.w[i].grad
            self.b[i].data -= lr * self.b[i].grad
            self.w[i].zero_grad()
            self.b[i].zero_grad()

    def save(self):
        x = [str(i.data) for i in self.w]
        text = '[' + ', '.join(x) + ']'
        if self.bias:
            x = [str(i.data) for i in self.b]
            text += '/[' + ', '.join(x) + ']'
        return text

    def load(self, text: str):
        text = text.split('/')
        # w、b是可迭代对象
        w = ast.literal_eval(text[0])
        self.w = [Tensor(i) for i in w]
        if len(text) == 2:
            self.bias = True
            b = ast.literal_eval(text[1])
            self.b = [Tensor(i) for i in b]
        else:
            self.bias = False

    def param(self):
        return self.w + self.b


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
            raise ValueError(f"Model mismatched: expect {self.__class__.__name__}, got {data.get('type')}!")
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


class MultiConv:
    """多通道卷积层"""

    def __init__(self, in_channel: int, out_channel: int, width: int, height: int,
                 stride_w: int = 1, stride_h: int = 1, pad: bool = True, bias: bool = True):
        """
        :param in_channel: int 输入通道数
        :param out_channel: int 输出通道数
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        self.cores = [[Conv2D(width, height, stride_w, stride_h, pad=False, bias=bias) for _ in range(in_channel)]
                      for _ in range(out_channel)]
        self.pad = pad

    def __call__(self, x):
        """
        :param x: list[list[Tensor,Tensor...],list[Tensor,Tensor...]...] 多通道的二维张量，由列表包裹
        :return: list[list[Tensor,Tensor...],list[Tensor,Tensor...]...]
        """
        if self.pad:
            x = [self.cores[0][0].padding(i) for i in x]
        x2 = []
        for chan in self.cores:
            x_chan = sum_chan2d([chan[i](x[i]) for i in range(len(chan))])
            x2.append(x_chan)
        return x2

    def grad_descent_zero(self, lr):
        for chan in self.cores:
            for i in chan:
                i.grad_descent_zero(lr)


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


class MultiAtt:
    def __init__(self, head_num, emb_size, qk_size=None, v_size=None):
        """
        多头注意力模块
        :param head_num: int 注意力头数量
        :param emb_size: int 输入词向量维度
        :param qk_size: int q、k维度
        :param v_size: int 输出向量维度
        """
        self.heads = [Attention(emb_size, qk_size, v_size) for _ in range(head_num)]
        self.emb_size = emb_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        """
        :param x: list[Tensor,Tensor...]  装着(多个词的词向量)的列表
        :return: list[Tensor,Tensor...]
        """
        out = [h(x, mask_list, tri_mask) for h in self.heads]
        out = sum_chan2d(out)
        return out

    def grad_descent_zero(self, k):
        for i in self.heads:
            i.grad_descent_zero(k)


class Transformer:
    """自注意力模块"""

    def __init__(self, head_num, emb_size):
        self.att = MultiAtt(head_num, emb_size)
        self.f1 = Dense(emb_size, emb_size * 4)
        self.f2 = Dense(emb_size * 4, emb_size)
        self.n1 = Norm()
        self.n2 = Norm()

    def __call__(self, x, mask_list=None, tri_mask=False):
        x2 = x
        x = self.att(x, mask_list, tri_mask)
        x = [self.n1(i) for i in x]
        x = [x[i] + x2[i] for i in range(len(x))]

        x2 = x
        x = [self.f2(self.f1(i).relu()) for i in x]
        x = [self.n2(i) for i in x]
        x = [x[i] + x2[i] for i in range(len(x))]
        return x

    def grad_descent_zero(self, lr):
        self.att.grad_descent_zero(lr)
        self.f1.grad_descent_zero(lr)
        self.f2.grad_descent_zero(lr)
        self.n1.grad_descent_zero(lr)
        self.n2.grad_descent_zero(lr)


class MiniTransformer:
    """简单的Transformer模块"""

    def __init__(self, head_num, emb_size, window_size, lowrank=False):
        self.a = MultiAtt(head_num, emb_size)
        if lowrank:
            self.f1 = MiniDense(emb_size * window_size, emb_size * window_size)
            self.f2 = MiniDense(emb_size * window_size, emb_size * window_size)
        else:
            self.f1 = Dense(emb_size * window_size, emb_size * window_size)
            self.f2 = Dense(emb_size * window_size, emb_size * window_size)
        self.n1 = Norm()
        self.n2 = Norm()
        self.emb_size = emb_size
        self.window_size = window_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        x2 = x  # 二维张量
        x = Tensor.connect(self.a(x, mask_list, tri_mask))
        x = self.n1(x)
        x += Tensor.connect(x2)
        x2 = x  # 一维张量
        x = self.f2(self.f1(x).gelu())
        x = self.n2(x)
        x += x2
        x = resize2d(x, self.emb_size, self.window_size)
        return x

    def grad_descent_zero(self, k):
        self.a.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.n1.grad_descent_zero(k)
        self.n2.grad_descent_zero(k)


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
            ten.data -= self.lr * ten.grad
            ten.zero_grad()


class Momentum(Optimizer):
    """动量优化器"""

    def __init__(self, params=None, lr=0.001, gamma=0.8):
        super().__init__(params)
        self.momentum = [Vector.zeros(len(ten)) for ten in self.params]
        self.lr = lr
        self.gamma = gamma

    def step(self):
        for i, tensor in enumerate(self.params):
            self.momentum[i] = self.gamma * self.momentum[i] + (1 - self.gamma) * tensor.grad
            tensor.data -= self.momentum[i] * self.lr
            tensor.zero_grad()

    def save(self):
        return str(self.momentum)

    def load(self, text):
        self.momentum = Vector(ast.literal_eval(text))


class Adam(Optimizer):
    """Adaptive Moment Estimation（自适应矩估计）"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.m = [Vector.zeros(len(ten)) for ten in self.params]  # 一阶动量
        self.s = [Vector.zeros(len(ten)) for ten in self.params]  # 二阶动量
        self.times = 1  # 时间步
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新一阶动量
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            # 更新二阶动量
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad ** 2
            # 偏差修正
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            tensor.data -= (self.lr * cm) / (cs ** 0.5 + self.eps)
            tensor.zero_grad()
        self.times += 1

    def save(self):
        return str(self.m) + '/' + str(self.s) + '/' + str(self.times)

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])


class AdamW(Optimizer):
    """AdamW优化器，在Adam基础上改进了权重衰减"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4):
        super().__init__(params)
        self.m = [Vector.zeros(len(ten)) for ten in self.params]
        self.s = [Vector.zeros(len(ten)) for ten in self.params]
        self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay  # 权重衰减系数

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad ** 2
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # 引入权重衰减
            tensor.data -= self.lr * (cm / (cs ** 0.5 + self.eps) + self.weight_decay * tensor.data)
            tensor.zero_grad()
        self.times += 1

    def save(self):
        return f"{str(self.m)}/{str(self.s)}/{str(self.times)}/{str(self.weight_decay)}"

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])
        self.weight_decay = float(text[3])


class Nadam(Optimizer):
    """Nadam优化器，结合Nesterov动量和Adam"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.m = [Vector.zeros(len(ten)) for ten in self.params]
        self.s = [Vector.zeros(len(ten)) for ten in self.params]
        self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * (tensor.grad ** 2)
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            # Nadam更新：融入Nesterov动量
            tensor.data -= (self.lr * (self.b1 * cm + (1 - self.b1) * tensor.grad / (1 - self.b1 ** self.times)) /
                            (cs ** 0.5 + self.eps))
            tensor.zero_grad()

        self.times += 1

    def save(self):
        return f"{str(self.m)}/{str(self.s)}/{str(self.times)}"

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])


class Lookahead(Optimizer):
    """Lookahead优化器，使用主优化器和慢更新策略"""

    def __init__(self, params=None, base_optimizer=Adam, k=5, alpha=0.5, **kwargs):
        super().__init__(params)
        # 初始化基础优化器（Adam）
        self.base_optimizer = base_optimizer(params, **kwargs)
        # 慢权重（初始化为参数的副本）
        self.slow_weights = [Vector(ten.data) for ten in self.params]
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
                self.slow_weights[i] += self.alpha * (self.params[i].data - self.slow_weights[i])
                # 将慢权重复制回参数
                self.params[i].data = Vector(self.slow_weights[i])

    def save(self):
        base_state = self.base_optimizer.save()
        return f"{base_state}/{str(self.slow_weights)}/{str(self.k)}/{str(self.alpha)}/{str(self.step_counter)}"

    def load(self, text):
        text = text.split('/')
        # 前半部分是基础优化器的状态
        base_state = '/'.join(text[:-4])
        self.base_optimizer.load(base_state)
        self.slow_weights = [Vector(ast.literal_eval(part)) for part in text[-4:-3]]
        self.k = int(text[-3])
        self.alpha = float(text[-2])
        self.step_counter = int(text[-1])


class RMSprop(Optimizer):
    """Root Mean Square Propagation（RMSprop），基于梯度平方的移动平均"""

    def __init__(self, params=None, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.s = [Vector.zeros(len(ten)) for ten in self.params]  # 二阶动量
        self.alpha = alpha  # 衰减系数
        self.lr = lr
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新二阶动量：s = alpha*s + (1-alpha)*grad^2
            self.s[i] = self.alpha * self.s[i] + (1 - self.alpha) * (tensor.grad ** 2)
            # 参数更新：theta = theta - lr * grad / (sqrt(s) + eps)
            tensor.data -= self.lr * tensor.grad / (self.s[i] ** 0.5 + self.eps)
            tensor.zero_grad()

    def save(self):
        return f"{str(self.s)}/{str(self.alpha)}/{str(self.lr)}/{str(self.eps)}"

    def load(self, text):
        text = text.split('/')
        self.s = Vector(ast.literal_eval(text[0]))
        self.alpha = float(text[1])
        self.lr = float(text[2])
        self.eps = float(text[3])


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
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))


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
            random.shuffle(self.indices)

    def __iter__(self):
        """迭代器初始化"""
        self.cursor = 0
        if self.shuffle:
            random.shuffle(self.indices)
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
        pred = 1 if out.data[0] > 0.5 else 0
        if pred == label.data[0]:
            correct += 1
    accuracy = correct / total

    loss.backward()
    optimizer.step()

    return loss.data[0], accuracy


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
        pred = 1 if out.data[0] > 0.5 else 0
        if pred == label.data[0]:
            correct += 1
    accuracy = correct / total

    return loss.data[0], accuracy


# 各种初始化方法
def my_init(size):
    """对单个张量初始化权重"""
    sigma = (2 / size) ** 0.5
    return Tensor([random.gauss(0, sigma) for _ in range(size)])


def xavier_init(inp_size: int, out_size: int):
    """Xavier初始化 - 适用于tanh/sigmoid等激活函数"""
    sigma = (2 / (inp_size + out_size)) ** 0.5
    return [Tensor([random.gauss(0, sigma) for _ in range(inp_size)]) for _ in range(out_size)]


def he_init(inp_size: int, out_size: int):
    """He初始化 - 适用于ReLU及其变体激活函数"""
    sigma = (2 / inp_size) ** 0.5
    return [Tensor([random.gauss(0, sigma) for _ in range(inp_size)]) for _ in range(out_size)]


def uniform_init(size, a=-0.05, b=0.05):
    """均匀分布初始化 - 适用于线性层"""
    return Tensor([random.uniform(a, b) for _ in range(size)])


# 各种工具函数
def sum_chan2d(x):
    """
    对多通道的二维张量求和，变成单通道二维张量
    :param x: 多通道二维张量，结构为 [批次][行][通道张量]
              例如: [[tensor1, tensor2], [tensor3, tensor4]]
    :return: 单通道二维张量，结构为 [行][求和后的张量]
    """
    rows = len(x[0])
    batches = len(x)

    out = [tensor.clone() for tensor in x[0]]
    for batch_idx in range(1, batches):
        if len(x[batch_idx]) != rows:
            raise ValueError(f"The row of batches must be {rows}.")
        for row_idx in range(rows):
            out[row_idx] += x[batch_idx][row_idx]
    return out


def resize2d(x, cols, rows):
    """把一维张量转为二维张量"""
    x2d = []
    for i in range(rows):
        x2d.append(x.cut(i * cols, (i + 1) * cols))
    return x2d


def func2d(x, func):
    """对二维张量进行张量级函数操作"""
    return [func(i) for i in x]


def deriv(func, x: Tensor):
    """函数的数值微分计算"""
    x = func(x)
    x2 = Tensor(x.data)
    x2.data[0] += 0.001
    x2 = func(x2)
    return (x2.data[0] - x.data[0]) / 0.001