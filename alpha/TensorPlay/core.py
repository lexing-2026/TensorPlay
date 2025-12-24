import warnings
import contextlib
import weakref
import numpy as np
from alpha import TensorPlay as tp
from typing import List, Union, Tuple, Optional, Any, Callable, Generator
warnings.filterwarnings("default", category=UserWarning)

# =============================================================================
# Config
# =============================================================================
class Config:
    precision = np.float32
    enable_grad = True
    training = True


@contextlib.contextmanager
def config(name: str, value: bool) -> Generator:
    """配置是否开启梯度计算的上下文管理器"""
    prev_mode = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, prev_mode)


def no_grad() -> contextlib._GeneratorContextManager:
    """上下文管理器，禁用梯度计算"""
    return config("enable_grad", False)

# =============================================================================
# Tensor
# =============================================================================
def to_data(data: Union[np.ndarray, tuple, list, int, float]) -> np.ndarray:
    """将数据转换为张量要求格式"""
    if isinstance(data, (tuple, list, int, float)):
        return np.array(data, dtype=Config.precision)
    if isinstance(data, np.ndarray):
        return data.astype(Config.precision)
    else:
        raise TypeError(f"Data must be a numpy array, tuple, list, int, or float (not {type(data).__name__})")


class Tensor:
    """张量：(B, H, W, C)"""

    __add__: Callable[..., 'Tensor']
    __radd__: Callable[..., 'Tensor']
    __sub__: Callable[..., 'Tensor']
    __rsub__: Callable[..., 'Tensor']
    __mul__: Callable[..., 'Tensor']
    __rmul__: Callable[..., 'Tensor']
    __truediv__: Callable[..., 'Tensor']
    __rtruediv__: Callable[..., 'Tensor']
    __pow__: Callable[..., 'Tensor']
    __rpow__: Callable[..., 'Tensor']
    __matmul__: Callable[..., 'Tensor']
    __getitem__: Callable[..., 'Tensor']
    reslice: Callable[..., 'Tensor']
    sum: Callable[..., 'Tensor']
    max: Callable[..., 'Tensor']
    min: Callable[..., 'Tensor']
    exp: Callable[..., 'Tensor']
    log: Callable[..., 'Tensor']
    mean: Callable[..., 'Tensor']
    relu: Callable[..., 'Tensor']
    leaky_relu: Callable[..., 'Tensor']
    gelu: Callable[..., 'Tensor']
    tanh: Callable[..., 'Tensor']
    softmax: Callable[..., 'Tensor']
    log_softmax: Callable[..., 'Tensor']
    sigmoid: Callable[..., 'Tensor']
    reshape: Callable[..., 'Tensor']
    clip: Callable[..., 'Tensor']
    dropout: Callable[..., 'Tensor']
    expand: Callable[..., 'Tensor']
    flatten: Callable[..., 'Tensor']
    transpose: Callable[..., 'Tensor']
    broadcast: Callable[..., 'Tensor']
    rebroadcast: Callable[..., 'Tensor']

    T: 'Tensor'

    def __init__(self, data: Union[np.ndarray, list, int, float], op: 'Operator' = None, name: str = None):
        self.data = to_data(data)
        self.grad = None
        self.op = op
        self.name = name
        self.rank = 0
        self.source_module = None  # 用于钩子机制

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def clone(self) -> 'Tensor':
        """返回当前张量的副本，确保梯度独立"""
        cloned_tensor = Tensor(self.data, op=self.op)
        if self.grad is not None:
            cloned_tensor.grad = Tensor(self.grad.data.copy())
        return cloned_tensor

    def detach(self) -> 'Tensor':
        """返回一个不追踪梯度的张量副本"""
        detached = Tensor(self.data)
        return detached

    def zero_grad(self) -> None:
        """清空梯度"""
        self.grad = None

    def one_grad(self) -> None:
        """将梯度设为1"""
        self.grad = Tensor(np.ones(self.shape))

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]]) -> 'Tensor':
        """创建一个指定形状的全0张量"""
        return Tensor(np.zeros(shape))

    def backward(self, clean: bool = True, retain_grad: bool = True, higher_grad: bool = False) -> None:
        """
        计算子图的反向传播
        :param clean: bool 是否清理计算图
        :param retain_grad: bool 是否保留下游梯度
        :param higher_grad: bool 是否支持多阶梯度
        """
        Operator.state = False
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
            with config('enable_grad', higher_grad):
                grads = op.propagate_grad()
                for i, grad in enumerate(grads):
                    if op.inp[i].grad is None:
                        op.inp[i].grad = grad
                    else:
                        op.inp[i].grad = op.inp[i].grad + grad
            if not retain_grad:
                op.out().grad = None
        Operator.state = True
        if clean:
            if higher_grad:
                # 保留反向计算图
                Operator.clean(specific_ops=op_list)
            else:
                Operator.clean()

# =============================================================================
# Operator
# =============================================================================
class Operator:
    """算子基类"""
    compute_list: List['Operator'] = []  # 记录计算顺序
    state: bool = True  # 是否在前向状态

    def __init__(self):
        """子类根据需要重写，没有额外参数不写"""
        if Config.enable_grad:
            self.compute_list.append(self)
        self.inp = None  # 输入张量
        self.out = None  # 输出张量
        self.rank = 0

    def __repr__(self) -> str:
        return f"Operator.{self.__class__.__name__}"

    def __call__(self, *args):
        """前向调用接口"""
        inputs = [inp if isinstance(inp, (Tensor, type(None))) else Tensor(inp) for inp in args]
        datas = [inp.data if inp is not None else None for inp in inputs]
        out = self._forward(*datas)
        if Config.enable_grad:
            self.inp = inputs
            self.out = weakref.ref(out)
            self.rank = max([inp.rank for inp in inputs])
            out.op = self
            out.rank = self.rank + 1
        return out

    def propagate_grad(self) -> Union[Tensor, List[Tensor]]:
        """后向调用接口，集成反向钩子调用"""
        if not Config.enable_grad and self.state:
            warnings.warn('Attention: forward() run with no grad...\n'
                          'If you are not computing higher-gradients, '
                          'please examine your code.', UserWarning, stacklevel=2)
        g = self._backward()
        # 调用反向钩子
        if self.out().source_module is not None:
            module = self.out().source_module
            module._call_backward_hooks(self.out(), self.inp)
        return g

    def _forward(self, *args: Any) -> Tensor:
        """前向具体运算"""
        raise NotImplementedError

    def _backward(self) -> Any:
        """后向具体计算"""
        raise NotImplementedError

    @classmethod
    def clean(cls, specific_ops: Optional[List['Operator']] = None) -> None:
        """清理计算图数据"""
        if specific_ops is not None:
            while specific_ops:
                ops = specific_ops.pop()
                ops.out().op = None
                cls.compute_list.remove(ops)
        else:
            while cls.compute_list:
                ops = cls.compute_list.pop()
                if ops.out() is not None:
                    ops.out().op = None
            cls.compute_list.clear()

# =============================================================================
# Layer
# =============================================================================
class Layer:
    """
    基础参数层，实现钩子功能，所有参数层都需要继承此类
    save和load方法自定义格式，必须互认
    """
    layer_list: List['Layer'] = []  # 基础参数层全局记录，兼容最底层实现

    def __init__(self, *args):
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_hooks = {}
        # 基础参数层只记录Layer类，Module以上不记录
        if isinstance(self, tp.Module):
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

# =============================================================================
# Optimizer
# =============================================================================
class Optimizer:
    """优化器类，储存参数需要重写接口函数save()和load()"""
    hooks = {}  # 用于存储钩子函数的字典，键为钩子函数的ID，值为钩子函数

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
        for hook in self.hooks:
            self.params = hook(self.params)
        self._step()

    def _step(self):
        """具体优化方法，必须重写"""
        raise NotImplementedError

    def zero_grad(self):
        """使参数的梯度归零"""
        for i in self.params:
            i.zero_grad()

    def register_hook(self, hook: Callable) -> int:
        """注册钩子函数"""
        self.hooks[id(hook)] = hook
        return id(hook)

    def remove_hook(self, handle: int):
        """移除指定的钩子函数"""
        if handle in self.hooks:
            del self.hooks[handle]