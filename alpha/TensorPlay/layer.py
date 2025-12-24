import json
import numpy as np
from typing import List, Union
from .core import Tensor, Layer, to_data
from .initializer import he_init
from .utils import _same_padding
from .operator import DenseOp, BatchNormOp, LayerNormOp, Conv2DOp, MaxPoolingOp, AveragePoolingOp

class ConstantTensor(Tensor, Layer):
    """单张量参数层"""

    def __init__(self, data: Union[list, np.ndarray]):
        Tensor.__init__(self, data)
        Layer.__init__(self)

    def save(self) -> str:
        return json.dumps(self.data.tolist())

    def load(self, text: str) -> None:
        self.data = to_data(json.loads(text))

    def param(self):
        return [self]


class Dense(Layer):
    """全连接层"""

    def __init__(self, inp_size: int, out_size: int, bias=True):
        self.w = he_init((out_size, inp_size))
        self.bias = bias
        if bias:
            self.b = Tensor.zeros((out_size, 1))
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return DenseOp(self)(a)

    def save(self) -> str:
        text = {'w': self.w.data.tolist()}
        if self.bias:
            text['b'] = self.b.data.tolist()
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


class BatchNorm(Layer):
    """批量归一化层"""
    def __init__(self, channels: int, eps: float = 1e-4, decay: float = 0.9):
        self.eps = eps
        self.decay = decay
        self.gamma = Tensor(np.ones(channels))
        self.beta = Tensor(np.zeros(channels))
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return BatchNormOp(self)(a)

    def save(self) -> str:
        text = {'gamma': self.gamma.data.tolist(), 'beta': self.beta.data.tolist(),
                'running_mean': self.running_mean.tolist(), 'running_var': self.running_var.tolist()}
        return json.dumps(text)

    def load(self, text: str) -> None:
        parts = json.loads(text)
        self.gamma = Tensor(parts['w'])
        self.beta = Tensor(parts['b'])
        self.running_mean = np.array(parts['running_mean'])
        self.running_var = np.array(parts['running_var'])

    def param(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class LayerNorm(Layer):
    """层归一化层"""

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.gamma = Tensor(1)
        self.beta = Tensor(0)
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormOp(self)(x)

    def save(self) -> str:
        text = {'gamma': self.gamma.data.tolist(), 'beta': self.beta.data.tolist()}
        return json.dumps(text)

    def load(self, text: str):
        parts = json.loads(text)
        self.gamma = Tensor(parts['gamma'])
        self.beta = Tensor(parts['beta'])

    def param(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Conv2D(Layer):
    """二维卷积层"""

    def __init__(self, in_channels: int, filters: int, kernel_size: int, strides=(1, 1), padding='valid',
                 bias: bool = True):
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        if padding == 'valid':
            self.padding = (0, 0)
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        # (KH, KW, in_channels, out_channels)
        self.w = he_init((*self.kernel_size, in_channels, filters))
        if bias:
            self.b = Tensor.zeros((1, filters))
        else:
            self.b = None
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == 'same':
            self.padding = _same_padding(x.shape[1], x.shape[2], self.kernel_size, self.strides)
        return Conv2DOp(self.strides, self.padding)(x, self.w, self.b)

    def save(self) -> str:
        text = {'w': self.w.data.tolist()}
        if self.b is not None:
            text['b'] = self.b.data.tolist()
        return json.dumps(text)

    def load(self, text: str) -> None:
        parts = json.loads(text)
        self.w = Tensor(parts['w'])
        if self.b is not None:
            self.b = Tensor(parts['b'])

    def param(self) -> List[Tensor]:
        return [self.w, self.b] if self.b is not None else [self.w]


class MaxPooling(Layer):
    """最大池化层"""

    def __init__(self, kernel_size: int, strides=(1, 1), padding='valid'):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        if padding == 'valid':
            self.padding = (0, 0)
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == 'same':
            self.padding = _same_padding(x.shape[1], x.shape[2], self.kernel_size, self.strides)
        return MaxPoolingOp(self.strides, self.padding)(x)

    def save(self) -> str:
        return 'None'

    def load(self, text: str):
        pass

    def param(self) -> None:
        return None


class AveragePooling(Layer):
    """最大池化层"""

    def __init__(self, kernel_size: int, strides=(1, 1), padding='valid'):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        if padding == 'valid':
            self.padding = (0, 0)
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if self.padding == 'same':
            self.padding = _same_padding(x.shape[1], x.shape[2], self.kernel_size, self.strides)
        return AveragePoolingOp(self.strides, self.padding)(x)

    def save(self) -> str:
        return 'None'

    def load(self, text: str):
        pass

    def param(self) -> None:
        return None


# 一些复杂结构的简单实现，用于实验，未更新到Module管理
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