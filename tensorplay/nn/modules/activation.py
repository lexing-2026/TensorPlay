import tensorplay
from tensorplay.nn import functional as F
from tensorplay import Tensor
from tensorplay.nn.parameter import Parameter

from .module import Module


__all__ = [
    "Threshold",
    "ReLU",
    "Sigmoid",
    "GELU",
    "Tanh",
    "PReLU",
]


class Threshold(Module):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = tensorplay.nn.Threshold(0, 0.5)
        >>> input = tensorplay.arange(-3, 3)
        >>> output = m(input)
    """

    __constants__ = ["threshold", "value", "inplace"]

    threshold: float
    value: float
    inplace: bool

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.threshold(input, self.threshold, self.value, self.inplace)


class ReLU(Module):
    r"""Applies the rectified linear unit function element-wise.

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.ReLU()
        >>> input = tensorplay.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = tensorplay.randn(2).unsqueeze(0)
        >>> output = tensorplay.cat((m(input), m(-input)))
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class PReLU(Module):
    r"""Applies the element-wise PReLU function.

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \ge 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    Examples::

        >>> m = nn.PReLU()
        >>> input = tensorplay.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.weight = Parameter(tensorplay.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        tensorplay.nn.init.constant_(self.weight, self.init)

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"num_parameters={self.num_parameters}"


class Sigmoid(Module):
    r"""Applies the Sigmoid function element-wise.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = tensorplay.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return tensorplay.sigmoid(input)


class Tanh(Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Tanh()
        >>> input = tensorplay.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return tensorplay.tanh(input)


class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SiLU()
        >>> input = tensorplay.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["approximate"]
    approximate: str

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"approximate={repr(self.approximate)}"