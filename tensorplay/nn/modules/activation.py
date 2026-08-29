import warnings
from typing import Optional

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
    "ReLU6",
    "Hardswish",
    "Hardsigmoid",
    "LeakyReLU",
    "ELU",
    "Mish",
    "SELU",
    "CELU",
    "Softplus",
    "Softmax",
    "LogSoftmax",
    "LogSigmoid",
    "Hardtanh",
    "Hardshrink",
    "Softshrink",
    "Tanhshrink",
    "Softmin",
    "Softsign",
    "GLU",
    "RReLU",
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

class ReLU6(Module):
    r"""Applies the element-wise function ``ReLU6(x) = min(max(0, x), 6)``.

    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu6(input, self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class Hardswish(Module):
    r"""Applies hardswish, element-wise: ``x * ReLU6(x + 3) / 6``.

    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class Hardsigmoid(Module):
    r"""Applies hardsigmoid, element-wise: ``ReLU6(x + 3) / 6``.

    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class LeakyReLU(Module):
    r"""Applies leaky_relu: ``max(0, x) + negative_slope * min(0, x)``.

    """

    __constants__ = ["negative_slope", "inplace"]

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}" + (", inplace=True" if self.inplace else "")


class ELU(Module):
    r"""Applies elu: ``max(0, x) + min(0, alpha * (exp(x) - 1))``.

    """

    __constants__ = ["alpha", "inplace"]

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}" + (", inplace=True" if self.inplace else "")


class Mish(Module):
    r"""Applies mish: ``x * tanh(softplus(x))``.

    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class SELU(Module):

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class CELU(Module):
    r"""Applies celu: ``max(0, x) + min(0, alpha * (exp(x / alpha) - 1))``."""

    __constants__ = ["alpha", "inplace"]

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}" + (", inplace=True" if self.inplace else "")


class Softplus(Module):
    r"""Applies softplus with linearization above `threshold * beta`."""

    __constants__ = ["beta", "threshold"]

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


class Softmax(Module):

    __constants__ = ["dim"]

    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return tensorplay.softmax(input, self.dim, dtype=None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):

    __constants__ = ["dim"]

    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return tensorplay.log_softmax(input, self.dim, dtype=None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSigmoid(Module):
    r"""Applies the Logsigmoid function element-wise.

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Run forward pass.
        """
        return F.logsigmoid(input)


class Hardtanh(Module):
    r"""Applies the HardTanh function element-wise.

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    __constants__ = ["min_val", "max_val", "inplace"]

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        if min_value is not None:
            warnings.warn(
                "keyword argument `min_value` is deprecated and renamed to `min_val`",
                FutureWarning,
                stacklevel=2,
            )
            min_val = min_value
        if max_value is not None:
            warnings.warn(
                "keyword argument `max_value` is deprecated and renamed to `max_val`",
                FutureWarning,
                stacklevel=2,
            )
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        if self.max_val <= self.min_val:
            raise AssertionError(
                f"max_val ({self.max_val}) must be greater than min_val ({self.min_val})"
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"min_val={self.min_val}, max_val={self.max_val}{inplace_str}"


class Hardshrink(Module):
    r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
            x & \text{ if } x > \lambda \\
            x & \text{ if } x < -\lambda \\
            0 & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5
    """

    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class Softshrink(Module):
    r"""Applies the soft shrinkage function element-wise.

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
            x - \lambda & \text{ if } x > \lambda \\
            x + \lambda & \text{ if } x < -\lambda \\
            0 & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value for the
            Softshrink formulation. Default: 0.5
    """

    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class Tanhshrink(Module):
    r"""Applies element-wise, :math:`\text{Tanhshrink}(x) = x - \text{Tanh}(x)`"""

    def forward(self, input: Tensor) -> Tensor:
        return F.tanhshrink(input)


class Softmin(Module):
    r"""Applies the Softmin function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Args:
        dim (int): A dimension along which Softmin will be computed (so every
            slice along dim will sum to 1).
    """

    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmin(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Softsign(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.softsign(input)


class GLU(Module):
    r"""Applies the Gaussian Error Linear Units function.

    .. math::
        \text{GLU}(a, b) = a \otimes \sigma(b)

    where :math:`a` is the first half of the input matrices and :math:`b` is
    the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1
    """

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class RReLU(Module):
    r"""Applies the randomized leaky rectified linear unit function, element-wise.

    Method described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_.

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    __constants__ = ["lower", "upper", "inplace"]

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"lower={self.lower}, upper={self.upper}{inplace_str}"
