# mypy: allow-untyped-defs
from collections.abc import Callable, Iterable
from math import sqrt
from typing import TypeVar

import tensorplay
from tensorplay import Tensor


__all__ = [
    "bartlett",
    "blackman",
    "cosine",
    "exponential",
    "gaussian",
    "general_cosine",
    "general_hamming",
    "hamming",
    "hann",
    "kaiser",
    "nuttall",
]

_T = TypeVar("_T")

# Text fragments shared by the docstrings of all window builders: the
# length and symmetry arguments every builder takes, the factory arguments
# forwarded to the tensor allocation ops, and the normalization note.
window_common_args = {
    "M": "M (int): number of points of the returned window.",
    "sym": "sym (bool, optional): if `False`, returns a periodic window, which is the "
    "usual choice for spectral analysis. If `True`, returns a symmetric window, "
    "which is the usual choice for filter design. Default: `True`.",
    "dtype": "dtype (:class:`tensorplay.dtype`, optional): the desired data type of the "
    "returned tensor. Default: if ``None``, uses the global default (see "
    ":func:`tensorplay.set_default_dtype`).",
    "layout": "layout (:class:`tensorplay.Layout`, optional): the desired layout of the "
    "returned tensor. Default: ``tensorplay.strided``.",
    "device": "device (:class:`tensorplay.device`, optional): the desired device of the "
    "returned tensor. Default: if ``None``, uses the current default tensor "
    "device (see :func:`tensorplay.set_default_device`).",
    "requires_grad": "requires_grad (bool, optional): whether autograd should record "
    "operations on the returned tensor. Default: ``False``.",
    "normalization": "The window is scaled so that its largest value is 1. The value 1 itself does "
    "not occur when :attr:`M` is even and :attr:`sym` is `True`.",
}


def _add_docstr(*args: str) -> Callable[[_T], _T]:
    r"""Joins the given strings and installs the result as the decorated
    object's docstring.

    This is only worth using when the docstring is assembled from several
    pieces, e.g. when a section is produced with str.format(). Otherwise
    write an ordinary docstring directly.
    """

    def decorator(o: _T) -> _T:
        o.__doc__ = "".join(args)
        return o

    return decorator


def _window_function_checks(
    function_name: str, M: int, dtype: tensorplay.dtype, layout: tensorplay.Layout
) -> None:
    r"""Validates the arguments shared by every window builder and runs
    before any window value is computed.

    Args:
        function_name (str): name of the calling window function, used in
            error messages.
        M (int): requested window length.
        dtype (:class:`tensorplay.dtype`): requested data type of the
            returned tensor.
        layout (:class:`tensorplay.Layout`): requested layout of the
            returned tensor.
    """
    if M < 0:
        raise ValueError(
            f"{function_name} requires non-negative window length, got M={M}"
        )
    if layout is not tensorplay.strided:
        raise ValueError(
            f"{function_name} is implemented for strided tensors only, got: {layout}"
        )
    if dtype not in [tensorplay.float32, tensorplay.float64]:
        raise ValueError(
            f"{function_name} expects float32 or float64 dtypes, got: {dtype}"
        )


@_add_docstr(
    r"""
Computes a window with an exponentially decaying waveform,
also known as the Poisson window.

The samples decay exponentially with the distance from the window center:

.. math::
    w_n = \exp{\left(-\frac{|n - c|}{\tau}\right)}

where `c` is the ``center`` of the window.
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    center (float, optional): location of the window center.
        Default: `M / 2` if `sym` is `False`, else `(M - 1) / 2`.
    tau (float, optional): decay parameter, conceptually a percentage in
        (0, 100]. With `tau = 100` the window degenerates to a constant.
        Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric exponential window of length 10 with decay 1.0.
    >>> # The center is (M - 1) / 2 with M = 10.
    >>> tensorplay.signal.windows.exponential(10)
    tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

    >>> # Periodic exponential window of length 10 with decay 0.5.
    >>> tensorplay.signal.windows.exponential(10, sym=False, tau=0.5)
    tensor([0., 0.0003, 0.0025, 0.0183, 0.1353, 1., 0.1353, 0.0183, 0.0025, 0.0003])
    """.format(**window_common_args),
)
def exponential(
    M: int,
    *,
    center: float | None = None,
    tau: float = 1.0,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("exponential", M, dtype, layout)

    if tau <= 0:
        raise ValueError(f"Tau must be positive, got: {tau} instead.")

    if sym and center is not None:
        raise ValueError("Center must be None for symmetric windows")

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / tau

    k = tensorplay.linspace(
        start=-center * constant,
        end=(-center + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    return tensorplay.exp(-tensorplay.abs(k))


@_add_docstr(
    r"""
Computes a window with a simple cosine waveform, also known as the sine
window.

The samples follow

.. math::
    w_n = \sin\left(\frac{\pi (n + 0.5)}{M}\right)

The 0.5 in the numerator shifts the sample positions by half a step, so the
window starts and ends at non-zero values (for a symmetric window the first
and last samples equal `\sin(\pi / (2M))`).
""",
    r"""

{normalization}

Args:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric cosine window.
    >>> tensorplay.signal.windows.cosine(10)
    tensor([0.1564, 0.454, 0.7071, 0.891, 0.9877, 0.9877, 0.891, 0.7071, 0.454, 0.1564])

    >>> # Periodic cosine window.
    >>> tensorplay.signal.windows.cosine(10, sym=False)
    tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1., 0.9595, 0.8413, 0.6549, 0.4154])
""".format(
        **window_common_args,
    ),
)
def cosine(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("cosine", M, dtype, layout)

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    start = 0.5
    constant = tensorplay.pi / (M + 1 if not sym and M > 1 else M)

    k = tensorplay.linspace(
        start=start * constant,
        end=(start + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    return tensorplay.sin(k)


@_add_docstr(
    r"""
Computes a window with a Gaussian waveform.

The samples follow a Gaussian bump centered in the window:

.. math::
    w_n = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    std (float, optional): standard deviation of the Gaussian; it controls
        how narrow or wide the window is. Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Gaussian window of length 10 with std 1.0.
    >>> tensorplay.signal.windows.gaussian(10)
    tensor([0., 0.0022, 0.0439, 0.3247, 0.8825, 0.8825, 0.3247, 0.0439, 0.0022, 0.])

    >>> # Periodic Gaussian window of length 10 with std 0.9.
    >>> tensorplay.signal.windows.gaussian(10, sym=False, std=0.9)
    tensor([0., 0.0001, 0.0039, 0.0847, 0.5394, 1., 0.5394, 0.0847, 0.0039, 0.0001])
""".format(
        **window_common_args,
    ),
)
def gaussian(
    M: int,
    *,
    std: float = 1.0,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("gaussian", M, dtype, layout)

    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got: {std} instead.")

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    start = -(M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / (std * sqrt(2))

    k = tensorplay.linspace(
        start=start * constant,
        end=(start + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    return tensorplay.exp(-(k**2))


@_add_docstr(
    r"""
Computes the Kaiser window.

The samples are

.. math::
    w_n = I_0 \left( \beta \sqrt{1 - \left( {\frac{n - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

where :math:`I_0` is the modified Bessel function of the first kind of order
zero, evaluated with :func:`tensorplay.i0`, and :math:`N = M - 1` for a
symmetric window, otherwise :math:`N = M`.
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    beta (float, optional): shape parameter of the window. Must be
        non-negative. Default: 12.0
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Kaiser window of length 5 with shape parameter 12.0.
    >>> tensorplay.signal.windows.kaiser(5)
    tensor([0.0001, 0.2157, 1., 0.2157, 0.0001])

    >>> # Periodic Kaiser window of length 5 with shape parameter 0.9.
    >>> tensorplay.signal.windows.kaiser(5, sym=False, beta=0.9)
    tensor([0.8244, 0.9348, 0.9926, 0.9926, 0.9348])
""".format(
        **window_common_args,
    ),
)
def kaiser(
    M: int,
    *,
    beta: float = 12.0,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("kaiser", M, dtype, layout)

    if beta < 0:
        raise ValueError(f"beta must be non-negative, got: {beta} instead.")

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return tensorplay.ones(
            (1,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    # Cast the shape parameter to the requested dtype before computing the
    # grid endpoints below, so that the endpoint arithmetic rounds in the
    # same precision as the window values and the argument of the square
    # root stays non-negative.
    beta = tensorplay.tensor(beta, dtype=dtype, device=device)

    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = tensorplay.minimum(
        beta,
        start + (M - 1) * constant,
    )

    # The grid endpoints are read back as Python scalars: linspace would
    # otherwise derive its result dtype from tensor endpoints and ignore
    # the requested dtype.
    k = tensorplay.linspace(
        start=start.item(),
        end=end.item(),
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    return tensorplay.i0(tensorplay.sqrt(beta * beta - tensorplay.pow(k, 2))) / tensorplay.i0(
        beta
    )


@_add_docstr(
    r"""
Computes the Hamming window.

The samples are

.. math::
    w_n = \alpha - \beta\ \cos \left( \frac{2 \pi n}{M - 1} \right)

with :math:`\alpha = 0.54` and :math:`\beta = 0.46`.
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Hamming window.
    >>> tensorplay.signal.windows.hamming(10)
    tensor([0.08, 0.1876, 0.4601, 0.77, 0.9723, 0.9723, 0.77, 0.4601, 0.1876, 0.08])

    >>> # Periodic Hamming window.
    >>> tensorplay.signal.windows.hamming(10, sym=False)
    tensor([0.08, 0.1679, 0.3979, 0.6821, 0.9121, 1., 0.9121, 0.6821, 0.3979, 0.1679])
""".format(**window_common_args),
)
def hamming(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_hamming(
        M,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Hann window.

The samples are

.. math::
    w_n = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{M - 1} \right)\right] =
    \sin^2 \left( \frac{\pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Hann window.
    >>> tensorplay.signal.windows.hann(10)
    tensor([0., 0.117, 0.4132, 0.75, 0.9698, 0.9698, 0.75, 0.4132, 0.117, 0.])

    >>> # Periodic Hann window.
    >>> tensorplay.signal.windows.hann(10, sym=False)
    tensor([0., 0.0955, 0.3455, 0.6545, 0.9045, 1., 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def hann(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_hamming(
        M,
        alpha=0.5,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Blackman window.

The samples are

.. math::
    w_n = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{M - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Blackman window.
    >>> tensorplay.signal.windows.blackman(5)
    tensor([-0., 0.34, 1., 0.34, -0.])

    >>> # Periodic Blackman window.
    >>> tensorplay.signal.windows.blackman(5, sym=False)
    tensor([-0., 0.2008, 0.8492, 0.8492, 0.2008])
""".format(**window_common_args),
)
def blackman(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("blackman", M, dtype, layout)

    return general_cosine(
        M,
        a=[0.42, 0.5, 0.08],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Bartlett window.

The samples form a triangle:

.. math::
    w_n = 1 - \left| \frac{2n}{M - 1} - 1 \right| = \begin{cases}
        \frac{2n}{M - 1} & \text{if } 0 \leq n \leq \frac{M - 1}{2} \\
        2 - \frac{2n}{M - 1} & \text{if } \frac{M - 1}{2} < n < M \\ \end{cases}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Bartlett window.
    >>> tensorplay.signal.windows.bartlett(10)
    tensor([0., 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.])

    >>> # Periodic Bartlett window.
    >>> tensorplay.signal.windows.bartlett(10, sym=False)
    tensor([0., 0.2, 0.4, 0.6, 0.8, 1., 0.8, 0.6, 0.4, 0.2])
""".format(**window_common_args),
)
def bartlett(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("bartlett", M, dtype, layout)

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return tensorplay.ones(
            (1,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    start = -1
    constant = 2 / (M if not sym else M - 1)

    k = tensorplay.linspace(
        start=start,
        end=start + (M - 1) * constant,
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    return 1 - tensorplay.abs(k)


@_add_docstr(
    r"""
Computes the general cosine window, a weighted sum of cosines whose
frequencies are integer multiples of the fundamental.

The samples are

.. math::
    w_n = \sum^{M-1}_{i=0} (-1)^i a_i \cos{ \left( \frac{2 \pi i n}{M - 1}\right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    a (Iterable): coefficient of each cosine term.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric general cosine window with 3 coefficients.
    >>> tensorplay.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True)
    tensor([0.54, 0.3376, 0.1288, 0.42, 0.9136, 0.9136, 0.42, 0.1288, 0.3376, 0.54])

    >>> # Periodic general cosine window with 2 coefficients.
    >>> tensorplay.signal.windows.general_cosine(10, a=[0.5, 1 - 0.5], sym=False)
    tensor([0., 0.0955, 0.3455, 0.6545, 0.9045, 1., 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def general_cosine(
    M,
    *,
    a: Iterable,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = tensorplay.get_default_dtype()

    _window_function_checks("general_cosine", M, dtype, layout)

    if M == 0:
        return tensorplay.empty(
            (0,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return tensorplay.ones(
            (1,), dtype=dtype, device=device, requires_grad=requires_grad
        )

    if not isinstance(a, Iterable):
        raise TypeError("Coefficients must be a list/tuple")

    if not a:
        raise ValueError("Coefficients cannot be empty")

    constant = 2 * tensorplay.pi / (M if not sym else M - 1)

    k = tensorplay.linspace(
        start=0,
        end=(M - 1) * constant,
        steps=M,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    a_i = tensorplay.tensor(
        [(-1) ** i * w for i, w in enumerate(a)],
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    i = tensorplay.arange(
        a_i.shape[0],
        dtype=a_i.dtype,
        device=a_i.device,
        requires_grad=a_i.requires_grad,
    )
    return (a_i.unsqueeze(-1) * tensorplay.cos(i.unsqueeze(-1) * k)).sum(0)


@_add_docstr(
    r"""
Computes the general Hamming window, the two-term member of the general
cosine family.

The samples are

.. math::
    w_n = \alpha - (1 - \alpha) \cos{ \left( \frac{2 \pi n}{M-1} \right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    alpha (float, optional): the window coefficient. Default: 0.54.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Symmetric Hamming window via the general Hamming builder.
    >>> tensorplay.signal.windows.general_hamming(10, sym=True)
    tensor([0.08, 0.1876, 0.4601, 0.77, 0.9723, 0.9723, 0.77, 0.4601, 0.1876, 0.08])

    >>> # Periodic Hann window via the general Hamming builder.
    >>> tensorplay.signal.windows.general_hamming(10, alpha=0.5, sym=False)
    tensor([0., 0.0955, 0.3455, 0.6545, 0.9045, 1., 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def general_hamming(
    M,
    *,
    alpha: float = 0.54,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_cosine(
        M,
        a=[alpha, 1.0 - alpha],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the minimum 4-term Blackman-Harris window described by Nuttall.

The window is a general cosine sum with the Nuttall coefficients
:math:`a_0 = 0.3635819`, :math:`a_1 = 0.4891775`, :math:`a_2 = 0.1365995`,
:math:`a_3 = 0.0106411`:

.. math::
    w_n = a_0 - a_1 \cos{(z_n)} + a_2 \cos{(2z_n)} - a_3 \cos{(3z_n)}

where :math:`z_n = \frac{2 \pi n}{M - 1}` for a symmetric window and
:math:`z_n = \frac{2 \pi n}{M}` for a periodic one.
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

References::

    - A. Nuttall, "Some windows with very good sidelobe behavior,"
      IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 1, pp. 84-91,
      Feb 1981. https://doi.org/10.1109/TASSP.1981.1163506

    - Heinzel G. et al., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows",
      February 15, 2002 https://holometer.fnal.gov/GH_FFT.pdf

Examples::

    >>> # Symmetric Nuttall window.
    >>> tensorplay.signal.windows.nuttall(5)
    tensor([0.0004, 0.227, 1., 0.227, 0.0004])

    >>> # Periodic Nuttall window.
    >>> tensorplay.signal.windows.nuttall(5, sym=False)
    tensor([0.0004, 0.1105, 0.7983, 0.7983, 0.1105])
""".format(**window_common_args),
)
def nuttall(
    M: int,
    *,
    sym: bool = True,
    dtype: tensorplay.dtype | None = None,
    layout: tensorplay.Layout = tensorplay.strided,
    device: tensorplay.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_cosine(
        M,
        a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
