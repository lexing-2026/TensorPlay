# mypy: allow-untyped-defs

import tensorplay
from tensorplay import Tensor
from tensorplay.distributions import constraints
from tensorplay.distributions.distribution import Distribution
from tensorplay.distributions.utils import _get_tracing_state, broadcast_all
from tensorplay.distributions._types import _Number, _size


__all__ = ["Laplace"]


class Laplace(Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Laplace(tensorplay.tensor([0.0]), tensorplay.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2 * self.scale.pow(2)

    @property
    def stddev(self) -> Tensor:
        return (2**0.5) * self.scale

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = tensorplay.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = tensorplay.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = tensorplay.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        finfo = tensorplay.finfo(self.loc.dtype)
        if _get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = tensorplay.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * tensorplay.log1p(
                -u.abs().clamp(min=finfo.tiny)
            )
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * tensorplay.log1p(-u.abs())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -tensorplay.log(2 * self.scale) - tensorplay.abs(value - self.loc) / self.scale

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * tensorplay.expm1(
            -(value - self.loc).abs() / self.scale
        )

    def icdf(self, value):
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * tensorplay.log1p(-2 * term.abs())

    def entropy(self):
        return 1 + tensorplay.log(2 * self.scale)
