# mypy: allow-untyped-defs

import tensorplay
from tensorplay import Tensor
from tensorplay.distributions import constraints
from tensorplay.distributions.exp_family import ExponentialFamily
from tensorplay.distributions.utils import broadcast_all
from tensorplay.distributions._types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration, generator=None):
    return tensorplay._standard_gamma(concentration, generator=generator)


class Gamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(tensorplay.tensor([1.0]), tensorplay.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> Tensor:
        return self.concentration / self.rate.pow(2)

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, _Number) and isinstance(rate, _Number):
            batch_shape = tensorplay.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = tensorplay.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(
        self,
        sample_shape: _size = tensorplay.Size(),
        *,
        generator: tensorplay.Generator | None = None,
    ) -> Tensor:
        with tensorplay.no_grad():
            return self.rsample(sample_shape, generator=generator)

    def rsample(
        self,
        sample_shape: _size = tensorplay.Size(),
        generator: tensorplay.Generator | None = None,
    ) -> Tensor:
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(
            self.concentration.expand(shape), generator=generator
        ) / self.rate.expand(shape)
        value.detach().clamp_(
            min=tensorplay.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = tensorplay.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            tensorplay.xlogy(self.concentration, self.rate)
            + tensorplay.xlogy(self.concentration - 1, value)
            - self.rate * value
            - tensorplay.lgamma(self.concentration)
        )

    def entropy(self):
        return (
            self.concentration
            - tensorplay.log(self.rate)
            + tensorplay.lgamma(self.concentration)
            + (1.0 - self.concentration) * tensorplay.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration - 1, -self.rate)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return tensorplay.lgamma(x + 1) + (x + 1) * tensorplay.log(-y.reciprocal())

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return tensorplay.special.gammainc(self.concentration, self.rate * value)
