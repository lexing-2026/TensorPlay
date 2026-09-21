"""Parametrizations built on top of :mod:`tensorplay.nn.utils.parametrize`.

Each function in this module takes a module and the name of one of its
tensors, registers a parametrization enforcing a structural constraint on
that tensor, and returns the module.  The constraint is maintained
throughout optimization because the gradient flows through the
parametrization map.

Available constraints:

- :func:`orthogonal_`: matrices with orthonormal rows or columns.
- :func:`weight_norm`: weights split into a magnitude and a direction.
- :func:`spectral_norm`: weights rescaled by their largest singular value.

Quantization-aware fake-quantization is intentionally not provided; this
module only covers smooth manifold constraints.
"""

from enum import auto, Enum

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import Tensor
from tensorplay.functional import _weight_norm, norm_except_dim
from tensorplay.nn.modules import Module
from tensorplay.nn.utils import parametrize


__all__ = ["orthogonal_", "spectral_norm", "weight_norm"]


def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = tensorplay.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10.0 * n * tensorplay.finfo(Q.dtype).eps
    return tensorplay.allclose(Q.mH @ Q, Id, atol=eps)


def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = tensorplay.geqrf(A)
    Q = tensorplay.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


class _OrthMaps(Enum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()


class _Orthogonal(Module):
    base: Tensor

    def __init__(
        self, weight, orthogonal_map: _OrthMaps, *, use_trivialization=True
    ) -> None:
        super().__init__()

        # For complex tensors it is not possible to recover the reflector
        # coefficients `tau` from the strictly lower-triangular reflectors:
        # the reflectors carry n(n-1) real parameters while the unitary
        # matrices need n^2, so the parametrization is not invertible when the
        # weights are optimised independently.  We simply reject that
        # combination.
        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError(
                "The householder parametrization does not support complex tensors."
            )

        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)

    def forward(self, X: Tensor) -> Tensor:
        n, k = X.size(-2), X.size(-1)
        transposed = n < k
        if transposed:
            X = X.mT
            n, k = k, n
        # Here n > k and X is a tall matrix
        if (
            self.orthogonal_map == _OrthMaps.matrix_exp
            or self.orthogonal_map == _OrthMaps.cayley
        ):
            # We just need n x k - k(k-1)/2 parameters
            X = X.tril()
            if n != k:
                # Embed into a square matrix
                X = tensorplay.cat(
                    [X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1
                )
            A = X - X.mH
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = tensorplay.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = tensorplay.eye(n, dtype=A.dtype, device=A.device)
                Q = tensorplay.linalg.solve(
                    tensorplay.add(Id, A, alpha=-0.5), tensorplay.add(Id, A, alpha=0.5)
                )
            # Q is now orthogonal (or unitary) of size (..., n, n)
            if n != k:
                Q = Q[..., :k]
            # Q is now the size of the X (albeit perhaps transposed)
        else:
            # X is real here, as we do not support householder with complex numbers
            A = X.tril(diagonal=-1)
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            Q = tensorplay.linalg.householder_product(A, tau)
            # The diagonal of X is 1's and -1's
            # We do not want to differentiate through this or update the diagonal of X hence the casting
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)

        if hasattr(self, "base"):
            Q = self.base @ Q
        if transposed:
            Q = Q.mT
        return Q

    @tensorplay.no_grad()
    def right_inverse(self, Q: Tensor) -> Tensor:
        if Q.shape != self.shape:
            raise ValueError(
                f"Expected a matrix or batch of matrices of shape {self.shape}. "
                f"Got a tensor of shape {Q.shape}."
            )

        Q_init = Q
        n, k = Q.size(-2), Q.size(-1)
        transpose = n < k
        if transpose:
            Q = Q.mT
            n, k = k, n

        # We always make sure to copy Q in every path
        if not hasattr(self, "base"):
            # Without the dynamic trivialization we only implement the
            # inverse of the forward map for the Householder parametrization.
            # For the Cayley map and the matrix exponential there is no known
            # closed-form way to find the skew-symmetric X that reproduces a
            # given orthogonal matrix through the composition
            # cat(X.tril(), 0) -> A = Y - Y^H -> map(A)[:, :k].
            if (
                self.orthogonal_map == _OrthMaps.cayley
                or self.orthogonal_map == _OrthMaps.matrix_exp
            ):
                raise NotImplementedError(
                    "It is not possible to assign to the matrix exponential "
                    "or the Cayley parametrizations when use_trivialization=False."
                )

            # Make Q orthogonal via the QR decomposition.
            # Here Q is always real because we do not support householder and complex matrices.
            A, tau = tensorplay.geqrf(Q)
            # We want to have a decomposition X = QR with diag(R) > 0, as otherwise we could
            # decompose an orthogonal matrix Q as Q = (-Q)@(-Id), which is a valid QR decomposition
            # The diagonal of Q is the diagonal of R from the qr decomposition
            A.diagonal(dim1=-2, dim2=-1).sign_()
            # Equality with zero is ok because a QR routine returns exactly zero when it does not
            # want to use a particular reflection
            A.diagonal(dim1=-2, dim2=-1)[tau == 0.0] *= -1
            return A.mT if transpose else A
        else:
            if n == k:
                # We check whether Q is orthogonal
                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:  # Is orthogonal
                    Q = Q.clone()
            else:
                # Complete Q into a full n x n orthogonal matrix
                N = tensorplay.randn(
                    *(Q.size()[:-2] + (n, n - k)), dtype=Q.dtype, device=Q.device
                )
                Q = tensorplay.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q

            # It is necessary to return the -Id, as we use the diagonal for the
            # Householder parametrization. Using -Id makes:
            # householder(zeros(m, n)) == eye(m, n)
            # Poor man's version of eye_like
            neg_Id = tensorplay.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.0)
            return neg_Id


def orthogonal_(
    module: Module,
    name: str = "weight",
    orthogonal_map: str | None = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    r"""Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the parametrized
    matrix :math:`Q \in \mathbb{K}^{m \times n}` is **orthogonal** as

    .. math::

        \begin{align*}
            Q^{\text{H}}Q &= \mathrm{I}_n \mathrlap{\qquad \text{if }m \geq n}\\
            QQ^{\text{H}} &= \mathrm{I}_m \mathrlap{\qquad \text{if }m < n}
        \end{align*}

    where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex
    and the transpose when :math:`Q` is real-valued, and
    :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
    In plain words, :math:`Q` will have orthonormal columns whenever :math:`m \geq n`
    and orthonormal rows otherwise.

    If the tensor has more than two dimensions, we consider it as a batch of matrices of shape `(..., m, n)`.

    The matrix :math:`Q` may be parametrized via three different ``orthogonal_map`` in terms of the original tensor:

    - ``"matrix_exp"``/``"cayley"``:
      the :func:`~tensorplay.matrix_exp` :math:`Q = \exp(A)` and the `Cayley map`_
      :math:`Q = (\mathrm{I}_n + A/2)(\mathrm{I}_n - A/2)^{-1}` are applied to a skew-symmetric
      :math:`A` to give an orthogonal matrix.
    - ``"householder"``: computes a product of Householder reflectors
      (:func:`~tensorplay.linalg.householder_product`).

    ``"matrix_exp"``/``"cayley"`` often make the parametrized weight converge faster than
    ``"householder"``, but they are slower to compute for very thin or very wide matrices.

    If ``use_trivialization=True`` (default), the parametrization implements the "Dynamic Trivialization Framework",
    where an extra matrix :math:`B \in \mathbb{K}^{n \times n}` is stored under
    ``module.parametrizations.weight[0].base``. This helps the
    convergence of the parametrized layer at the expense of some extra memory use.
    See `Trivializations for Gradient-Based Optimization on Manifolds`_ .

    Initial value of :math:`Q`:
    If the original tensor is not parametrized and ``use_trivialization=True`` (default), the initial value
    of :math:`Q` is that of the original tensor if it is orthogonal (or unitary in the complex case)
    and it is orthogonalized via the QR decomposition otherwise (see :func:`tensorplay.linalg.qr`).
    Same happens when it is not parametrized and ``orthogonal_map="householder"`` even when ``use_trivialization=False``.
    Otherwise, the initial value is the result of the composition of all the registered
    parametrizations applied to the original tensor.

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~tensorplay.nn.utils.parametrize.register_parametrization`.


    .. _`Cayley map`: https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    .. _`Trivializations for Gradient-Based Optimization on Manifolds`: https://arxiv.org/abs/1909.09501

    Args:
        module (nn.Module): module on which to register the parametrization.
        name (str, optional): name of the tensor to make orthogonal. Default: ``"weight"``.
        orthogonal_map (str, optional): One of the following: ``"matrix_exp"``, ``"cayley"``, ``"householder"``.
            Default: ``"matrix_exp"`` if the matrix is square or complex, ``"householder"`` otherwise.
        use_trivialization (bool, optional): whether to use the dynamic trivialization framework.
            Default: ``True``.

    Returns:
        The original module with an orthogonal parametrization registered to the specified
        weight

    Example::

        >>> orth_linear = orthogonal_(nn.Linear(20, 40))
        >>> Q = orth_linear.weight
        >>> tensorplay.dist(Q.T @ Q, tensorplay.eye(20))
        tensor(4.9332e-07)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    # We could implement this for 1-dim tensors as the maps on the sphere
    # but I believe it'd bite more people than it'd help
    if weight.ndim < 2:
        raise ValueError(
            "Expected a matrix or batch of matrices. "
            f"Got a tensor of {weight.ndim} dimensions."
        )

    if orthogonal_map is None:
        orthogonal_map = (
            "matrix_exp"
            if weight.size(-2) == weight.size(-1) or weight.is_complex()
            else "householder"
        )

    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError(
            'orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
            f"Got: {orthogonal_map}"
        )
    orth = _Orthogonal(weight, orth_enum, use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module


class _WeightNorm(Module):
    def __init__(
        self,
        dim: int | None = 0,
    ) -> None:
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    def forward(self, weight_g, weight_v):
        return _weight_norm(weight_v, weight_g, self.dim)

    def right_inverse(self, weight):
        weight_g = norm_except_dim(weight, 2, self.dim)
        weight_v = weight

        return weight_g, weight_v


def weight_norm(module: Module, name: str = "weight", dim: int = 0):
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` with two parameters: one specifying the magnitude
    and one specifying the direction.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm parametrization

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
        )
        >>> m.parametrizations.weight.original0.size()
        tensorplay.Size([40, 1])
        >>> m.parametrizations.weight.original1.size()
        tensorplay.Size([40, 20])

    """
    _weight_norm_param = _WeightNorm(dim)
    parametrize.register_parametrization(module, name, _weight_norm_param, unsafe=True)

    def _weight_norm_compat_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        g_key = f"{prefix}{name}_g"
        v_key = f"{prefix}{name}_v"
        if g_key in state_dict and v_key in state_dict:
            original0 = state_dict.pop(g_key)
            original1 = state_dict.pop(v_key)
            state_dict[f"{prefix}parametrizations.{name}.original0"] = original0
            state_dict[f"{prefix}parametrizations.{name}.original1"] = original1

    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    return module


class _SpectralNorm(Module):
    def __init__(
        self,
        weight: Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        # Precondition
        if weight.ndim <= 1:
            raise AssertionError(
                f"Expected weight to have more than 1 dimension, got {weight.ndim}"
            )

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @tensorplay.no_grad()
    def _power_method(self, weight_mat: Tensor, n_power_iterations: int) -> None:
        # The `u` and `v` buffers are updated in-place during the power
        # iteration, so that a replica of this module shares the refined
        # singular vectors with its source.  After the update they are
        # cloned before normalizing the weight so that a second forward
        # pass on the same graph can still differentiate through the values
        # this pass consumed (the common `loss = D(real) - D(fake)`
        # pattern needs this).
        #
        # Precondition
        if weight_mat.ndim <= 1:
            raise AssertionError(
                f"Expected weight_mat to have more than 1 dimension, got {weight_mat.ndim}"
            )

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u.copy_(F.normalize(tensorplay.mv(weight_mat, self._v), dim=0, eps=self.eps))
            self._v.copy_(F.normalize(tensorplay.mv(weight_mat.H, self._u), dim=0, eps=self.eps))

    def forward(self, weight: Tensor) -> Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=tensorplay.contiguous_format)
            v = self._v.clone(memory_format=tensorplay.contiguous_format)
            sigma = tensorplay.vdot(u, tensorplay.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: Tensor) -> Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~tensorplay.nn.utils.parametrize.register_parametrization`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::

        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> tensorplay.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(
            module,
            (
                tensorplay.nn.ConvTranspose1d,
                tensorplay.nn.ConvTranspose2d,
                tensorplay.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module, name, _SpectralNorm(weight, n_power_iterations, dim, eps)
    )
    return module
