"""Spectral normalization as a forward pre-hook.

The weight is divided by its largest singular value, estimated by power
iteration: two unit vectors ``u`` and ``v`` are refined against the weight
matrix before every training forward, and ``sigma = u^T W v`` is the estimate
they converge to.  The original parameter is kept as ``weight_orig`` and the
normalized weight is recomputed as a plain attribute, so the optimizer only
ever sees the unnormalized tensor.
"""

from typing import Any, Optional, TypeVar

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.nn.parameter import Parameter, UninitializedParameter

__all__ = [
    "SpectralNorm",
    "SpectralNormLoadStateDictPreHook",
    "SpectralNormStateDictHook",
    "spectral_norm",
    "remove_spectral_norm",
]

_Module = TypeVar("_Module")


class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # Not enforced at initialization; the power iterations converge to it.
    #
    # Version 1 stores `u` and `v` as buffers and keeps `weight` a plain
    # attribute; eval mode rescales the stored original weight with the
    # current sigma estimate instead of persisting the normalized value.
    _version: int = 1

    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        """Fold every axis but `dim` into the columns of a matrix."""
        weight_mat = weight
        if self.dim != 0:
            order = [self.dim] + [d for d in range(weight_mat.dim()) if d != self.dim]
            weight_mat = weight_mat.permute(order)
        height = weight_mat.size(0)
        return weight_mat.reshape([height, -1])

    def compute_weight(self, module, do_power_iteration: bool):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with tp.no_grad():
                for _ in range(self.n_power_iterations):
                    v = F.normalize(tp.mv(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = F.normalize(tp.mv(weight_mat, v), dim=0, eps=self.eps)
                if self.n_power_iterations > 0:
                    # The refined vectors are written back into the module
                    # buffers, while the local copies are detached clones: a
                    # second forward on the same graph must not differentiate
                    # through values this pass overwrote.
                    getattr(module, self.name + "_u").copy_(u)
                    getattr(module, self.name + "_v").copy_(v)
                    u = u.clone()
                    v = v.clone()

        sigma = tp.dot(u, tp.mv(weight_mat, v))
        return weight / sigma

    def remove(self, module) -> None:
        with tp.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, Parameter(weight.detach()))

    def __call__(self, module, inputs: Any) -> None:
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Returns a vector `v` such that `u = normalize(W @ v)` and
        # `u @ W @ v = sigma`.  pinverse handles a singular `W^T W`.
        v = tp.linalg.multi_dot(
            [weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(),
             u.unsqueeze(1)]
        ).squeeze(1)
        return v.mul_(target_sigma / tp.dot(u, tp.mv(weight_mat, v)))

    @staticmethod
    def apply(
        module, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> "SpectralNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on the same "
                    f"parameter {name}"
                )

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None"
            )
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized "
                "parameters. Make sure to run the dummy forward before applying "
                "spectral normalization"
            )

        with tp.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            height, width = weight_mat.size(0), weight_mat.size(1)
            u = F.normalize(
                weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps
            )
            v = F.normalize(
                weight.new_empty(width).normal_(0, 1), dim=0, eps=fn.eps
            )

        del module._parameters[name]
        module.register_parameter(name + "_orig", weight)
        # The recomputed weight lives as a plain attribute so all sorts of
        # consumers (weight init among them) can keep reading `module.weight`.
        setattr(module, name, weight.detach())
        module.register_buffer(name + "_u", u)
        module.register_buffer(name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(
            SpectralNormLoadStateDictPreHook(fn)
        )
        return fn


class SpectralNormStateDictHook:
    """Record the layout version of the normalization into the metadata."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata["spectral_norm"][key] = self.fn._version


class SpectralNormLoadStateDictPreHook:
    """Rebuild `v` for pre-version checkpoints.

    Older checkpoints (version None) carry the normalized `weight` next to
    `weight_orig`, `u` but no usable `v`.  From the invariant
    `u = normalize(W_orig @ v)` with `W = W_orig / sigma`,
    `v` solves `W_orig @ x = u` and is rescaled so `u @ W_orig @ v = sigma`.
    """

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_u", "_v"))
                and weight_key not in state_dict
            ):
                # A hand-assembled state dict that already has the new layout
                # but no metadata yet: treat it as current.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_u"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with tp.no_grad():
                weight_orig = state_dict[weight_key + "_orig"]
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + "_v"] = v


def spectral_norm(
    module: _Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> _Module:
    """Divide ``module.<name>`` by its largest singular value on every forward."""
    if dim is None:
        # Transposed convolutions carry their output channels on axis 1; every
        # other layer this applies to carries them on axis 0.
        if isinstance(
            module,
            (
                tp.nn.ConvTranspose1d,
                tp.nn.ConvTranspose2d,
                tp.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module: _Module, name: str = "weight") -> _Module:
    """Undo :func:`spectral_norm`, restoring the plain parameter."""
    for key, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[key]
            break
    else:
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for key, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[key]
            break

    for key, hook in module._load_state_dict_pre_hooks.items():
        # The registration path wraps the hook; unwrap before matching.
        wrapped = getattr(hook, "hook", hook)
        if isinstance(wrapped, SpectralNormLoadStateDictPreHook) \
                and wrapped.fn.name == name:
            del module._load_state_dict_pre_hooks[key]
            break

    return module
