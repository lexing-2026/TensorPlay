"""Autocast kernel generation (AutocastGenerated.cpp).

Structure mirrors aten/src/ATen/autocast_mode.{h,cpp}:

* The generic AT_FORALL_* macro blocks from ATen/autocast_mode.h feed
  AutocastCUDA upstream (and MTIA/MAIA/XPU, which TensorPlay lacks).
  They are registered for CUDA only here as well.
* CPU does NOT use the generic macros upstream: autocast_mode.cpp carries a
  hand-written KERNEL_CPU registration block with materially different lists
  (e.g. addmv/addr/mv/einsum/softmax/layer_norm are CUDA-only; linear and the
  fp32 linalg/fft/pooling tail are CPU-only).  Those lists are reproduced
  below verbatim, intersected with the operators available in
  native_functions.yaml.
* binary_cross_entropy is policy `banned` on CUDA (binary_cross_entropy_banned
  raises) but plain `fp32` on the CPU list.

Each listed op is wrapped and registered under AutocastCPU/AutocastCUDA; the
generated dispatch sites consult those keys before the autograd keys so casts
are autograd-exposed and inputs are saved for backward post-cast.
"""

from __future__ import annotations

from .api_types import cpp_arg_type, cpp_return_type
from .model import NativeFunction, Type

# ===========================================================================
# ATen/autocast_mode.h -- generic macro blocks (AutocastCUDA upstream)
# ===========================================================================

# AT_FORALL_LOWER_PRECISION_FP (intersected with native_functions.yaml)
AT_FORALL_LOWER_PRECISION_FP = [
    'mm', 'matmul', 'addmm', 'bmm', 'baddbmm',
    'conv1d', 'conv2d', 'conv3d',
    'conv_transpose2d', 'conv_transpose3d',
    'einsum', 'mv', 'scaled_dot_product_attention',
    # upstream also wraps: addmv, addr, addbmm, prelu
    'addmv', 'addr', 'addbmm', 'prelu',
]

# AT_FORALL_FP32 (intersected with native_functions.yaml)
AT_FORALL_FP32 = [
    'acos', 'asin', 'cosh', 'sinh', 'tan',
    'exp', 'expm1', 'log', 'log10', 'log1p', 'log2', 'rsqrt',
    'layer_norm', 'group_norm', 'nll_loss', 'mse_loss',
    # upstream also wraps: erfinv, reciprocal, pow variants, softplus,
    # renorm, logsumexp, dist/pdist, and the fp32 loss family
    'erfinv', 'reciprocal', 'pow.Tensor_Scalar', 'pow.Tensor_Tensor',
    'softplus', 'renorm', 'logsumexp', 'dist', 'pdist',
    'kl_div', 'l1_loss', 'smooth_l1_loss', 'huber_loss',
    'binary_cross_entropy_with_logits', 'poisson_nll_loss',
    'cosine_embedding_loss', 'hinge_embedding_loss',
    'margin_ranking_loss', 'multilabel_margin_loss',
    'soft_margin_loss', 'triplet_margin_loss', 'multi_margin_loss',
    # upsample family runs in fp32 under autocast
    'upsample_nearest1d', 'upsample_nearest2d', 'upsample_nearest3d',
    'upsample_linear1d', 'upsample_bilinear2d',
    'upsample_trilinear3d', 'upsample_bicubic2d',
]

# AT_FORALL_FP32_SET_OPT_DTYPE (intersected with native_functions.yaml)
AT_FORALL_FP32_SET_OPT_DTYPE = [
    'softmax', 'log_softmax', 'sum', 'prod',
    'sum.dim_IntList', 'prod.dim_IntList',
    'cumsum', 'cumprod',
]

# AT_FORALL_PROMOTE
AT_FORALL_PROMOTE = ['atan2', 'addcdiv', 'addcmul', 'dot',
                     'vdot', 'index_put', 'scatter_add']

# ===========================================================================
# aten/src/ATen/autocast_mode.cpp -- hand-written KERNEL_CPU block.
# Reproduced op-for-op (upstream order preserved); ops missing from
# native_functions.yaml are dropped by the generator's intersection pass.
# ===========================================================================

CPU_LOWER_PRECISION_FP = [
    'conv1d', 'conv2d', 'conv3d',
    'bmm', 'mm', 'baddbmm', 'addmm', '_addmm_activation',
    'addbmm', 'linear',
    '_convolution',
    'matmul', 'conv_tbc', 'mkldnn_rnn_layer',
    'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
    'prelu', 'scaled_dot_product_attention',
    '_native_multi_head_attention', 'linalg_vecdot',
]

CPU_FP32 = [
    'avg_pool3d', 'binary_cross_entropy', 'grid_sampler', 'polar',
    'prod', 'prod.dim_IntList',
    'quantile', 'nanquantile',
    'stft', 'cdist',
    'trace', 'view_as_complex',
    'cholesky_inverse', 'cholesky_solve', 'inverse',
    'lu_solve', 'orgqr', 'ormqr', 'pinverse',
    'max_pool3d', 'max_unpool2d', 'max_unpool3d',
    'adaptive_avg_pool3d',
    'reflection_pad1d', 'reflection_pad2d',
    'replication_pad1d', 'replication_pad2d', 'replication_pad3d',
    'mse_loss', 'cosine_embedding_loss', 'nll_loss',
    'hinge_embedding_loss', 'poisson_nll_loss',
    'smooth_l1_loss', 'l1_loss', 'huber_loss',
    'margin_ranking_loss', 'soft_margin_loss',
    'triplet_margin_loss', 'multi_margin_loss',
    'ctc_loss', 'kl_div', 'multilabel_margin_loss',
    'binary_cross_entropy_with_logits',
    'fft_fft', 'fft_ifft',
    'linalg_solve', 'linalg_cholesky',
    'linalg_svdvals', 'linalg_eigvals', 'linalg_inv',
    'linalg_qr', 'linalg_lstsq',
]

CPU_PROMOTE = ['stack', 'cat', 'index_copy']

# Upstream wraps norm via AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE on CUDA et
# al (fp32_append_dtype: run in fp32 by appending dtype to the redispatch).
# TensorPlay's norm overloads take no dtype argument, so the closest available
# behavior is the plain fp32 cast policy on both backends.
NORM_APPEND_DTYPE = ['norm', 'norm.dim']


# ===========================================================================
# Per-backend policy resolution, mirroring the registration blocks in
# ATen/autocast_mode.cpp: AutocastCUDA expands the generic AT_FORALL_* macros,
# AutocastCPU uses the explicit KERNEL_CPU lists.  `banned` reproduces
# upstream's binary_cross_entropy_banned (CUDA et al; CPU runs BCE in fp32).
# ===========================================================================

_CUDA_POLICIES: dict[str, set[str]] = {
    'lower_precision_fp': set(AT_FORALL_LOWER_PRECISION_FP),
    'fp32': set(AT_FORALL_FP32) | set(NORM_APPEND_DTYPE),
    'fp32_set_opt_dtype': set(AT_FORALL_FP32_SET_OPT_DTYPE),
    'promote': set(AT_FORALL_PROMOTE),
    'banned': {'binary_cross_entropy'},
}

_CPU_POLICIES: dict[str, set[str]] = {
    'lower_precision_fp': set(CPU_LOWER_PRECISION_FP),
    'fp32': set(CPU_FP32) | set(NORM_APPEND_DTYPE),
    'promote': set(CPU_PROMOTE),
}

_DEVICE_POLICIES = {'CPU': _CPU_POLICIES, 'CUDA': _CUDA_POLICIES}


def autocast_policy_of(func_name: str, device_key: str | None = None) -> str | None:
    """Policy for `func_name` on `device_key`, or None when unwrapped.

    Exact full-name entries ("pow.Tensor_Scalar", "prod.dim_IntList") only --
    bare base names are resolved by the caller for plain (non-overloaded)
    variants.
    """
    table = _DEVICE_POLICIES.get(device_key or '', {})
    if not table:
        return None
    for policy, ops in table.items():
        if func_name in ops:
            return policy
    return None


def autocast_registered_ops() -> set[str]:
    """Every op probed for autocast anywhere (feeds gen_tpx probe insertion)."""
    ops: set[str] = set()
    for table in _DEVICE_POLICIES.values():
        for s in table.values():
            ops |= s
    return ops


def _arg_expr(policy: str, a) -> str:
    if policy in ('lower_precision_fp', 'fp32', 'promote'):
        return f'::tensorplay::autocast::cached_cast(__to_type, {a.name}, __device_type)'
    if policy == 'fp32_set_opt_dtype':
        if a.type.kind == 'DType':
            return f'::tensorplay::autocast::set_opt_dtype(DType::Float32, {a.name})'
        # Upstream casts every tensor arg to fp32 so reductions accumulate in
        # float; passing them raw leaves the ToCopy node out of the graph and
        # backward then hands a float grad to a lower-precision node.
        return f'::tensorplay::autocast::cached_cast(DType::Float32, {a.name}, __device_type)'
    return a.name


def generate_autocast_registration(funcs: list[NativeFunction]) -> str:
    lines = [
        '// Generated by tools/codegen/main.py -- DO NOT EDIT',
        '#include "Dispatcher.h"',
        '#include "DispatchKey.h"',
        '#include "Device.h"',
        '#include "DType.h"',
        '#include "autocast_mode.h"',
        '#include "autocast_cast.h"',
        '#include "tensorplay/ops/TPXOpsGenerated.h"',
        '',
        'namespace tensorplay {',
        'namespace {',
        '',
    ]

    kernels: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for f in funcs:
        name = f.func_name
        base = f.base_name
        if name in seen or f.skip_implementation:
            continue
        # Out/mutable variants are excluded from autocast (upstream parity:
        # ATen autocast falls back to the un-wrapped op for out= overloads).
        if any(a.type.is_mutable_ref for a in f.args):
            continue
        for device_key in ('CPU', 'CUDA'):
            # Overload-aware policy lookup: an explicit full name
            # ("pow.Tensor_Scalar") wins; the bare base name matches only the
            # plain variant so upstream pairs like sum/sum.dim_IntList are
            # wrapped individually when both are listed.
            policy = autocast_policy_of(name, device_key)
            if policy is None and name == base:
                policy = autocast_policy_of(base, device_key)
            if policy is None:
                continue
            seen.add(name)
            kernel = (f'autocast_kernel_{name.replace(".", "_")}_'
                      f'{device_key.lower()}')
            kernels.append((name, kernel, device_key))

            sig_args = [f'{cpp_arg_type(a.type)} {a.name}' for a in f.args]
            ret = cpp_return_type(f)
            ret_void = ret == 'void'
            lines.append(f'{ret} {kernel}({", ".join(sig_args)}) {{')
            lines.append(f'    const DeviceType __device_type = DeviceType::{device_key};')
            lines.append('    ::tensorplay::autocast::ExcludeAutocastGuard no_autocast(__device_type);')

            call_str = ', '.join(_arg_expr(policy, a) for a in f.args)
            plain = ', '.join(a.name for a in f.args)
            call = f'::tensorplay::tpx::ops::{f.cpp_name}'

            if policy == 'banned':
                # binary_cross_entropy_banned (ATen/autocast_mode.cpp)
                lines.append(
                    '    TP_THROW(RuntimeError, "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\\n"'
                    ' "Many models use a sigmoid layer right before the binary cross entropy layer.\\n"'
                    ' "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\\n"'
                    ' "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\\n"'
                    ' "safe to autocast.");')
            elif policy in ('lower_precision_fp', 'fp32'):
                if policy == 'lower_precision_fp':
                    lines.append(
                        '    const DType __to_type = ::tensorplay::autocast::'
                        'get_lower_precision_fp_from_device_type(__device_type);')
                else:
                    lines.append('    const DType __to_type = DType::Float32;')
                lines.append(f'    return {call}({call_str});' if not ret_void
                             else f'    {call}({call_str});')
            elif policy == 'fp32_set_opt_dtype':
                lines.append(
                    '    if (::tensorplay::autocast::firstarg_is_eligible('
                    f'__device_type, {plain})) {{')
                lines.append(f'        return {call}({call_str});' if not ret_void
                             else f'        {call}({call_str});')
                if ret_void:
                    lines.append('        return;')
                lines.append('    }')
                lines.append(f'    return {call}({plain});' if not ret_void
                             else f'    {call}({plain});')
            else:  # promote
                lines.append(
                    '    const DType __to_type = ::tensorplay::autocast::promote_type(')
                lines.append(
                    '        ::tensorplay::autocast::get_lower_precision_fp_from_device_type(__device_type),')
                lines.append(f'        __device_type, {plain});')
                lines.append(f'    return {call}({call_str});' if not ret_void
                             else f'    {call}({call_str});')
            lines.append('}')
            lines.append('')

    lines += ['} // anonymous namespace', '']
    # Upstream registers these under TORCH_LIBRARY_IMPL(aten, AutocastCPU/CUDA);
    # the library KEY must be the Autocast key -- registering under the bare
    # backend key would shadow the real CPU/CUDA kernels and recurse forever.
    for device_key, lib_key, lib_name in (
        ('CPU', 'AutocastCPU', 'AutocastKernelsCPU'),
        ('CUDA', 'AutocastCUDA', 'AutocastKernelsCUDA'),
    ):
        lines.append(f'TENSORPLAY_LIBRARY_IMPL({lib_key}, {lib_name}) {{')
        for name, kernel, key in kernels:
            if key != device_key:
                continue
            lines.append(f'    m.impl("{name}", &{kernel});')
        lines.append('}')
        lines.append('')
    lines.append('} // namespace tensorplay')
    lines.append('')
    return '\n'.join(lines)
