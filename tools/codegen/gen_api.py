"""Generators for TensorGenerated.h/.cpp and TensorRedispatchGenerated.h.

Structure follows the generator's split between redispatch and public methods:
every op gets (1) a backend `redispatch` entry point that resolves its kernel
through the Dispatcher, bumps versions of mutated arguments, and is shared by
the eager path and the autograd wrappers, and (2) a public Tensor method that
performs device checks and consults Autocast/Autograd dispatch keys before
funneling into redispatch.
"""

from __future__ import annotations

from .api_types import (
    autograd_node_name,
    binding_default,
    call_arg_expr,
    cpp_arg_type,
    cpp_default,
    cpp_return_type,
    stub_arg_type,
    stub_arg_type_for,
)
from .model import (
    Argument,
    NativeFunction,
    redispatch_key as _redispatch_key,
    redispatch_name as _redispatch_name,
)


_RANDOM_TRANSFORM_OPS = frozenset({
    "rand",
    "randn",
    "randint",
    "randperm",
    "rand_like",
    "randint_like",
    "randn_like",
    "bernoulli",
    "normal",
    "exponential_",
    "normal_",
    "random_",
    "uniform_",
    "bernoulli_",
    "multinomial",
})


def _variant_instances(f: NativeFunction):
    for v in f.variants:
        yield v


# Method variants declared by hand in p10/include/Tensor.h.  Ownership of
# these members stays with the handwritten class body: generation keeps the
# dispatcher-level redispatch helpers and the free registration surface, but
# must not redeclare the members (a duplicate declaration inside the same
# class is a hard C++ error, and two implementations of one method would
# break the single-ownership rule).  Members listed here also cover the
# defaulted-parameter case: a handwritten no-argument member plus a generated
# one-argument member with a default value makes zero-argument call sites
# ambiguous, so the generated form cannot coexist either.
_HANDWRITTEN_TENSOR_METHODS = frozenset({
    "dim", "numel", "is_contiguous", "retains_grad", "detach", "as_strided",
    "select", "sparse_dim", "dense_dim", "is_coalesced", "coalesce",
    "_values", "_indices", "sparse_mask", "to", "is_pinned", "pin_memory",
    "item", "view",
})


def _owns_method_variant(f: NativeFunction, variant: str) -> bool:
    return f.manual_cpp_binding or (
        variant == "method" and f.cpp_name in _HANDWRITTEN_TENSOR_METHODS
    )


def _scalar_dim_arg(f: NativeFunction) -> Argument | None:
    candidates = [
        a for a in f.args
        if a.name == "dim" and a.type.is_list
        and a.type.kind == "int64_t"
    ]
    return candidates[0] if len(candidates) == 1 else None


def _scalar_dim_overload_arg(f: NativeFunction) -> Argument | None:
    dim_arg = _scalar_dim_arg(f)
    if dim_arg is None:
        return None
    for a in f.args:
        if a is dim_arg:
            break
        if a.name == "requires_grad":
            continue
        if a.default and a.name not in f.cpp_no_default_args:
            return None
    return dim_arg


def _overload_key(f: NativeFunction, variant: str,
                  *, scalar_dim: bool = False) -> tuple:
    """Identity of a Tensor member for C++ overloading purposes.

    Parameter *types* decide: argument names, default values, constness, and
    staticness never distinguish overloads, so two members of the same class
    with identical parameter type lists are a redeclaration conflict even
    when their schema entries differ.
    """
    scalar_dim_arg = _scalar_dim_arg(f) if scalar_dim else None
    params = tuple(
        "int64_t" if a is scalar_dim_arg else cpp_arg_type(a.type)
        for a in f.args
        if a.name != "requires_grad"
        and not (variant == "method" and a.name == "self")
    )
    return (f.cpp_name, params)


def _is_const_method(f: NativeFunction) -> bool:
    self_a = f.self_arg()
    return bool(self_a) and not self_a.type.is_mutable_ref


def _member_params(f: NativeFunction, variant: str) -> list:
    """(C++ parameter type, has-default) pairs in schema order."""
    out = []
    for a in f.args:
        if a.name == "requires_grad":
            continue
        if variant == "method" and a.name == "self":
            continue
        out.append((cpp_arg_type(a.type), bool(a.default)))
    return out


def _prefix_ambiguous(prev: list, new: list) -> bool:
    """True when a shared-prefix call site is viable for both members.

    Two members with the same name collide when their leading parameter
    types agree up to the longest common prefix and every parameter past
    that prefix is defaulted on both sides: a call supplying only the
    prefix is then viable for each member and overload resolution cannot
    pick one.
    """
    k = 0
    while k < len(prev) and k < len(new) and prev[k][0] == new[k][0]:
        k += 1
    return (all(d for _, d in prev[k:])
            and all(d for _, d in new[k:]))


def plan_members(funcs: list[NativeFunction]) -> dict:
    """Decide which variant renders each C++ member of class Tensor.

    Schema overloads flatten onto one C++ member name, and code generation
    only keeps overloads that form one resolvable overload set:
    an operator and its overloads whose C++ parameter lists coincide
    collapse into a single member.  When a later overload would instead
    create a prefix ambiguity (all-defaulted tails behind a shared
    prefix), it yields the C++ member to the earlier variant and keeps
    only its dispatcher-level surface -- the redispatch helper and the
    registration entry remain, so dispatch behavior is unchanged.
    """
    plan: dict = {}
    emitted: dict = {}
    for f in funcs:
        for variant in _variant_instances(f):
            owned = _owns_method_variant(f, variant)
            params = _member_params(f, variant)
            collides = False
            for _pv in emitted.get(f.cpp_name, ()):
                if _prefix_ambiguous(_pv, params):
                    collides = True
                    break
            if not (collides or owned):
                emitted.setdefault(f.cpp_name, []).append(params)
            plan[(f.func_name, variant)] = not collides and not owned
    return plan


def _sig_args(f: NativeFunction, variant: str, *, with_defaults: bool,
              scalar_dim: bool = False) -> list[str]:
    out = []
    scalar_dim_arg = _scalar_dim_arg(f) if scalar_dim else None
    for a in f.args:
        if variant == "method" and a.name == "self":
            continue
        if a.name == "requires_grad":
            continue
        arg_type = "int64_t" if a is scalar_dim_arg else cpp_arg_type(a.type)
        s = f"{arg_type} {a.name}"
        if (with_defaults and not f.is_out and a.default
                and a is not scalar_dim_arg
                and a.name not in f.cpp_no_default_args):
            s += f" = {cpp_default(a.type, a.default)}"
        out.append(s)
    return out


def method_signature(f: NativeFunction, variant: str, *, declaration: bool,
                     qualified: bool = False,
                     scalar_dim: bool = False) -> str:
    ret = cpp_return_type(f)
    args = ", ".join(_sig_args(
        f, variant, with_defaults=declaration, scalar_dim=scalar_dim))
    qual = "Tensor::" if qualified else ""
    sig = f"{ret} {qual}{f.cpp_name}({args})"
    if variant == "method" and _is_const_method(f):
        sig += " const"
    return sig


# ---------------------------------------------------------------------------
# Device resolution (shared between the .cpp emitters)
# ---------------------------------------------------------------------------

def _device_source(f: NativeFunction, variant: str) -> tuple[str, str]:
    """Return (dispatch_key_source, target_device_expr)."""
    cpu = "Device(DeviceType::CPU)"
    device_arg = f.arg("device")
    if device_arg is not None:
        self_a = f.self_arg()
        if device_arg.type.is_opt:
            if f.base_name.endswith("_like") and self_a is not None:
                if variant == "method":
                    val = (f"{device_arg.name}.has_value() ? *{device_arg.name} "
                           ": device()")
                else:
                    val = (f"{device_arg.name}.has_value() ? *{device_arg.name} "
                           f": {self_a.name}.device()")
                return val, val
            val = (f"{device_arg.name}.has_value() ? *{device_arg.name} : {cpu}")
            return val, val
        return device_arg.name, device_arg.name
    if variant == "method" and f.self_arg() is not None:
        return "device()", "device()"
    first_tensor = next((a for a in f.args if a.type.kind == "Tensor"
                         and not a.type.is_opt and not a.type.is_list), None)
    if first_tensor is not None:
        return f"{first_tensor.name}.device()", f"{first_tensor.name}.device()"
    first_list = next((a for a in f.args if a.type.is_tensor_like and a.type.is_list), None)
    if first_list is not None:
        val = f"deviceForTensorArg({first_list.name})"
        return val, val
    return cpu, cpu


def _dispatch_key_expr(f: NativeFunction, variant: str, *, redispatch: bool) -> str:
    """Select the highest active transform across every tensor argument."""
    if f.func_name == "copy_":
        target = "self"
        if variant == "method" and not redispatch:
            target = "*this"
        return f"dispatchKeyForTensorArgs({target})"
    if f.func_name in _RANDOM_TRANSFORM_OPS:
        _dispatch, device = _device_source(f, variant)
        return f"transform::dispatch_key_for_random(computeDispatchKey({device}))"
    names = []
    for a in f.args:
        if not a.type.is_tensor_like:
            continue
        if variant == "method" and a.name == "self" and not redispatch:
            names.append("*this")
        else:
            names.append(a.name)
    if names:
        return f"dispatchKeyForTensorArgs({', '.join(names)})"
    _dispatch, device = _device_source(f, variant)
    return f"computeDispatchKey({device})"


# ---------------------------------------------------------------------------
# TensorGenerated.h  (included inside class Tensor)
# ---------------------------------------------------------------------------

def generate_header(funcs: list[NativeFunction]) -> str:
    lines = [
        "// Generated by tools/codegen/main.py -- DO NOT EDIT",
        "#pragma once",
        "#include <tuple>",
        "",
    ]
    member_plan = plan_members(funcs)
    seen_decl = set()
    for f in funcs:
        for variant in _variant_instances(f):
            if not member_plan.get((f.func_name, variant), True):
                continue
            # (`int[]`) intentionally share the same std::vector ABI.  Their
            # schema defaults may differ (`padding=0` vs `padding=[]`), but
            # C++ cannot overload those declarations.  Deduplicate on the
            # parameter types, matching generate_cpp's definition dedup.
            key = _overload_key(f, variant)
            if key in seen_decl:
                continue
            seen_decl.add(key)
            # Function variants become static members of Tensor (factory
            # surface, e.g. Tensor::zeros), method variants are instance
            # methods -- using the same generated layout.
            prefix = "static " if variant == "function" else ""
            lines.append(prefix + method_signature(f, variant, declaration=True) + ";")
            lines.append("")
            if _scalar_dim_overload_arg(f) is not None:
                scalar_key = _overload_key(f, variant, scalar_dim=True)
                if scalar_key not in seen_decl:
                    seen_decl.add(scalar_key)
                    lines.append(
                        prefix + method_signature(
                            f, variant, declaration=True, scalar_dim=True
                        ) + ";"
                    )
                    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# detail::redispatch_* helpers + Tensor methods (.cpp)
# ---------------------------------------------------------------------------

_CPP_INCLUDES = """// Generated by tools/codegen/main.py -- DO NOT EDIT
#include "Tensor.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "DispatchKey.h"
#include "GradMode.h"
#include "InferenceMode.h"
#include "DType.h"
#include "Scalar.h"
#include "SizesAndStrides.h"
#include "Device.h"
#include "TypePromotion.h"
#include "autocast_mode.h"
#include "TransformDispatch.h"
#include "Profiler.h"
#ifdef USE_CUDA
#include "CUDARuntime.h"
#endif
#include <tuple>
#include <utility>
"""


def _emit_redispatch(lines, f, variant, dev_src, helper_name):
    ret = cpp_return_type(f)
    tmpl = [ret]
    rd_args, rd_call = [], []
    for a in f.args:
        if a.name == "requires_grad":
            continue
        st = stub_arg_type(a.type)
        tmpl.append(st)
        rd_args.append(f"{st} {a.name}")
        rd_call.append(a.name)

    rd_dev = dev_src
    if variant == "method" and f.self_arg() is not None:
        rd_dev = dev_src.replace("device()", "self.device()")
    mutable = [a.name for a in f.mutable_args]

    lines.append("namespace detail {")
    lines.append(f"TENSORPLAY_API {ret} {helper_name}({', '.join(rd_args)}) {{")
    # Op-level profiler record: every dispatched execution passes through
    # exactly one redispatch funnel, so this is the single instrumentation
    # dispatch).  Inactive cost: one acquire-load of a static atomic.
    lines.append(
        f'    tensorplay::prof::OpRecord __tp_prof_rec("{f.func_name}");')
    # record_shapes: only tensor-like args are captured, and only when the
    # session requested shapes (second load sits inside the active branch,
    # so the fully-inactive path still costs exactly one load).
    shape_args = [
        a for a in f.args
        if a.type.is_tensor_like and a.name != "requires_grad"
    ]
    if shape_args:
        lines.append(
            "    if (tensorplay::prof::g_capture_shapes.load("
            "std::memory_order_acquire)) {")
        lines.append("        std::vector<std::vector<int64_t>> __tp_prof_shapes;")
        lines.append("        std::vector<int32_t> __tp_prof_dtypes;")
        for a in shape_args:
            t = a.type
            if t.is_list:
                lines.append(
                    f"        for (const auto& __tp_t : {a.name}) {{")
                if t.list_elem_opt:
                    lines.append("            if (!__tp_t.has_value()) continue;")
                    lines.append(
                        "            __tp_prof_shapes.push_back("
                        "static_cast<std::vector<int64_t>>(__tp_t->shape()));")
                    lines.append(
                        "            __tp_prof_dtypes.push_back(static_cast<int32_t>"
                        "(__tp_t->dtype()));")
                else:
                    lines.append(
                        "            __tp_prof_shapes.push_back("
                        "static_cast<std::vector<int64_t>>(__tp_t.shape()));")
                    lines.append(
                        "            __tp_prof_dtypes.push_back(static_cast<int32_t>"
                        "(__tp_t.dtype()));")
                lines.append("        }")
            elif t.is_opt:
                lines.append(
                    f"        if ({a.name}.has_value()) {{")
                lines.append(
                    f"            __tp_prof_shapes.push_back(static_cast<"
                    f"std::vector<int64_t>>({a.name}->shape()));")
                lines.append(
                    f"            __tp_prof_dtypes.push_back(static_cast<int32_t>"
                    f"({a.name}->dtype()));")
                lines.append("        }")
            elif t.is_mutable_ref:
                lines.append(
                    f"        __tp_prof_shapes.push_back("
                    f"static_cast<std::vector<int64_t>>({a.name}.shape()));")
                lines.append(
                    f"        __tp_prof_dtypes.push_back(static_cast<int32_t>"
                    f"({a.name}.dtype()));")
            else:
                lines.append(
                    f"        __tp_prof_shapes.push_back("
                    f"static_cast<std::vector<int64_t>>({a.name}.shape()));")
                lines.append(
                    f"        __tp_prof_dtypes.push_back(static_cast<int32_t>"
                    f"({a.name}.dtype()));")
        lines.append(
            "        __tp_prof_rec.set_io_meta(std::move(__tp_prof_shapes),"
            " std::move(__tp_prof_dtypes));")
        lines.append("    }")
    # GPU timeline: pool-backed cudaEvent pair around dispatched CUDA work
    # (USE_CUDA builds only; resolved at session stop after one sync).
    lines.append("#ifdef USE_CUDA")
    lines.append("    tensorplay::prof::GpuTimerPair __tp_gpu(__tp_prof_rec);")
    lines.append("#endif")
    if f.device_guard:
        lines.append("#ifdef USE_CUDA")
        # Brace-init: a parenthesized placeholder like Device(DeviceType::CPU)
        # re-parses as a parameter declaration on MSVC (C2751).
        lines.append(f"    cuda::OptionalCUDAGuard device_guard{{{rd_dev}}};")
        lines.append("#endif")
    lines.append(
        f'    static const OperatorHandle op_handle = '
        f'Dispatcher::singleton().findHandle("{f.func_name}");')
    # The redispatch helpers are free functions: a method variant's implicit
    # receiver becomes the explicit `self` parameter here, so every receiver
    # expression must be rewritten the same way as the device source above.
    dispatch_key_expr = _dispatch_key_expr(f, variant, redispatch=True)
    if variant == "method" and f.self_arg() is not None:
        dispatch_key_expr = dispatch_key_expr.replace("device()", "self.device()")
    if f.func_name in _RANDOM_TRANSFORM_OPS:
        key_expr = dispatch_key_expr
    else:
        # A wrapped tensor's dispatch key names the active transform level:
        # route it to the registered batch rule instead of collapsing it to
        # the backend, which would run the raw kernel on the wrapper.
        # Unwrapped arguments keep resolving to their backend.
        lines.append(f"    DispatchKey __tp_arg_key = {dispatch_key_expr};")
        key_expr = "(is_vmap_key(__tp_arg_key) ? __tp_arg_key : toBackendKey(__tp_arg_key))"
    lines.append(
        f"    DispatchKey dispatch_key = {key_expr};"
    )
    lines.append("#ifdef USE_CUDA")
    lines.append(f"    __tp_gpu.arm({rd_dev});")
    lines.append("#endif")

    dispatch_stub = f"DispatchStub<{', '.join(tmpl)}>"
    if rd_call:
        call = f"{dispatch_stub}::call(op_handle, dispatch_key, {', '.join(rd_call)})"
    else:
        call = f"{dispatch_stub}::call(op_handle, dispatch_key)"
    ret_void = ret == "void"
    if not mutable:
        if ret_void:
            lines.append(f"    {call};")
        else:
            lines.append(f"    auto&& __tp_result = {call};")
        lines.append("#ifdef USE_CUDA")
        lines.append("    __tp_gpu.close();")
        lines.append("#endif")
        # Output allocation volume for Tensor-returning ops (memory view).
        if not ret_void and ret == "Tensor":
            lines.append(
                "    if (tensorplay::prof::g_capture_shapes.load("
                "std::memory_order_acquire)) {")
            lines.append(
                "        __tp_prof_rec.set_output_bytes("
                "static_cast<int64_t>(__tp_result.numel()) * "
                "static_cast<int64_t>(__tp_result.itemsize()));")
            lines.append("    }")
        if not ret_void:
            lines.append("    return std::move(__tp_result);")
    else:
        lines.append(f"    {call};" if ret_void else
                     f"    auto&& __tp_result = {call};")
        lines.append("#ifdef USE_CUDA")
        lines.append("    __tp_gpu.close();")
        lines.append("#endif")
        for m in mutable:
            at = f.arg(m).type
            if at.is_mutable_tensor_list:
                lines.append(f"    for (const auto& __tp_tensor : {m}) {{")
                lines.append("        if (__tp_tensor.defined() && !InferenceMode::is_enabled()) __tp_tensor.unsafeGetTensorImpl()->bump_version();")
                lines.append("    }")
            else:
                lines.append(f"    if (!InferenceMode::is_enabled()) {m}.unsafeGetTensorImpl()->bump_version();")
        lines.append("    return;" if ret_void else
                     "    return std::forward<decltype(__tp_result)>(__tp_result);")
    lines.append("}")
    lines.append("} // namespace detail")
    lines.append("")
    return helper_name


def _emit_device_checks(lines, f, variant, target_dev):
    if f.device_check == "NoCheck" or f.cpp_name == "copy_":
        return
    target_expr = f"({target_dev})"
    target_text = f"{target_expr}.toString()"
    is_factory_like = f.base_name.endswith("_like")
    checked = {"Tensor", "Tensor?", "Tensor[]"}
    for a in f.args:
        if variant == "method" and a.name == "self" and not is_factory_like:
            continue
        if is_factory_like and a.name == "self":
            continue
        t = a.type
        if t.is_mutable_ref:
            kind = "Tensor"
        elif t.is_mutable_tensor_list:
            kind = "Tensor[]"
        elif t.is_tensor_like:
            if t.is_list:
                kind = "Tensor?[]" if t.list_elem_opt else "Tensor[]"
            else:
                kind = "Tensor?" if t.is_opt else "Tensor"
        else:
            continue
        if kind == "Tensor?":
            cond = (f"{a.name}.has_value() && {a.name}->defined() && "
                    f"{a.name}->device() != {target_expr}")
        elif kind in ("Tensor[]", "Tensor?[]"):
            cond = None
        else:
            cond = f"{a.name}.defined() && {a.name}.device() != {target_expr}"
        if kind in ("Tensor[]", "Tensor?[]"):
            lines.append(f"    for (const auto& t : {a.name}) {{")
            tensor_expr = "t->" if kind == "Tensor?[]" else "t."
            present = "t.has_value() && " if kind == "Tensor?[]" else ""
            lines.append(
                f"        if ({present}{tensor_expr}defined() && "
                f"{tensor_expr}device() != {target_expr}) {{")
            shown_device = "t->device()" if kind == "Tensor?[]" else "t.device()"
            lines.append(
                '            TP_THROW(DeviceMismatchError, "Expected all tensors to be on the same device, but found one (in '
                + a.name + ') on " + ' + shown_device + '.toString() + " and another (target) on " + '
                + target_text + ');')
            lines.append("        }")
            lines.append("    }")
            continue
        lines.append(f"    if ({cond}) {{")
        shown = f"{a.name}" if kind != "Tensor[]" else f"in {a.name}"
        lhs_dev = (f"{a.name}->device().toString()" if kind == "Tensor?"
                   else f"{a.name}.device().toString()")
        lines.append(
            '            TP_THROW(DeviceMismatchError, "Expected all tensors to be on the same device, but found one (' + shown + ') on " + '
            + lhs_dev + ' + " and another (target) on " + ' + target_text + ');')
        lines.append("    }")


def generate_cpp(funcs: list[NativeFunction], *,
                 autocast_ops: set[str], autograd_ops: dict[str, str]) -> str:
    """autograd_ops maps dispatcher op name -> backward node name."""
    lines = [_CPP_INCLUDES.rstrip("\n"), "", "namespace tensorplay {", ""]

    # Pass 1: backend redispatch helpers, one per schema and variant.
    seen_rd = set()
    for f in funcs:
        if f.manual_kernel_registration:
            continue
        for variant in _variant_instances(f):
            rd_name = _redispatch_name(f, variant)
            rd_key = _redispatch_key(f, variant)
            if rd_key in seen_rd:
                continue
            seen_rd.add(rd_key)
            dev_src, _ = _device_source(f, variant)
            _emit_redispatch(lines, f, variant, dev_src, rd_name)

    # Pass 2: public Tensor members.  Deduplicate on parameter types (names,
    # defaults, staticness, and constness never distinguish overloads); the
    # member plan additionally routes ambiguous overload forms and handwritten
    # method variants away from this surface.  Both variants are members of
    # Tensor here: static factories for the function surface, instance
    # methods otherwise.
    member_plan = plan_members(funcs)
    seen_def = set()
    for f in funcs:
        for variant in _variant_instances(f):
            def_key = _overload_key(f, variant)
            if def_key in seen_def:
                continue
            seen_def.add(def_key)
            if not member_plan.get((f.func_name, variant), True):
                continue
            dev_src, target_dev = _device_source(f, variant)

            rd_name = _redispatch_name(f, variant)

            # ---- public entry ----
            # Both variants are members of Tensor here: static factories for
            # the function surface, instance methods otherwise.
            lines.append(method_signature(f, variant, declaration=False,
                                          qualified=True) + " {")

            if f.cpp_name == "copy_":
                lines.append('    if (!impl_ || !src.impl_) TP_THROW(RuntimeError, "Tensor not defined");')
                lines.append("    if (this->shape() != src.shape()) {")
                lines.append('    TP_THROW(RuntimeError, "copy_(): shapes mismatch (broadcasting not yet supported)");')
                lines.append("    }")

            if f.cpp_name == "contiguous":
                # contiguous): resolve contiguity BEFORE any dispatch. When the
                # input is already contiguous the input tensor itself is
                # returned; otherwise internal kernel call sites (e.g. the
                # unary pointwise kernels) would route through the autograd
                # `contiguous` wrapper, which tags the aliased input with a
                # ContiguousBackward node and corrupts the graph (leaf inputs
                # would silently become non-leaves).
                self_expr = "(*this)" if variant == "method" else "self"
                lines.append("    {")
                lines.append("        auto __fmt = static_cast<MemoryFormat>(memory_format);")
                lines.append("        if (__fmt == MemoryFormat::Preserve) __fmt = MemoryFormat::Contiguous;")
                lines.append(f"        if ({self_expr}.is_contiguous(__fmt)) return {self_expr};")
                lines.append("    }")

            _emit_device_checks(lines, f, variant, target_dev)

            call_args = []
            for a in f.args:
                if a.name == "requires_grad":
                    continue
                if variant == "method" and a.name == "self":
                    call_args.append("*this")
                else:
                    call_args.append(call_arg_expr(f.base_name, a))
            call_str = ", ".join(call_args)

            ret = cpp_return_type(f)
            ret_void = ret == "void"

            tmpl = [ret] + [stub_arg_type_for(f.base_name, a)
                           for a in f.args if a.name != "requires_grad"]

            # Transform keys must be consumed before autocast or autograd.
            # A wrapped tensor carries the physical backend in its payload,
            # while its public dispatch key identifies the active transform.
            tensor_like_args = [a for a in f.args if a.type.is_tensor_like]
            if tensor_like_args or f.func_name in _RANDOM_TRANSFORM_OPS:
                lines.append("    {")
                lines.append(
                    f"        DispatchKey __tp_key = "
                    f"{_dispatch_key_expr(f, variant, redispatch=False)};")
                lines.append("        if (is_vmap_key(__tp_key)) {")
                lines.append(
                    '            static const OperatorHandle __tp_handle = '
                    f'Dispatcher::singleton().findHandle("{f.func_name}");')
                lines.append(
                    "            if (!__tp_handle || !__tp_handle.getKernel(__tp_key)) {")
                lines.append(
                    f'                TP_THROW(NotImplementedError, "No native batch rule registered for {f.func_name}");')
                lines.append("            }")
                transform_call = (
                    f"DispatchStub<{', '.join(tmpl)}>::call(__tp_handle, "
                    f"__tp_key, {call_str})"
                )
                if ret_void:
                    lines.append(f"            {transform_call};")
                    lines.append("            return;")
                else:
                    lines.append(f"            return {transform_call};")
                lines.append("        }")
                lines.append("    }")

            # Autocast key outranks Autograd (casts precede VariableType).
            if f.func_name in autocast_ops:
                lines.append("    {")
                lines.append(
                    '        static const OperatorHandle __ac_handle = '
                    f'Dispatcher::singleton().findHandle("{f.func_name}");')
                lines.append(
                    f"        DispatchKey __ac_key = toAutocastKey(computeDispatchKey({dev_src}));")
                # is_enabled first: one thread-local load short-circuits the
                # common (disabled) path before touching the dispatch table.
                lines.append(
                    "        if (autocast::is_enabled(__ac_key) && __ac_handle && __ac_handle.getKernel(__ac_key)) {")
                ac_call = f"DispatchStub<{', '.join(tmpl)}>::call(__ac_handle, __ac_key, {call_str})"
                if ret_void:
                    lines.append(f"            {ac_call};")
                    lines.append("            return;")
                else:
                    lines.append(f"            return {ac_call};")
                lines.append("        }")
                lines.append("    }")

            node = autograd_ops.get(f.func_name)
            if node:
                lines.append("    if (GradMode::is_enabled()) {")
                lines.append(
                    '        static const OperatorHandle ag_handle = '
                    f'Dispatcher::singleton().findHandle("{f.func_name}");')
                lines.append(
                    f"        DispatchKey ag_key = toAutogradKey(computeDispatchKey({dev_src}));")
                lines.append("        if (ag_handle && ag_handle.getKernel(ag_key)) {")
                ag_call = f"DispatchStub<{', '.join(tmpl)}>::call(ag_handle, ag_key, {call_str})"
                if ret_void:
                    lines.append(f"            {ag_call};")
                    lines.append("            return;")
                else:
                    lines.append(f"            return {ag_call};")
                lines.append("        }")
                lines.append("    }")

            tail = f"detail::{rd_name}({call_str})"
            lines.append(f"    {'return ' + tail + ';' if not ret_void else tail + ';'}")
            lines.append("}")
            lines.append("")

            scalar_dim_arg = _scalar_dim_overload_arg(f)
            if scalar_dim_arg is not None:
                scalar_def_key = _overload_key(
                    f, variant, scalar_dim=True)
                if scalar_def_key not in seen_def:
                    seen_def.add(scalar_def_key)
                    lines.append(method_signature(
                        f, variant, declaration=False, qualified=True,
                        scalar_dim=True) + " {")
                    scalar_call_args = []
                    for a in f.args:
                        if a.name == "requires_grad":
                            continue
                        if variant == "method" and a.name == "self":
                            scalar_call_args.append("*this")
                        elif a is scalar_dim_arg:
                            scalar_call_args.append("std::vector<int64_t>{dim}")
                        else:
                            scalar_call_args.append(
                                call_arg_expr(f.base_name, a))
                    scalar_call = ", ".join(scalar_call_args)
                    qualifier = "Tensor::"
                    if ret_void:
                        lines.append(
                            f"    {qualifier}{f.cpp_name}({scalar_call});")
                        lines.append("    return;")
                    else:
                        lines.append(
                            f"    return {qualifier}{f.cpp_name}({scalar_call});")
                    lines.append("}")
                    lines.append("")

    lines.append("} // namespace tensorplay")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TensorRedispatchGenerated.h
# ---------------------------------------------------------------------------

def generate_redispatch_header(funcs: list[NativeFunction]) -> str:
    lines = [
        "// Generated by tools/codegen/main.py -- DO NOT EDIT",
        "#pragma once",
        '#include "Tensor.h"',
        '#include "Macros.h"',
        "",
        "namespace tensorplay {",
        "namespace detail {",
        "",
    ]
    seen_rd = set()
    for f in funcs:
        if f.manual_kernel_registration:
            continue
        for variant in _variant_instances(f):
            rd_name = _redispatch_name(f, variant)
            rd_key = _redispatch_key(f, variant)
            if rd_key in seen_rd:
                continue
            seen_rd.add(rd_key)
            ret = cpp_return_type(f)
            rd_args = [f"{stub_arg_type(a.type)} {a.name}"
                       for a in f.args if a.name != "requires_grad"]
            lines.append(
                f"TENSORPLAY_API {ret} {rd_name}({', '.join(rd_args)});")
            lines.append("")
    lines.extend(["} // namespace detail", "} // namespace tensorplay"])
    return "\n".join(lines)
