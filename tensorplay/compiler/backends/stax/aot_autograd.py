"""Ahead-of-time reverse pass.

Reads the derivative expressions, evaluates them against symbolic
values, and emits the native graph that produces every gradient in
one execution.
"""
from __future__ import annotations

import numbers
import re
from typing import Any

from ....graph import GraphModule, Node
from .ir import (
    _AotFusedSpec,
    _AotNativeSymbol,
    _AotNativeTuple,
    _AotShape,
    _AUTOCAST_GEMM_OPS,
    _FUSED_OUT_DTYPE_CODES,
    _FUSED_PROGRAM_OPCODES,
    _adjoint_promotion_deferrable,
    _dtype_width,
    _NativeLowering,
    _nodes,
    _set_scalar_attr,
    _spatial_int_list,
    _target_name,
    _traced_value,
)
from .loops import dtype_name, promotes_on_load
from .lowering import _lower_native
from .pointwise import _register_stax_cuda_pointwise_op


class _AotNativeGraphBuilder:
    """Small native-IR builder used by the source-derived reverse pass.

    Elementwise work is not emitted one node at a time: binary/unary calls
    accumulate into a program buffer that is emitted as a single fused
    pointwise node whenever a non-elementwise op, an output, or an operand
    outside the buffer needs materializing.  One fused node becomes one
    kernel launch, so derivative-formula chains stop paying per-op
    dispatch and memory round trips.
    """

    def __init__(self, native_module: Any):
        self.native_module = native_module
        self.graph = native_module.Graph()
        self._literal_symbols: dict[int, _AotNativeSymbol] = {}
        # Elementwise program buffer (empty when idle).
        self._fused_ops: list[tuple[int, Any, Any]] = []
        self._fused_temp_pos: dict[int, int] = {}
        self._fused_temp_symbols: dict[int, _AotNativeSymbol] = {}
        self._fused_shape: tuple[int, ...] | None = None
        self._fused_dtype: Any = None
        self._fused_pending: dict[int, tuple[_AotFusedSpec, int]] = {}
        # Element type overrides for buffered results: a conversion edge
        # consumed only by a kernel that wants another width rides the
        # producer's store instead of a standalone conversion node.
        self._fused_out_dtypes: dict[int, Any] = {}
        self._cuda_examples: dict[Any, Any] = {}
        self._cast_memo: dict[tuple[int, Any], tuple[_AotNativeSymbol, _AotNativeSymbol]] = {}
        # Conversions into the arithmetic width that only a program consumer
        # can apply on load: symbol id -> (symbol, requested width).
        self._deferred_casts: dict[int, tuple[_AotNativeSymbol, Any]] = {}

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in getattr(value, "shape", ()))

    @staticmethod
    def _symbol(value: Any) -> _AotNativeSymbol | None:
        return value if isinstance(value, _AotNativeSymbol) else None

    def input(self, example_value: Any) -> _AotNativeSymbol:
        symbol = _AotNativeSymbol(self, self.graph.add_input(), self._shape(example_value))
        dtype = getattr(example_value, "dtype", None)
        if dtype is not None:
            symbol.dtype = dtype
        try:
            if example_value.device.is_cuda():
                self._cuda_examples.setdefault(dtype, example_value)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return symbol

    def literal(self, value: Any) -> _AotNativeSymbol:
        """Lift a captured tensor constant into the backward graph inputs."""
        key = id(value)
        symbol = self._literal_symbols.get(key)
        if symbol is None:
            symbol = self.input(value)
            self._literal_symbols[key] = symbol
        return symbol

    def _add_inputs(self, native_node: Any, args: tuple[Any, ...]) -> list[_AotNativeSymbol]:
        symbols: list[_AotNativeSymbol] = []
        for value in args:
            if not isinstance(value, _AotNativeSymbol):
                raise TypeError("AOT native op received a non-Tensor argument")
            self._materialize(value)
            native_node.add_input(value.value)
            symbols.append(value)
        return symbols

    # -- elementwise program buffer -----------------------------------------

    def _materialize(self, symbol: _AotNativeSymbol) -> None:
        """Give a buffered or spilled elementwise result a real graph value.

        A temporary still in the buffer flushes the program with only that
        result emitted; sibling intermediates spill and stay dormant until a
        consumer pulls them.  A spilled temporary is re-emitted as its own
        node, so a shared subexpression consumed by a later formula pays for
        that single result instead of forcing every partial sum of the
        program through memory.
        """
        if symbol is None or symbol.value is not None:
            return
        index = self._fused_temp_pos.get(id(symbol))
        if index is not None:
            self._fused_flush([index])
            return
        entry = self._fused_pending.get(id(symbol))
        if entry is not None:
            spec, index = entry
            del self._fused_pending[id(symbol)]
            self._emit_program(spec.ops, spec.temps, [index], spec.out_dtypes)
            return
        raise RuntimeError("AOT elementwise buffer lost a temporary symbol")

    def _promote_fused_dtype(self, dtype: Any) -> None:
        """Widen the program's result element type to a wider operand.

        Buffered results already carry the previous width in their symbols;
        they hold no data yet, so retargeting the symbols keeps every later
        consumer reading the true stored type.  Results an explicit store
        override has claimed keep their claimed width.
        """
        previous = self._fused_dtype
        self._fused_dtype = dtype
        for symbol in self._fused_temp_symbols.values():
            if symbol.dtype == previous:
                symbol.dtype = dtype

    def _fused_append(
        self, opcode: int, lhs: tuple[str, Any], rhs: tuple[str, Any]
    ) -> _AotNativeSymbol:
        index = len(self._fused_ops)
        self._fused_ops.append((opcode, lhs, rhs))
        symbol = _AotNativeSymbol(self, None, self._fused_shape, self._fused_dtype)
        self._fused_temp_pos[id(symbol)] = index
        self._fused_temp_symbols[id(symbol)] = symbol
        return symbol

    def _fused_try_elementwise(
        self,
        op_name: str,
        lhs_symbol: _AotNativeSymbol | None,
        rhs_symbol: _AotNativeSymbol | None,
        lhs_raw: Any,
        rhs_raw: Any,
        *,
        unary: bool = False,
    ) -> _AotNativeSymbol | None:
        if op_name == "conj":
            dtype = getattr(lhs_symbol, "dtype", None)
            if dtype is None or "complex" in str(dtype):
                return None
            op_name = "pos"
        opcode = _FUSED_PROGRAM_OPCODES.get(op_name)
        if opcode is None:
            return None
        if unary:
            operands = (("sym", lhs_symbol), ("const", 0.0))
            shape = tuple(lhs_symbol.shape)
        elif lhs_symbol is not None and rhs_symbol is not None:
            try:
                shape = self._broadcast_shape(lhs_symbol.shape, rhs_symbol.shape)
            except ValueError:
                return None
            operands = (("sym", lhs_symbol), ("sym", rhs_symbol))
        elif lhs_symbol is not None:
            operands = (("sym", lhs_symbol), ("const", float(rhs_raw)))
            shape = tuple(lhs_symbol.shape)
        elif rhs_symbol is not None:
            operands = (("const", float(lhs_raw)), ("sym", rhs_symbol))
            shape = tuple(rhs_symbol.shape)
        else:
            return None
        for kind, value in operands:
            if kind == "sym" and value.value is None:
                if id(value) in self._fused_temp_pos:
                    continue
                if id(value) not in self._fused_pending:
                    return None  # operand of a dropped program: fall back
                # A spilled temp can join this buffer once it is revived as
                # a real input value.
                self._materialize(value)
        # The fused evaluator widens every input to one arithmetic width on
        # load, so operands of different storage sizes share a program as
        # long as they stay in the same precision family; float64 runs its
        # own family because the arithmetic width follows the widest operand.
        # A wider operand promotes the element type of every program result.
        for kind, value in operands:
            if kind == "sym" and value.dtype is not None:
                if self._fused_dtype is None:
                    self._fused_dtype = value.dtype
                else:
                    names = {str(value.dtype), str(self._fused_dtype)}
                    if sum("float64" in name for name in names) == 1:
                        return None
                    if _dtype_width(value.dtype) > _dtype_width(
                        self._fused_dtype
                    ):
                        self._promote_fused_dtype(value.dtype)
        if self._fused_shape is None:
            self._fused_shape = shape
        elif shape != self._fused_shape:
            # Generation change: nothing here is emitted yet, so spill the
            # whole generation and let consumers pull results on demand.
            self._fused_flush([])
            self._fused_shape = shape
        return self._fused_append(opcode, operands[0], operands[1])

    def _fused_flush(self, wanted: list[int] | None = None) -> None:
        """Retire the buffer generation; emit only the requested results.

        Results nobody asked for spill into the pending registry and stay
        dormant until a consumer pulls them; results nobody ever pulls cost
        nothing at all.
        """
        if not self._fused_ops:
            return
        ops = self._fused_ops
        temp_pos = self._fused_temp_pos
        temp_symbols = self._fused_temp_symbols
        promoted = self._fused_dtype
        self._fused_ops = []
        self._fused_temp_pos = {}
        self._fused_temp_symbols = {}
        self._fused_shape = None
        self._fused_dtype = None

        temps = tuple(
            temp_symbols[temp_id]
            for temp_id, _ in sorted(temp_pos.items(), key=lambda kv: kv[1])
        )
        # Every result names its stored element type: unclaimed temps take
        # the program's promoted width, so a program mixing input widths
        # never depends on the first input's storage type as a default.
        out_dtypes = {
            temp_pos[temp_id]: self._fused_out_dtypes.get(temp_id, promoted)
            for temp_id in temp_pos
        }
        self._fused_out_dtypes = {}
        spec = _AotFusedSpec(tuple(ops), temps, out_dtypes)
        emitted = list(range(len(ops))) if wanted is None else list(wanted)
        for index, symbol in enumerate(temps):
            if index not in emitted:
                self._fused_pending[id(symbol)] = (spec, index)
        if emitted:
            self._emit_program(spec.ops, temps, emitted, spec.out_dtypes)

    def _emit_program(
        self,
        ops: tuple[tuple[int, Any, Any], ...],
        temps: tuple[_AotNativeSymbol, ...],
        wanted: list[int],
        out_dtypes: dict[int, Any] | None = None,
    ) -> None:
        temp_pos = {id(symbol): index for index, symbol in enumerate(temps)}
        # Dependency closure of the requested results, in program order; the
        # closure is the whole program only when every temp is wanted.  A
        # single-result emit moves that result last because the one-output
        # evaluator returns the final temp of the program.
        closure: set[int] = set()
        stack = list(wanted)
        while stack:
            index = stack.pop()
            if index in closure:
                continue
            closure.add(index)
            for kind, value in ops[index][1:]:
                if kind == "sym":
                    position = temp_pos.get(id(value))
                    if position is not None:
                        stack.append(position)
        sequence = sorted(closure)
        if len(wanted) == 1:
            sequence.remove(wanted[0])
            sequence.append(wanted[0])
        mapping = {index: position for position, index in enumerate(sequence)}

        input_symbols: list[_AotNativeSymbol] = []
        input_pos: dict[int, int] = {}
        constants: list[float] = []
        const_pos: dict[float, int] = {}
        for index in sequence:
            for kind, value in ops[index][1:]:
                if kind == "sym":
                    if id(value) in temp_pos:
                        continue  # program-internal temporary
                    if id(value) in input_pos:
                        continue
                    input_pos[id(value)] = len(input_symbols)
                    input_symbols.append(value)
                    continue
                number = float(value)
                if number not in const_pos:
                    const_pos[number] = len(constants)
                    constants.append(number)
        input_count = len(input_symbols)
        program: list[int] = []
        for index in sequence:
            opcode, lhs, rhs = ops[index]
            program.append(opcode)
            for kind, value in (lhs, rhs):
                if kind == "const":
                    program.append(-const_pos[value] - 1)
                elif id(value) in temp_pos:
                    program.append(input_count + mapping[temp_pos[id(value)]])
                else:
                    program.append(input_pos[id(value)])

        emitted = list(wanted)
        output_refs = [input_count + mapping[index] for index in emitted]
        op_name = None
        if not out_dtypes and 1 <= input_count <= 32 and len(emitted) <= 32:
            first_shape = tuple(input_symbols[0].shape)
            first_dtype = input_symbols[0].dtype
            uniform = all(
                tuple(symbol.shape) == first_shape
                and symbol.dtype == first_dtype
                for symbol in input_symbols
            )
            if uniform:
                example = self._cuda_examples.get(first_dtype)
                if (
                    example is not None
                    and tuple(int(item) for item in example.shape) == first_shape
                    and example.dtype == first_dtype
                ):
                    op_name = _register_stax_cuda_pointwise_op(
                        program,
                        constants,
                        tuple(output_refs),
                        input_count,
                        [example] * input_count,
                        (repr(first_dtype),) * len(output_refs),
                    )
                else:
                    op_name = _register_stax_cuda_pointwise_op(
                        program,
                        constants,
                        tuple(output_refs),
                        input_count,
                        [],
                    )
        node = self.graph.create_node(
            "custom_op" if op_name is not None else "fused_pointwise",
            f"aot_fused_pointwise_{len(self.graph.nodes)}",
        )
        for symbol in input_symbols:
            node.add_input(symbol.value)
        if op_name is None:
            node.set_int_attr("input_count", input_count)
            node.set_ints_attr("program", program)
            node.set_floats_attr("constants", constants)
            node.set_ints_attr("output_refs", output_refs)
        else:
            node.set_str_attr("op_name", op_name)

        outputs = [node.add_output() for _ in emitted]
        if out_dtypes:
            codes = []
            for index in emitted:
                dtype = out_dtypes.get(index)
                name = (
                    str(dtype).rsplit(".", 1)[-1] if dtype is not None else None
                )
                codes.append(_FUSED_OUT_DTYPE_CODES.get(name, -1))
            if any(code >= 0 for code in codes):
                node.set_ints_attr("output_dtypes", codes)
        node_to_output = dict(zip(emitted, outputs))
        for position, output in node_to_output.items():
            temps[position].value = output

    @staticmethod
    def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for left, right in zip(reversed(lhs), reversed(rhs)):
            if left != right and left != 1 and right != 1:
                raise ValueError(f"incompatible AOT shapes: {lhs} and {rhs}")
            result.append(max(left, right))
        longer = lhs if len(lhs) >= len(rhs) else rhs
        result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
        return tuple(reversed(result))

    def binary(self, op_name: str, lhs: Any, rhs: Any) -> _AotNativeSymbol:
        lhs_symbol = self._symbol(lhs)
        rhs_symbol = self._symbol(rhs)
        if lhs_symbol is None and hasattr(lhs, "shape"):
            lhs_symbol = self.literal(lhs)
        if rhs_symbol is None and hasattr(rhs, "shape"):
            rhs_symbol = self.literal(rhs)
        if lhs_symbol is None and rhs_symbol is None:
            if op_name == "add":
                return lhs + rhs
            if op_name == "sub":
                return lhs - rhs
            if op_name == "mul":
                return lhs * rhs
            if op_name == "div":
                return lhs / rhs
            raise NotImplementedError(f"AOT scalar operation is unsupported: {op_name}")
        fused = self._fused_try_elementwise(
            op_name, lhs_symbol, rhs_symbol, lhs, rhs
        )
        if fused is not None:
            return fused
        # Flush pending elementwise programs before the consumer node exists:
        # a program flushed afterwards would execute after this node reads it.
        if lhs_symbol is not None:
            self._materialize(lhs_symbol)
        if rhs_symbol is not None:
            self._materialize(rhs_symbol)
        if lhs_symbol is not None:
            lhs_symbol = self._force_cast(lhs_symbol)
        if rhs_symbol is not None:
            rhs_symbol = self._force_cast(rhs_symbol)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        shape = lhs_symbol.shape if lhs_symbol is not None else rhs_symbol.shape
        if lhs_symbol is not None and rhs_symbol is not None:
            native_node.add_input(lhs_symbol.value)
            native_node.add_input(rhs_symbol.value)
            shape = self._broadcast_shape(lhs_symbol.shape, rhs_symbol.shape)
        else:
            symbol = lhs_symbol if lhs_symbol is not None else rhs_symbol
            scalar = rhs if lhs_symbol is not None else lhs
            native_node.add_input(symbol.value)
            _set_scalar_attr(native_node, scalar, 1 if lhs_symbol is not None else 0)
        result = _AotNativeSymbol(self, native_node.add_output(), shape)
        result.dtype = (lhs_symbol if lhs_symbol is not None else rhs_symbol).dtype
        return result

    def unary(
        self,
        op_name: str,
        value: _AotNativeSymbol,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> _AotNativeSymbol:
        fused = self._fused_try_elementwise(
            op_name, value, None, None, None, unary=True
        )
        if fused is not None:
            return fused
        self._materialize(value)
        value = self._force_cast(value)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        native_node.add_input(value.value)
        result = _AotNativeSymbol(self, native_node.add_output(), shape or value.shape)
        result.dtype = value.dtype
        return result

    def helper(
        self,
        op_name: str,
        args: tuple[_AotNativeSymbol, ...],
        *,
        attrs: dict[str, Any] | None = None,
        shape: tuple[int, ...] | None = None,
        outputs: int = 1,
        output_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> _AotNativeSymbol | _AotNativeTuple:
        # Pending elementwise programs must become nodes before the consumer:
        # the native executor walks creation order, so a program flushed after
        # this node would still be undefined when this node reads it.
        for value in args:
            self._materialize(value)
        args = tuple(self._force_cast(value) for value in args)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        symbols = self._add_inputs(native_node, args)
        del symbols
        for key, value in (attrs or {}).items():
            if isinstance(value, bool) or isinstance(value, int):
                native_node.set_int_attr(key, int(value))
            elif isinstance(value, numbers.Real):
                native_node.set_float_attr(key, float(value))
            elif isinstance(value, str):
                native_node.set_str_attr(key, value)
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                native_node.set_ints_attr(key, [int(item) for item in value])
            else:
                raise TypeError(f"unsupported AOT native attribute: {key}={value!r}")
        if outputs == 1:
            result = _AotNativeSymbol(
                self, native_node.add_output(), shape or args[0].shape
            )
            result.dtype = args[0].dtype if args else None
            return result
        result = _AotNativeTuple(
            tuple(
                _AotNativeSymbol(
                    self,
                    native_node.add_output(),
                    output_shapes[index]
                    if output_shapes is not None and index < len(output_shapes)
                    else shape or args[0].shape,
                    args[0].dtype if args else None,
                )
                for index in range(outputs)
            )
        )
        result.node = native_node
        return result

    def reshape(self, value: _AotNativeSymbol, shape: Any) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper("reshape", (value,), attrs={"shape": normalized}, shape=normalized)  # type: ignore[return-value]

    def _deferrable_widening(self, from_dtype: Any, to_dtype: Any) -> bool:
        """Whether a conversion can ride the consumer's load-time promotion.

        Which storage kinds a load lifts into the arithmetic width is decided
        once, next to that width; a request is deferrable when it asks for
        exactly the width the lift produces.  Every other pair is a real change
        of storage, and a consumer that names a storage type instead of an
        arithmetic width has no load step to fold it into.
        """
        if from_dtype is None or to_dtype is None:
            return False
        return promotes_on_load(from_dtype) and dtype_name(to_dtype) == "float32"

    def _force_cast(self, symbol: _AotNativeSymbol) -> _AotNativeSymbol:
        """Give a load-widened value real storage of its declared width.

        Native operators, graph outputs and saved values name a storage type
        and read the descriptor as it stands, so a conversion left pending
        for a program has to become a real node before they see it.
        """
        entry = self._deferred_casts.get(id(symbol))
        if entry is None:
            return symbol
        del self._deferred_casts[id(symbol)]
        _, dtype = entry
        name = str(dtype).rsplit(".", 1)[-1]
        result = self.helper(
            "cast", (symbol,), attrs={"dtype": name}, shape=symbol.shape
        )
        if isinstance(result, _AotNativeSymbol):
            result.dtype = dtype
        return result

    def cast(self, value: _AotNativeSymbol, dtype: Any) -> _AotNativeSymbol:
        """Convert a symbol to ``dtype`` (no-op when already in that type).

        Conversions are memoized per (symbol, dtype): one captured value
        feeding several consuming kernels (e.g. a convolution gradient split
        into input/weight/bias contributions) is converted once and the
        converted value is shared.
        """
        if dtype is None or value.dtype == dtype:
            return value
        memo_key = (id(value), dtype)
        memo = self._cast_memo.get(memo_key)
        if memo is not None:
            return memo[1]
        index = self._fused_temp_pos.get(id(value))
        pending = None if index is not None else self._fused_pending.get(id(value))
        if index is not None or pending is not None:
            # The buffered or spilled result has not been emitted yet: its
            # storage type can still be retargeted, so the producer stores
            # the converted value directly and no conversion node exists.
            if index is not None:
                current = self._fused_out_dtypes.get(id(value), value.dtype)
            else:
                spec, index = pending
                current = spec.out_dtypes.get(index)
            if current is None or current == dtype:
                if index is not None:
                    self._fused_out_dtypes[id(value)] = dtype
                else:
                    pending[0].out_dtypes[index] = dtype
                value.dtype = dtype
                self._cast_memo[memo_key] = (value, value)
                return value
            # A different width was already claimed for this result: fall
            # through to a standalone conversion after materializing.
            self._materialize(value)
        if self._deferrable_widening(value.dtype, dtype):
            # The value now has a graph value, so its storage type is fixed.
            # A program consumer widens it on load anyway, so hand out a
            # symbol that declares the requested width and let the consumers
            # that name a storage type force the conversion instead.
            widened = _AotNativeSymbol(self, value.value, value.shape, dtype)
            self._deferred_casts[id(widened)] = (widened, dtype)
            self._cast_memo[memo_key] = (value, widened)
            return widened
        name = str(dtype).rsplit(".", 1)[-1]
        result = self.helper("cast", (value,), attrs={"dtype": name}, shape=value.shape)
        if isinstance(result, _AotNativeSymbol):
            result.dtype = dtype
            # The entry keeps the source symbol alive: the key is its id(),
            # and ids are only stable while the object is referenced.
            self._cast_memo[memo_key] = (value, result)
            return result
        return result

    def zeros_like(
        self, template: _AotNativeSymbol, shape: Any
    ) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper(
            "zeros_like", (template,), attrs={"shape": normalized}, shape=normalized
        )  # type: ignore[return-value]

    def expand(self, value: _AotNativeSymbol, shape: Any) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper("expand", (value,), attrs={"shape": normalized}, shape=normalized)  # type: ignore[return-value]

    def unsqueeze(self, value: _AotNativeSymbol, dim: int) -> _AotNativeSymbol:
        rank = len(value.shape)
        normalized_dim = int(dim)
        if normalized_dim < 0:
            normalized_dim += rank + 1
        if normalized_dim < 0 or normalized_dim > rank:
            raise ValueError(
                f"AOT unsqueeze dimension {dim} is out of range for rank {rank}"
            )
        shape = value.shape[:normalized_dim] + (1,) + value.shape[normalized_dim:]
        return self.helper(
            "unsqueeze", (value,), attrs={"dim": normalized_dim}, shape=shape
        )  # type: ignore[return-value]

    def squeeze(
        self, value: _AotNativeSymbol, dim: Any = None
    ) -> _AotNativeSymbol:
        if dim is None:
            shape = tuple(item for item in value.shape if item != 1)
            attrs: dict[str, Any] = {}
        else:
            normalized_dim = int(dim)
            if normalized_dim < 0:
                normalized_dim += len(value.shape)
            if (
                normalized_dim < 0
                or normalized_dim >= len(value.shape)
                or value.shape[normalized_dim] != 1
            ):
                raise ValueError("AOT squeeze dimension is not singleton")
            shape = value.shape[:normalized_dim] + value.shape[normalized_dim + 1 :]
            attrs = {"dim": normalized_dim}
        return self.helper("squeeze", (value,), attrs=attrs, shape=shape)  # type: ignore[return-value]

    def cat(
        self, values: tuple[_AotNativeSymbol, ...], dim: int, shape: Any
    ) -> _AotNativeSymbol:
        if not values:
            raise ValueError("AOT cat requires at least one value")
        return self.helper(
            "cat", values, attrs={"dim": int(dim)}, shape=tuple(int(item) for item in shape)
        )  # type: ignore[return-value]

    def split(
        self,
        value: _AotNativeSymbol,
        sizes: tuple[int, ...],
        dim: int,
        shapes: tuple[tuple[int, ...], ...],
    ) -> _AotNativeTuple:
        result = self.helper(
            "split",
            (value,),
            attrs={"split_sizes": sizes, "dim": int(dim)},
            shape=shapes[0] if shapes else value.shape,
            outputs=len(shapes),
        )
        if not isinstance(result, _AotNativeTuple):
            raise TypeError("AOT split did not produce multiple outputs")
        for symbol, shape in zip(result.values, shapes):
            symbol.shape = _AotShape(shape)
        return result

    def sum(
        self, value: _AotNativeSymbol, dim: Any = None, keepdim: bool = False
    ) -> _AotNativeSymbol:
        if dim is None:
            return self.helper("sum", (value,), shape=())  # type: ignore[return-value]
        dims = tuple(int(item) for item in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        normalized_dims = tuple(item if item >= 0 else item + len(value.shape) for item in dims)
        shape = list(value.shape)
        if keepdim:
            for item in normalized_dims:
                shape[item] = 1
        else:
            for item in sorted(normalized_dims, reverse=True):
                shape.pop(item)
        return self.helper(
            "sum",
            (value,),
            attrs={"dim": normalized_dims, "keepdim": bool(keepdim)},
            shape=tuple(shape),
        )  # type: ignore[return-value]

def _aot_decomposition_functions() -> dict[str, Any]:
    """Registered decompositions, indexed by the operator name a formula uses.

    The reverse pass evaluates derivative expressions against symbolic values,
    so an operator with a registered decomposition is expressed the way its
    decomposition body does: through the operators that body dispatches.  This
    is the one table both the export path and this lowering read.
    """
    try:
        from ...._decomp import decomposition_table
        import tensorplay._decomp.decompositions  # noqa: F401 - fills the table
    except (ImportError, ModuleNotFoundError, AttributeError):
        return {}
    functions: dict[str, Any] = {}
    for overload, function in decomposition_table.items():
        name = str(overload)
        if "." in name:
            name = name.rsplit(".", 1)[0]
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        functions.setdefault(name, function)
    return functions


def _aot_derivative_specs() -> dict[str, tuple[Any, dict[str, str]]]:
    """Read the local derivative schema used by TensorPlay code generation."""
    from pathlib import Path

    from tools.codegen.model import parse_derivatives_yaml, parse_schema

    yaml_path = Path(__file__).resolve().parents[4] / "config" / "derivatives.yaml"
    result: dict[str, tuple[Any, dict[str, str]]] = {}
    for definition in parse_derivatives_yaml(str(yaml_path)):
        parsed = parse_schema(definition["name"])
        formulas = {
            key: value for key, value in definition.items() if key != "name"
        }
        result[parsed.func_name] = (parsed, formulas)
    return result

def _aot_formula_python(formula: str, tensor_params: set[str]) -> str:
    """Compile one derivatives.yaml formula into a Python expression.

    Shares the codegen expression AST (tokenizer + parser); the emitter
    renders against the runtime formula env -- builder callables like
    add/mul/t/sum plus get_tuple -- instead of the C++ text the generated
    autograd nodes need.
    """
    from tools.codegen.gen_autograd import (
        BinOp, BoolLit, Braced, Call, Method, Neg, Num, StrLit, Var,
        TENSOR_METHODS, parse_expr,
    )

    symbols = set(tensor_params) | {"grad", "grad_output", "result"}

    def is_tensor(expr: Any) -> bool:
        if isinstance(expr, Var):
            return expr.name in symbols
        if isinstance(expr, Neg):
            return looks_tensor(expr.value)
        if isinstance(expr, Method):
            return expr.name.rstrip("_") in TENSOR_METHODS
        if isinstance(expr, Call):
            leaf = expr.callee.split("::")[-1].split("<")[0]
            return leaf not in ("Scalar",)
        if isinstance(expr, BinOp):
            return is_tensor(expr.left) or looks_tensor(expr.right)
        return False

    def looks_tensor(expr: Any) -> bool:
        return is_tensor(expr) or isinstance(expr, BinOp)

    def emit(expr: Any) -> str:
        if isinstance(expr, Num):
            return expr.text
        if isinstance(expr, BoolLit):
            return "True" if expr.text == "true" else "False"
        if isinstance(expr, StrLit):
            return expr.text
        if isinstance(expr, Var):
            if expr.name in {"true", "false"}:
                return "True" if expr.name == "true" else "False"
            if expr.name in {"std::nullopt", "c10::nullopt"}:
                return "None"
            return expr.name
        if isinstance(expr, Neg):
            inner = emit(expr.value)
            return f"neg({inner})" if looks_tensor(expr.value) else f"-{inner}"
        if isinstance(expr, Braced):
            # Python target: a braced list renders as a tuple (builder.sum
            # dims, reshape shapes), matching _aot_default_value.
            items = [emit(item) for item in expr.items]
            if len(items) == 1:
                return f"({items[0]},)"
            return f"({', '.join(items)})"
        if isinstance(expr, Call):
            args = ", ".join(emit(a) for a in expr.args)
            get = re.fullmatch(r"std::get<(\d+)>", expr.callee)
            if get:
                return f"get_tuple({get.group(1)}, {args})"
            if expr.callee in {"std::nullopt", "c10::nullopt"}:
                return "None"
            callee = expr.callee.split("::")[-1]
            return f"{callee}({args})"
        if isinstance(expr, Method):
            recv = emit(expr.receiver)
            args = ", ".join(emit(a) for a in expr.args)
            name = expr.name
            if name in {"dtype", "scalar_type"}:
                return "None"
            base = name[:-1] if name.endswith("_") and name[:-1] in TENSOR_METHODS else name
            if base in TENSOR_METHODS:
                return f"{TENSOR_METHODS[base]}({recv}, {args})" if args \
                    else f"{TENSOR_METHODS[base]}({recv})"
            return f"{recv}.{name}({args})" if args else f"{recv}.{name}()"
        if isinstance(expr, BinOp):
            left = emit(expr.left)
            right = emit(expr.right)
            left_tensor = is_tensor(expr.left)
            right_tensor = looks_tensor(expr.right)
            if expr.op in "+-" and left_tensor:
                return f"{'add' if expr.op == '+' else 'sub'}({left}, {right})"
            if expr.op == "*" and left_tensor:
                return f"mul({left}, {right})"
            if expr.op == "/" and left_tensor:
                return f"div({left}, {right})"
            if expr.op == "*" and right_tensor:
                return f"mul({right}, {left})"
            if expr.op == "-" and right_tensor:
                return f"neg(sub({right}, {left}))"
            return f"({left} {expr.op} {right})"
        raise NotImplementedError(f"AOT formula node is unsupported: {expr!r}")

    return emit(parse_expr(formula))

def _build_aot_formula_env(
    builder: _AotNativeGraphBuilder,
    *,
    batch_norm_cache: dict[tuple[int, ...], _AotNativeTuple],
    tuple_op_cache: dict[tuple[int, ...], _AotNativeTuple],
    autocast_state: dict[str, Any],
    attention_lse: dict[int, tuple[_AotNativeSymbol, _AotNativeSymbol]] | None = None,
) -> dict[str, Any]:
    import tensorplay

    def binary(name: str):
        return lambda lhs, rhs: builder.binary(name, lhs, rhs)

    # Mixed-precision execution: the captured graph runs its matrix products
    # in the reduced precision selected at dispatch time, while the derivative
    # expressions receive gradients in the accumulation type.  Every matrix
    # product of the reverse pass therefore narrows its operands back to the
    # reduced element type before the kernel runs.
    def _narrow(value: Any) -> Any:
        dtype = autocast_state.get("dtype")
        if not isinstance(value, _AotNativeSymbol):
            return value
        return builder.cast(value, dtype)

    def matmul_binary(name: str):
        def invoke(lhs: Any, rhs: Any) -> _AotNativeSymbol:
            return builder.binary(name, _narrow(lhs), _narrow(rhs))

        return invoke

    def unary(name: str):
        return lambda value: builder.unary(name, value)

    def _sum_dim_backward_env(grad, self_value, dim, keepdim):
        dims = dim if isinstance(dim, (list, tuple)) else [dim]
        normalized = sorted(int(item) for item in dims)
        if not keepdim:
            for item in normalized:
                grad = builder.unsqueeze(grad, item)
        return builder.expand(
            grad, tuple(int(item) for item in self_value.shape)
        )

    def tanh_backward_env(grad, output_value):
        # grad * (1 - output^2).conj(): identical to the decomposition the
        # dispatcher uses, restated over builder primitives.
        inner = builder.binary("mul", output_value, output_value)
        return builder.binary(
            "mul", grad, builder.unary("conj", builder.binary("sub", 1, inner))
        )

    def get_tuple(index: int, value: Any):
        # A builder helper returns its ports as a tuple value; a registered
        # decomposition returns a plain tuple.  Either answers a port select.
        if isinstance(value, _AotNativeTuple):
            return value.values[int(index)]
        return tuple(value)[int(index)]

    def batch_norm_backward(*args: Any):
        key = tuple(id(item) if isinstance(item, _AotNativeSymbol) else hash(repr(item)) for item in args)
        cached = batch_norm_cache.get(key)
        if cached is not None:
            return cached
        grad, input_value, weight, running_mean, running_var, training, eps = args
        tensor_args = (grad, input_value)
        attrs = {
            "has_weight": weight is not None,
            "has_running_mean": running_mean is not None,
            "has_running_var": running_var is not None,
            "training": bool(training),
            "eps": float(eps),
        }
        optional = tuple(item for item in (weight, running_mean, running_var) if item is not None)
        value = builder.helper(
            "batch_norm_backward",
            tensor_args + optional,
            attrs=attrs,
            shape=input_value.shape,
            outputs=3,
        )
        assert isinstance(value, _AotNativeTuple)
        batch_norm_cache[key] = value
        return value

    def _shared_tuple_key(args: tuple[Any, ...]) -> tuple[int, ...]:
        # One backward tuple must be shared by every gradient slot of the
        # same forward node: the derivative formulas each re-derive the same
        # call, and re-emitting it per slot triples the expensive work.
        return tuple(
            id(item) if isinstance(item, _AotNativeSymbol) else hash(repr(item))
            for item in args
        )

    def group_norm_backward(
        grad, input_value, num_groups, weight=None, bias=None, eps=1e-5
    ):
        # The kernel accumulates in float32 and reads a float32 gradient;
        # a narrower gradient widens once at this kernel boundary instead of
        # on every upstream gradient edge.
        if grad.dtype is not None and _dtype_width(grad.dtype) < _dtype_width(
            tensorplay.float32
        ):
            grad = builder.cast(grad, tensorplay.float32)
        tensor_args = [grad, input_value]
        has_weight = weight is not None
        has_bias = bias is not None
        if has_weight:
            tensor_args.append(weight)
        if has_bias:
            tensor_args.append(bias)
        channels = int(input_value.shape[1])
        key = _shared_tuple_key((grad, input_value, num_groups, weight, bias, eps))
        cached = tuple_op_cache.get(key)
        if cached is not None:
            return cached
        value = builder.helper(
            "group_norm_backward",
            tuple(tensor_args),
            attrs={
                "num_groups": int(num_groups),
                "has_weight": has_weight,
                "has_bias": has_bias,
                "eps": float(eps),
            },
            shape=input_value.shape,
            outputs=3,
            output_shapes=(
                tuple(input_value.shape),
                (channels,),
                (channels,),
            ),
        )
        assert isinstance(value, _AotNativeTuple)
        tuple_op_cache[key] = value
        return value

    def native_group_norm_backward(
        grad,
        input_value,
        mean,
        rstd,
        weight,
        batch,
        channels,
        spatial,
        group,
        output_mask,
    ):
        """The fused reverse pass: one node per normalization site.

        The registered decomposition answers the same name for a backend that
        fuses the coefficient arithmetic into the pointwise program; this entry
        keeps the site on the fused kernel, which is where the row statistics
        and the two partial reductions stay in one launch.
        """
        mask = tuple(bool(item) for item in output_mask)
        if not any(mask):
            return (None, None, None)
        if grad.dtype is not None and _dtype_width(grad.dtype) < _dtype_width(
            tensorplay.float32
        ):
            grad = builder.cast(grad, tensorplay.float32)
        key = _shared_tuple_key((grad, input_value, mean, rstd, weight, mask))
        cached = tuple_op_cache.get(key)
        if cached is not None:
            return cached
        result = builder.helper(
            "native_group_norm_backward",
            (grad, input_value, mean, rstd, weight),
            attrs={
                "N": int(batch),
                "C": int(channels),
                "HxW": int(spatial),
                "group": int(group),
                "output_mask": [int(flag) for flag in mask],
            },
            shape=input_value.shape,
            outputs=3,
            output_shapes=(
                tuple(input_value.shape) if mask[0] else (),
                (int(channels),) if mask[1] else (),
                (int(channels),) if mask[2] else (),
            ),
        )
        tuple_op_cache[key] = result
        return result

    def scaled_dot_product_attention_backward(
        grad, query, key, value, is_causal=False, impl=0
    ):
        # The fused forward hands back the softmax normalizer next to its
        # result.  Reading it here keeps the gradient on the fused route,
        # which consumes those statistics directly instead of rebuilding the
        # whole score matrix first.
        lse_entry = (attention_lse or {}).get(id(query))
        grad = _narrow(grad)
        query = _narrow(query)
        key = _narrow(key)
        value = _narrow(value)
        key_tuple = _shared_tuple_key((grad, query, key, value, is_causal, impl))
        cached = tuple_op_cache.get(key_tuple)
        if cached is not None:
            return cached
        if lse_entry is not None:
            output, logsumexp = lse_entry
            result = builder.helper(
                "_scaled_dot_product_attention_backward_with_lse",
                (grad, query, key, value, output, logsumexp),
                attrs={"is_causal": bool(is_causal), "impl": int(impl)},
                shape=query.shape,
                outputs=3,
                output_shapes=(query.shape, key.shape, value.shape),
            )
            assert isinstance(result, _AotNativeTuple)
            tuple_op_cache[key_tuple] = result
            return result
        result = builder.helper(
            "scaled_dot_product_attention_backward",
            (grad, query, key, value),
            attrs={"is_causal": bool(is_causal), "impl": int(impl)},
            shape=query.shape,
            outputs=3,
            output_shapes=(query.shape, key.shape, value.shape),
        )
        assert isinstance(result, _AotNativeTuple)
        tuple_op_cache[key_tuple] = result
        return result

    def convolution_backward(
        grad,
        input_value,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        del bias_sizes
        mask = tuple(bool(item) for item in output_mask)
        if len(mask) != 3:
            raise ValueError("convolution_backward output_mask must have three entries")
        key = _shared_tuple_key(
            (
                grad,
                input_value,
                weight,
                tuple(stride),
                tuple(padding),
                tuple(dilation),
                bool(transposed),
                tuple(output_padding),
                int(groups),
            )
        )
        cached = tuple_op_cache.get(key)
        if cached is not None:
            if cached.node is not None and cached.mask is not None:
                merged = tuple(left or right for left, right in zip(cached.mask, mask))
                if merged != cached.mask:
                    cached.node.set_ints_attr("output_mask", [int(item) for item in merged])
                    cached.mask = merged
            return cached
        narrowed = (_narrow(grad), _narrow(input_value), _narrow(weight))
        bias_shape = (
            (int(weight.shape[1]) * int(groups),)
            if bool(transposed)
            else (int(weight.shape[0]),)
        )
        result = builder.helper(
            "convolution_backward",
            narrowed,
            attrs={
                "stride": tuple(int(item) for item in stride),
                "padding": tuple(int(item) for item in padding),
                "dilation": tuple(int(item) for item in dilation),
                "transposed": bool(transposed),
                "output_padding": tuple(int(item) for item in output_padding),
                "groups": int(groups),
                "output_mask": tuple(int(item) for item in mask),
            },
            shape=input_value.shape,
            outputs=3,
            output_shapes=(
                tuple(input_value.shape),
                tuple(weight.shape),
                bias_shape,
            ),
        )
        if not isinstance(result, _AotNativeTuple):
            raise TypeError("convolution_backward did not produce three outputs")
        result.mask = mask
        tuple_op_cache[key] = result
        return result

    def _conv_axis_gradient(slot: int, transposed: bool):
        # The captured conv spellings carry their own derivative formulas,
        # each asking for one gradient of the same convolution.  Every one of
        # these names funnels into a masked slot of the shared convolution
        # backward, so one tuple node serves all requested slots.
        def axis_gradient(
            grad,
            input_value,
            weight,
            stride,
            padding,
            dilation,
            groups,
            output_padding=(0, 0),
        ):
            shape = getattr(input_value, "shape", ())
            rank = max(len(tuple(shape)) - 2, 1) if shape else 2

            def pair(value: Any) -> tuple[int, ...]:
                if isinstance(value, (tuple, list)):
                    return tuple(int(item) for item in value)
                return (int(value),) * rank

            mask = tuple(slot == index for index in range(3))
            return get_tuple(
                slot,
                convolution_backward(
                    grad,
                    input_value,
                    weight,
                    None,
                    pair(stride),
                    pair(padding),
                    pair(dilation),
                    transposed,
                    pair(output_padding),
                    groups,
                    mask,
                ),
            )

        return axis_gradient

    def _conv_transpose_axis_gradient(slot: int):
        def axis_gradient(
            grad,
            input_value,
            weight,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        ):
            return _conv_axis_gradient(slot, True)(
                grad,
                input_value,
                weight,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
            )

        return axis_gradient

    conv1d_grad_input = _conv_axis_gradient(0, False)
    conv1d_grad_weight = _conv_axis_gradient(1, False)
    conv1d_grad_bias = _conv_axis_gradient(2, False)
    conv2d_grad_input = _conv_axis_gradient(0, False)
    conv2d_grad_weight = _conv_axis_gradient(1, False)
    conv2d_grad_bias = _conv_axis_gradient(2, False)
    conv3d_grad_input = _conv_axis_gradient(0, False)
    conv3d_grad_weight = _conv_axis_gradient(1, False)
    conv3d_grad_bias = _conv_axis_gradient(2, False)
    conv_transpose1d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose1d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose1d_grad_bias = _conv_transpose_axis_gradient(2)
    conv_transpose2d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose2d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose2d_grad_bias = _conv_transpose_axis_gradient(2)
    conv_transpose3d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose3d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose3d_grad_bias = _conv_transpose_axis_gradient(2)

    def max_pool_backward(grad, input_value, kernel_size, stride, padding, dilation, ceil_mode):
        values = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        # Spatial rank follows the pooling input: a scalar or single-entry
        # parameter applies to every spatial dimension alike.
        input_rank = len(tuple(input_value.shape)) - 2 if hasattr(input_value, "shape") else None
        rank = input_rank if input_rank in (1, 2, 3) else len(values)
        if rank not in (1, 2, 3):
            raise TypeError("max_pool spatial parameters must have one to three entries")

        def spatial_arg(value: Any) -> tuple[int, ...]:
            items = tuple(value) if isinstance(value, (tuple, list)) else (value,)
            if len(items) == 1 and rank > 1:
                items = items * rank
            return items

        return builder.helper(
            f"max_pool{rank}d_backward",
            (grad, input_value),
            attrs={
                "kernel_size": spatial_arg(kernel_size),
                "stride": spatial_arg(stride),
                "padding": spatial_arg(padding),
                "dilation": spatial_arg(dilation),
                "ceil_mode": bool(ceil_mode),
            },
            shape=input_value.shape,
        )

    def adaptive_avg_pool_backward(grad, input_value):
        rank = len(tuple(input_value.shape)) - 2
        if rank not in (1, 2, 3):
            raise TypeError("adaptive_avg_pool input must have one to three spatial dimensions")
        return builder.helper(
            f"adaptive_avg_pool{rank}d_backward",
            (grad, input_value),
            shape=input_value.shape,
        )

    def avg_pool_pair(value: Any, default: tuple[int, int] | None = None) -> tuple[int, int]:
        if value is None:
            if default is None:
                raise TypeError("avg_pool2d requires a spatial parameter")
            return default
        if isinstance(value, bool):
            raise TypeError("avg_pool2d spatial parameters must be integers")
        if isinstance(value, int):
            item = int(value)
            return (item, item)
        if isinstance(value, (tuple, list)):
            if len(value) == 1 and isinstance(value[0], int) and not isinstance(value[0], bool):
                item = int(value[0])
                return (item, item)
            if len(value) == 2 and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                return (int(value[0]), int(value[1]))
        raise TypeError("avg_pool2d spatial parameters must contain one or two integers")

    def avg_pool2d_backward(
        grad,
        input_value,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        values = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        # Spatial rank follows the pooling input, not the kernel tuple: a
        # scalar kernel_size widens across all spatial dimensions of the
        # input, so a 4-D input with kernel_size=2 pools in 2-D.
        input_rank = len(tuple(input_value.shape)) - 2 if hasattr(input_value, "shape") else None
        rank = input_rank if input_rank in (1, 2, 3) else len(values)
        if rank not in (1, 2, 3):
            raise TypeError("avg_pool spatial parameters must have one to three entries")

        def spatial(value: Any, default: tuple[int, ...] | None = None) -> tuple[int, ...]:
            if value is None:
                if default is None:
                    raise TypeError("avg_pool requires a spatial parameter")
                return default
            if isinstance(value, bool):
                raise TypeError("avg_pool spatial parameters must be integers")
            if isinstance(value, int):
                return (int(value),) * rank
            if isinstance(value, (tuple, list)):
                if len(value) == 1 and isinstance(value[0], int) and not isinstance(value[0], bool):
                    return (int(value[0]),) * rank
                if len(value) == rank and all(
                    isinstance(item, int) and not isinstance(item, bool) for item in value
                ):
                    return tuple(int(item) for item in value)
            raise TypeError("avg_pool spatial parameters have the wrong rank")

        kernel = spatial(kernel_size)
        stride_values = spatial(stride, kernel)
        padding_values = spatial(padding, (0,) * rank)
        if not isinstance(ceil_mode, bool) or not isinstance(count_include_pad, bool):
            raise TypeError("avg_pool2d boolean parameters must be bool")
        attrs: dict[str, Any] = {
            "kernel_size": kernel,
            "stride": stride_values,
            "padding": padding_values,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
        }
        if divisor_override is not None:
            if isinstance(divisor_override, bool) or not isinstance(divisor_override, int):
                raise TypeError("avg_pool2d divisor_override must be an integer or None")
            attrs["divisor_override"] = int(divisor_override)
        return builder.helper(
            f"avg_pool{rank}d_backward",
            (grad, input_value),
            attrs=attrs,
            shape=input_value.shape,
        )

    def maybe_multiply(value: Any, factor: Any) -> Any:
        # Derivative-formula helper: multiplication by a unit scalar is
        # skipped; anything else lowers to a native multiply that accepts
        # symbol/symbol or symbol/scalar operands.
        if isinstance(factor, numbers.Real) and factor == 1:
            return value
        if isinstance(value, numbers.Real) and value == 1:
            return factor
        return builder.binary("mul", value, factor)

    def maybe_divide(value: Any, divisor: Any) -> Any:
        if isinstance(divisor, numbers.Real) and divisor == 1:
            return value
        return builder.binary("div", value, divisor)

    def mul_tensor_backward(grad, other, *_args):
        return builder.binary("mul", grad, other)

    def div_tensor_self_backward(grad, other, *_args):
        return builder.binary("div", grad, other)

    def div_tensor_other_backward(grad, self_value, other, *_args):
        numerator = builder.binary(
            "mul", builder.unary("neg", grad), self_value
        )
        denominator = builder.binary("mul", other, other)
        return builder.binary("div", numerator, denominator)

    def threshold_backward(grad, output, threshold):
        return builder.helper(
            "threshold_backward",
            (grad, output),
            attrs={"threshold": threshold},
            shape=grad.shape,
        )

    def index_select_backward(grad, self_value, dim, index):
        return builder.helper(
            "index_select_backward",
            (grad, index),
            attrs={
                "self_sizes": tuple(int(item) for item in self_value.shape),
                "dim": int(dim),
            },
            shape=self_value.shape,
        )

    def gather_backward(grad, self_value, dim, index, sparse_grad=False):
        return builder.helper(
            "gather_backward",
            (grad, self_value, index),
            attrs={"dim": int(dim), "sparse_grad": bool(sparse_grad)},
            shape=self_value.shape,
        )

    def silu_backward(grad, input_value):
        sigmoid = builder.unary("sigmoid", input_value)
        correction = builder.binary(
            "add",
            1,
            builder.binary(
                "mul",
                input_value,
                builder.binary("sub", 1, sigmoid),
            ),
        )
        return builder.binary("mul", grad, builder.binary("mul", sigmoid, correction))

    resolved: dict[str, Any] = {
        "add": binary("add"),
        "sub": binary("sub"),
        "mul": binary("mul"),
        "div": binary("div"),
        # The dim-variety sum backward restates the reduction's broadcast
        # inverse: reinsert the singleton axes the reduction removed, then
        # broadcast the tangent to the input extent.  The generated C++
        # autograd resolves the same-named helper at link time; this entry
        # serves the interpreted reverse-graph builder only.
        "_sum_dim_backward": _sum_dim_backward_env,
        "tanh_backward": tanh_backward_env,
        "matmul": matmul_binary("matmul"),
        "mm": matmul_binary("mm"),
        "neg": unary("neg"),
        "pos": unary("pos"),
        "t": unary("t"),
        "reshape": builder.reshape,
        "expand": builder.expand,
        "squeeze": builder.squeeze,
        "sum": builder.sum,
        "get_tuple": get_tuple,
        "maybe_multiply": maybe_multiply,
        "maybe_divide": maybe_divide,
        "mul_tensor_backward": mul_tensor_backward,
        "div_tensor_self_backward": div_tensor_self_backward,
        "div_tensor_other_backward": div_tensor_other_backward,
        "batch_norm_backward": batch_norm_backward,
        "scaled_dot_product_attention_backward": scaled_dot_product_attention_backward,
        "convolution_backward": convolution_backward,
        "conv1d_grad_input": conv1d_grad_input,
        "conv1d_grad_weight": conv1d_grad_weight,
        "conv1d_grad_bias": conv1d_grad_bias,
        "conv2d_grad_input": conv2d_grad_input,
        "conv2d_grad_weight": conv2d_grad_weight,
        "conv2d_grad_bias": conv2d_grad_bias,
        "conv3d_grad_input": conv3d_grad_input,
        "conv3d_grad_weight": conv3d_grad_weight,
        "conv3d_grad_bias": conv3d_grad_bias,
        "conv_transpose1d_grad_input": conv_transpose1d_grad_input,
        "conv_transpose1d_grad_weight": conv_transpose1d_grad_weight,
        "conv_transpose1d_grad_bias": conv_transpose1d_grad_bias,
        "conv_transpose2d_grad_input": conv_transpose2d_grad_input,
        "conv_transpose2d_grad_weight": conv_transpose2d_grad_weight,
        "conv_transpose2d_grad_bias": conv_transpose2d_grad_bias,
        "conv_transpose3d_grad_input": conv_transpose3d_grad_input,
        "conv_transpose3d_grad_weight": conv_transpose3d_grad_weight,
        "conv_transpose3d_grad_bias": conv_transpose3d_grad_bias,
        "max_pool2d_backward": max_pool_backward,
        "max_pool1d_backward": max_pool_backward,
        "max_pool3d_backward": max_pool_backward,
        "adaptive_avg_pool2d_backward": adaptive_avg_pool_backward,
        "adaptive_avg_pool1d_backward": adaptive_avg_pool_backward,
        "adaptive_avg_pool3d_backward": adaptive_avg_pool_backward,
        "avg_pool2d_backward": avg_pool2d_backward,
        "avg_pool1d_backward": avg_pool2d_backward,
        "avg_pool3d_backward": avg_pool2d_backward,
        "conj": unary("conj"),
        "sin": unary("sin"),
        "cos": unary("cos"),
        "exp": unary("exp"),
        "log": unary("log"),
        "sqrt": unary("sqrt"),
        "rsqrt": unary("rsqrt"),
        "sigmoid": unary("sigmoid"),
        "tanh": unary("tanh"),
        "abs": unary("abs"),
        "sign": unary("sign"),
        "threshold_backward": threshold_backward,
        "silu_backward": silu_backward,
        "group_norm_backward": group_norm_backward,
        "native_group_norm_backward": native_group_norm_backward,
        "index_select_backward": index_select_backward,
        "gather_backward": gather_backward,
    }
    # Registered decompositions answer for every operator without a builder
    # call of its own, so a derivative expression and a decomposition body
    # reach the native graph through the same value space.  The entries above
    # name the ones the reverse pass calls directly, so they take precedence.
    for name, function in _aot_decomposition_functions().items():
        resolved.setdefault(name, function)
    return resolved

#: Composite spellings and the native form the forward canonicalizes to.  The
#: native form names the extents separately and reports values beside its
#: result, which is what the derivative expressions below are written
#: against.
_AOT_NATIVE_SPELLING: dict[str, str] = {"group_norm": "native_group_norm"}


def _native_group_norm_arguments(node: Node, extents: Any) -> list[tuple[str, Any]] | None:
    """Map ``group_norm(x, g, w, b, eps)`` onto the native schema's names."""
    if len(node.args) != 5 or node.kwargs:
        return None
    input_node, num_groups, weight_node, bias_node, eps = node.args
    shape = getattr(extents, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    channels = int(shape[1])
    if channels % int(num_groups):
        return None
    spatial = 1
    for extent in shape[2:]:
        spatial *= int(extent)
    return [
        ("input", input_node),
        ("weight", weight_node),
        ("bias", bias_node),
        ("N", int(shape[0])),
        ("C", channels),
        ("HxW", spatial),
        ("group", int(num_groups)),
        ("eps", float(eps)),
    ]


def _aot_schema_for(
    specs: dict[str, tuple[Any, dict[str, str]]], op_name: str
) -> tuple[Any, dict[str, str]] | None:
    candidates = [op_name]
    if op_name in {"add", "sub", "mul", "div"}:
        candidates.insert(0, f"{op_name}.Tensor")
    for candidate in candidates:
        if candidate in specs:
            return specs[candidate]
    return None

def _aot_default_value(value: Any) -> Any:
    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "{}":
        return ()
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return tuple(int(item.strip()) for item in value[1:-1].split(",") if item.strip())
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

def _aot_add_adjoint(
    builder: _AotNativeGraphBuilder,
    adjoints: dict[Node, _AotNativeSymbol],
    target: Any,
    contribution: Any,
) -> bool:
    if not isinstance(target, Node) or not isinstance(contribution, _AotNativeSymbol):
        return contribution is None
    previous = adjoints.get(target)
    adjoints[target] = contribution if previous is None else builder.binary(
        "add", previous, contribution
    )
    return True

def _build_aot_backward(
    graph_module: GraphModule,
    native_module: Any,
    forward_lowering: _NativeLowering,
    saved_nodes: list[Node],
    runtime_values: dict[Node, Any],
    runtime_cast_values: dict[tuple[Node, Any], Any],
    runtime_inputs: list[Any],
    public_node: Node,
    saved_result_values: dict[Node, tuple[Any, ...]] | None = None,
) -> tuple[Any, list[int]] | None:
    try:
        specs = _aot_derivative_specs()
    except (ImportError, ModuleNotFoundError, OSError):
        # Derivative tooling/config unavailable: AOT lowering is optional;
        # the caller falls back to the non-AOT native path.
        return None
    builder = _AotNativeGraphBuilder(native_module)
    external_nodes = list(graph_module.graph.placeholders) + [
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    ]
    external_symbols: list[_AotNativeSymbol] = []
    forward_symbols: dict[Node, _AotNativeSymbol] = {}
    for index, node in enumerate(external_nodes):
        symbol = builder.input(runtime_inputs[index])
        external_symbols.append(symbol)
        forward_symbols[node] = symbol
    # Literal tensors are lifted by native lowering after placeholders and
    # module attributes.  They are immutable graph inputs and can therefore
    # be shared by every derivative expression without saving activations.
    for value in forward_lowering.constant_values:
        builder.literal(value)
    saved_symbols: list[_AotNativeSymbol] = []
    for node in saved_nodes:
        actual = runtime_values.get(node)
        if actual is None:
            return None
        symbol = builder.input(actual)
        saved_symbols.append(symbol)
        forward_symbols[node] = symbol
    for key in forward_lowering.autocast_outputs:
        source_node, dtype = key
        source = forward_symbols.get(source_node)
        actual = runtime_cast_values.get(key)
        if actual is None:
            return None
        if source is None:
            traced = _traced_value(graph_module, source_node)
            if traced is None:
                return None
            source = _AotNativeSymbol(
                builder, None, tuple(int(item) for item in traced.shape), traced.dtype
            )
            forward_symbols[source_node] = source
        converted = builder.input(actual)
        builder._cast_memo[(id(source), dtype)] = (source, converted)
    # Values a forward reported beside its result (an attention normalizer, a
    # normalization's row statistics), addressed by the symbol of the operand
    # the derivative expression names first.  Graph inputs are positional, so
    # they are registered with the other saved values, ahead of the incoming
    # gradient.
    attention_lse: dict[int, tuple[_AotNativeSymbol, _AotNativeSymbol]] = {}
    saved_results: dict[Node, tuple[_AotNativeSymbol, ...]] = {}
    for result_node, extra_values in (saved_result_values or {}).items():
        symbols = tuple(builder.input(value) for value in extra_values)
        saved_results[result_node] = symbols
        output_symbol = forward_symbols.get(result_node)
        if output_symbol is None or not result_node.args:
            continue
        first_symbol = forward_symbols.get(result_node.args[0])
        if first_symbol is not None and len(symbols) == 1:
            attention_lse[id(first_symbol)] = (output_symbol, symbols[0])
    tangent = builder.input(runtime_values[public_node])
    adjoints: dict[Node, _AotNativeSymbol] = {public_node: tangent}
    view_adjoints: dict[Node, dict[int, _AotNativeSymbol]] = {}
    batch_norm_cache: dict[tuple[int, ...], _AotNativeTuple] = {}
    tuple_op_cache: dict[tuple[int, ...], _AotNativeTuple] = {}
    autocast_state: dict[str, Any] = {"dtype": None}
    formula_env = _build_aot_formula_env(
        builder,
        batch_norm_cache=batch_norm_cache,
        tuple_op_cache=tuple_op_cache,
        autocast_state=autocast_state,
        attention_lse=attention_lse,
    )

    def sum_to_shape(value: _AotNativeSymbol, target_shape: tuple[int, ...]) -> _AotNativeSymbol | None:
        current_shape = tuple(int(item) for item in value.shape)
        if len(current_shape) < len(target_shape):
            # An under-shaped contribution (a formula leaning on scalar
            # broadcast, e.g. a mean gradient divided by numel) grows by
            # prepending singleton axes; the expand below sizes the rest.
            for _ in range(len(target_shape) - len(current_shape)):
                value = builder.unsqueeze(value, 0)
            current_shape = tuple(int(item) for item in value.shape)
        leading = len(current_shape) - len(target_shape)
        reduce_dims = list(range(leading))
        for index, target_dim in enumerate(target_shape):
            current_index = leading + index
            current_dim = current_shape[current_index]
            if target_dim == 1 and current_dim != 1:
                reduce_dims.append(current_index)
            elif target_dim != current_dim and current_dim != 1:
                return None
        reduced = builder.sum(value, tuple(reduce_dims), keepdim=True) if reduce_dims else value
        if tuple(reduced.shape) != tuple(target_shape):
            if len(tuple(reduced.shape)) > len(target_shape):
                reduced = builder.reshape(reduced, target_shape)
            else:
                reduced = builder.expand(reduced, target_shape)
        return reduced

    def add_adjoint(target: Any, contribution: Any) -> bool:
        if not isinstance(target, Node) or not isinstance(contribution, _AotNativeSymbol):
            return contribution is None
        target_symbol = forward_symbols.get(target)
        target_value = runtime_values.get(target)
        if target_symbol is not None:
            target_shape = tuple(int(item) for item in target_symbol.shape)
        elif target_value is not None and hasattr(target_value, "shape"):
            target_shape = tuple(int(item) for item in target_value.shape)
        else:
            target_shape = None
        if target_shape is not None:
            contribution = sum_to_shape(contribution, target_shape)
            if contribution is None:
                return False
        # Gradient edges carry the forward output's element type: a formula
        # contribution computed in a promoted type is narrowed here so the
        # consuming formulas see the same element widths the captured graph
        # executed with.  A pure widening on a pointwise edge is exempt: the
        # consuming formulas widen operands to one arithmetic width anyway,
        # so keeping the narrower storage skips a conversion pass without
        # changing any computed value.
        target_dtype = None
        if target_value is not None and hasattr(target_value, "dtype"):
            target_dtype = target_value.dtype
        elif target_symbol is not None:
            target_dtype = target_symbol.dtype
        else:
            traced = _traced_value(graph_module, target)
            target_dtype = getattr(traced, "dtype", None)
            if target_dtype is None:
                try:
                    external_value = runtime_inputs[external_nodes.index(target)]
                except ValueError:
                    external_value = None
                target_dtype = getattr(external_value, "dtype", None)
        if (
            target_dtype is not None
            and contribution.dtype is not None
            and contribution.dtype != target_dtype
            and not _adjoint_promotion_deferrable(
                target, contribution.dtype, target_dtype
            )
        ):
            contribution = builder.cast(contribution, target_dtype)
        return _aot_add_adjoint(builder, adjoints, target, contribution)

    for node in reversed(graph_module.graph.nodes):
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        grad = adjoints.get(node)
        if grad is None and node not in view_adjoints:
            continue
        op_name = _target_name(node.target)
        # Ops whose dispatch narrows their inputs to a reduced precision at
        # capture time run their matrix products in that same element type;
        # the derivative callables read this slot to narrow their operands.
        if op_name in _AUTOCAST_GEMM_OPS:
            sample = node.meta.get("val")
            autocast_state["dtype"] = getattr(sample, "dtype", None)
        else:
            autocast_state["dtype"] = None
        if op_name == "getitem":
            if (
                node.op == "call_function"
                and len(node.args) == 2
                and isinstance(node.args[0], Node)
                and isinstance(node.args[1], tuple)
                and len(node.args[1]) == 2
                and isinstance(node.args[1][0], slice)
                and node.args[1][0] == slice(None)
                and node.args[1][1] is None
                and grad is not None
            ):
                source = node.args[0]
                source_symbol = forward_symbols.get(source)
                if source_symbol is None or len(grad.shape) != len(source_symbol.shape) + 1:
                    return None
                # This form inserts one singleton axis after the leading
                # slice.  Its reverse is an axis removal, not a broadcast
                # reduction: the inserted axis may sit in the middle of the
                # shape, so a generic leading-dimension reduction is wrong.
                if tuple(grad.shape[0:1]) != tuple(source_symbol.shape[0:1]):
                    return None
                contribution = builder.squeeze(grad, dim=1)
                if tuple(contribution.shape) != tuple(source_symbol.shape):
                    return None
                if not add_adjoint(source, contribution):
                    return None
                continue
            if (
                node.op != "call_function"
                or len(node.args) != 2
                or not isinstance(node.args[0], Node)
                or not isinstance(node.args[1], int)
                or node.args[0].op not in {
                    "call_function",
                    "call_method",
                }
                or _target_name(node.args[0].target)
                not in {"chunk", "split", "split_with_sizes", "unbind"}
                or grad is None
            ):
                return None
            source = node.args[0]
            sample = source.meta.get("val")
            if not isinstance(sample, (tuple, list)):
                return None
            index = int(node.args[1])
            if index < 0:
                index += len(sample)
            if index < 0 or index >= len(sample):
                return None
            slots = view_adjoints.setdefault(source, {})
            previous = slots.get(index)
            slots[index] = grad if previous is None else builder.binary(
                "add", previous, grad
            )
            continue
        if op_name in {"chunk", "split", "split_with_sizes", "unbind"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source = node.args[0]
            sample = node.meta.get("val")
            slots = view_adjoints.pop(node, None)
            if not isinstance(sample, (tuple, list)) or not slots:
                return None
            source_symbol = forward_symbols.get(source)
            source_value = runtime_values.get(source)
            if source_symbol is not None:
                source_shape = tuple(int(item) for item in source_symbol.shape)
            elif source_value is not None and hasattr(source_value, "shape"):
                source_shape = tuple(int(item) for item in source_value.shape)
            else:
                return None
            kwargs = dict(node.kwargs or {})
            if op_name == "unbind":
                if set(kwargs) - {"dim"} or len(node.args) > 2:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                dim_arg = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
            else:
                if set(kwargs) - {"dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                dim_arg = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
            if isinstance(dim_arg, bool) or not isinstance(dim_arg, int):
                return None
            dim_arg = int(dim_arg)
            rank = len(source_shape)
            if dim_arg < 0:
                dim_arg += rank
            if dim_arg < 0 or dim_arg >= rank:
                return None
            parts: list[_AotNativeSymbol] = []
            template = next(iter(slots.values()))
            for index, output in enumerate(sample):
                part = slots.get(index)
                if part is None:
                    part_shape = getattr(output, "shape", None)
                    if part_shape is None:
                        return None
                    part = builder.zeros_like(template, part_shape)
                if op_name == "unbind":
                    part = builder.unsqueeze(part, dim_arg)
                parts.append(part)
            contribution = builder.cat(tuple(parts), dim_arg, source_shape)
            if not add_adjoint(source, contribution):
                return None
            continue
        if op_name == "cat":
            if grad is None or not node.args or not isinstance(node.args[0], (tuple, list)):
                return None
            sources = tuple(node.args[0])
            if not sources or any(not isinstance(item, Node) for item in sources):
                return None
            dim_arg = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            if not isinstance(dim_arg, int):
                return None
            shapes: list[tuple[int, ...]] = []
            sizes: list[int] = []
            for source in sources:
                source_symbol = forward_symbols.get(source)
                source_value = runtime_values.get(source)
                if source_symbol is not None:
                    shape_tuple = tuple(int(item) for item in source_symbol.shape)
                elif source_value is not None and hasattr(source_value, "shape"):
                    shape_tuple = tuple(int(item) for item in source_value.shape)
                else:
                    return None
                shapes.append(shape_tuple)
                dim_index = dim_arg if dim_arg >= 0 else dim_arg + len(shape_tuple)
                if dim_index < 0 or dim_index >= len(shape_tuple):
                    return None
                sizes.append(shape_tuple[dim_index])
            parts = builder.split(grad, tuple(sizes), dim_arg, tuple(shapes))
            for source, part in zip(sources, parts.values):
                if not add_adjoint(source, part):
                    return None
            continue
        if op_name == "index_select":
            if grad is None:
                return None
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                input_node = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index_node = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                input_node = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index_node = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            if (
                not isinstance(input_node, Node)
                or not isinstance(dim, int)
                or isinstance(dim, bool)
                or not isinstance(index_node, Node)
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            index_symbol = forward_symbols.get(index_node)
            if input_symbol is None or index_symbol is None:
                return None
            contribution = formula_env["index_select_backward"](
                grad, input_symbol, dim, index_symbol
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue
        if op_name == "gather":
            if grad is None:
                return None
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                input_node = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index_node = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                input_node = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index_node = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            if (
                not isinstance(input_node, Node)
                or not isinstance(dim, int)
                or isinstance(dim, bool)
                or not isinstance(index_node, Node)
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            index_symbol = forward_symbols.get(index_node)
            if input_symbol is None or index_symbol is None:
                return None
            contribution = formula_env["gather_backward"](
                grad, input_symbol, dim, index_symbol
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue
        if grad is None:
            return None
        if op_name == "linear":
            if len(node.args) not in {2, 3} or not all(
                isinstance(item, Node) for item in node.args[:2]
            ):
                return None
            input_node, weight_node = node.args[:2]
            bias_node = node.args[2] if len(node.args) == 3 else None
            if bias_node is not None and not isinstance(bias_node, Node):
                return None
            input_value = forward_symbols[input_node]
            weight_value = forward_symbols[weight_node]
            sample = node.meta.get("val")
            ac_dtype = getattr(sample, "dtype", None)
            narrow = builder.cast
            backward = builder.helper(
                "linear_backward",
                (
                    narrow(input_value, ac_dtype),
                    narrow(grad, ac_dtype),
                    narrow(weight_value, ac_dtype),
                ),
                attrs={"output_mask": (1, 1, 1)},
                outputs=3,
                output_shapes=(
                    tuple(input_value.shape),
                    tuple(weight_value.shape),
                    (int(weight_value.shape[0]),),
                ),
            )
            if not isinstance(backward, _AotNativeTuple):
                return None
            input_grad, weight_grad, bias_grad = backward.values
            if not add_adjoint(input_node, input_grad):
                return None
            if not add_adjoint(weight_node, weight_grad):
                return None
            if bias_node is not None and not add_adjoint(bias_node, bias_grad):
                return None
            continue

        if op_name in {"reshape", "view"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            input_node = node.args[0]
            input_symbol = forward_symbols.get(input_node)
            input_value = runtime_values.get(input_node)
            if input_symbol is not None:
                input_shape = tuple(int(item) for item in input_symbol.shape)
            elif input_value is not None and hasattr(input_value, "shape"):
                input_shape = tuple(int(item) for item in input_value.shape)
            else:
                return None
            contribution = builder.reshape(grad, input_shape)
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "flatten":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source_value = runtime_values.get(node.args[0])
            if source_value is None:
                return None
            # the input storage.  Keep that metadata-only dependency out of
            # the saved-tensor list so the rebuilt graph can omit the pooled
            # activation while still producing the exact reshape backward.
            contribution = builder.reshape(grad, tuple(source_value.shape))
            if not add_adjoint(node.args[0], contribution):
                return None
            continue

        if op_name in {"max_pool1d", "max_pool2d", "max_pool3d"}:
            # The captured op carries no indices value, so the derivative
            # formula for the with-indices overload cannot apply; the native
            # backward kernel recomputes the argmax instead.
            if len(node.args) != 7 or not isinstance(node.args[0], Node):
                return None
            if bool(node.args[6]):
                return None
            input_node = node.args[0]
            if input_node not in forward_symbols:
                return None
            rank = int(op_name[-2])
            kernel = _spatial_int_list(node.args[1], length=rank)
            stride = _spatial_int_list(node.args[2], default=kernel, length=rank)
            padding = _spatial_int_list(node.args[3], default=[0] * rank, length=rank)
            dilation = _spatial_int_list(node.args[4], default=[1] * rank, length=rank)
            if any(item is None for item in (kernel, stride, padding, dilation)):
                return None
            contribution = builder.helper(
                f"max_pool{rank}d_backward",
                (grad, forward_symbols[input_node]),
                attrs={
                    "kernel_size": tuple(kernel),
                    "stride": tuple(stride),
                    "padding": tuple(padding),
                    "dilation": tuple(dilation),
                    "ceil_mode": bool(node.args[5]),
                },
                shape=forward_symbols[input_node].shape,
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "dropout":
            if len(node.args) != 4 or not isinstance(node.args[0], Node):
                return None
            input_node, probability, training, _inplace = node.args
            if not isinstance(probability, numbers.Real) or not isinstance(training, bool):
                return None
            if float(probability) == 0.0 or not training:
                if not add_adjoint(input_node, grad):
                    return None
                continue
            return None

        if op_name == "interpolate":
            if len(node.args) != 7 or not isinstance(node.args[0], Node):
                return None
            input_node, output_size, scale_factor, mode, align_corners, recompute, antialias = node.args
            if (
                scale_factor is not None
                or mode != "nearest"
                or align_corners is not None
                or recompute is not None
                or antialias is not False
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            input_value = runtime_values.get(input_node)
            if input_symbol is not None:
                input_shape = tuple(int(item) for item in input_symbol.shape)
            elif input_value is not None and hasattr(input_value, "shape"):
                input_shape = tuple(int(item) for item in input_value.shape)
            else:
                return None
            output_size = tuple(int(item) for item in output_size)
            spatial_rank = len(input_shape) - 2
            if spatial_rank not in (1, 2, 3):
                return None
            contribution = builder.helper(
                f"upsample_nearest{spatial_rank}d_backward",
                (grad,),
                attrs={
                    "output_size": output_size,
                    "input_size": input_shape,
                },
                shape=input_shape,
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "permute":
            if len(node.args) < 2 or not isinstance(node.args[0], Node):
                return None
            source = forward_symbols.get(node.args[0])
            if source is None:
                return None
            dims = node.args[1]
            if len(node.args) > 2:
                dims = tuple(node.args[1:])
            else:
                dims = tuple(int(item) for item in dims)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in dims):
                return None
            contribution = builder.helper(
                "permute_backward",
                (grad, source),
                attrs={"dims": tuple(int(item) for item in dims)},
                shape=source.shape,
            )
            if not add_adjoint(node.args[0], contribution):
                return None
            continue

        if op_name == "float":
            if len(node.args) != 1 or not isinstance(node.args[0], Node):
                return None
            if not add_adjoint(node.args[0], grad):
                return None
            continue

        # A captured call_method carries no overload suffix: ``sum(dim=...)``
        # lands here spelled ``sum``, whose whole-tensor schema has no ``dim``
        # parameter.  Route calls that pass a dimension to the dim-variety
        # schema when one exists; every other spelling keeps the base schema.
        node_kwargs = dict(node.kwargs or {})
        has_dim_arg = "dim" in node_kwargs or (
            len(node.args) > 1 and isinstance(node.args[1], (int, list, tuple))
        )
        schema = None
        if has_dim_arg:
            schema = _aot_schema_for(specs, f"{op_name}.dim_IntList")
        if schema is None:
            schema = _aot_schema_for(specs, op_name)
        if schema is None:
            return None
        # A forward that reported values beside its result ran under the
        # native spelling; derive against that schema so the formula reads
        # those values by the names the schema gives them.
        reported = saved_results.get(node)
        if reported is not None:
            native_name = _AOT_NATIVE_SPELLING.get(op_name)
            native_schema = (
                _aot_schema_for(specs, native_name) if native_name else None
            )
            if native_schema is not None:
                op_name = native_name
                schema = native_schema
        parsed, formulas = schema
        if node.op == "call_method":
            schema_args = tuple(parsed.args)
            if len(node.args) > len(schema_args):
                return None
            bound_args: dict[str, Any] = {}
            for arg, value in zip(schema_args, node.args):
                if arg.kwonly:
                    return None
                bound_args[arg.name] = value
            schema_names = {arg.name for arg in schema_args}
            for name, value in (node.kwargs or {}).items():
                if name not in schema_names or name in bound_args:
                    return None
                bound_args[name] = value
            if any(
                arg.name not in bound_args and arg.default is None
                for arg in schema_args
            ):
                return None
            arg_values = tuple(
                (arg.name, bound_args[arg.name])
                for arg in schema_args
                if arg.name in bound_args
            )
        elif op_name == "batch_norm":
            names = (
                "input", "running_mean", "running_var", "weight", "bias",
                "training", "momentum", "eps",
            )
            arg_values = tuple(zip(names, node.args))
        elif op_name in _AOT_NATIVE_SPELLING.values() and reported is not None:
            remapped = _native_group_norm_arguments(node, _traced_value(graph_module, node))
            if remapped is None:
                return None
            arg_values = tuple(remapped)
        else:
            arg_values = tuple(zip((arg.name for arg in parsed.args), node.args))
        if op_name == "batch_norm":
            context = {name: value for name, value in arg_values}
        else:
            context = dict(arg_values)
        for arg in parsed.args:
            if arg.name not in context and arg.default is not None:
                context[arg.name] = _aot_default_value(arg.default)
        context["grad"] = grad
        # Derivative expressions name the output mask a multi-output backward
        # should fill; the builder tracks which ports have a consumer, and an
        # unmasked formula fills all of them.
        context["grad_input_mask"] = (True,) * len(
            getattr(parsed, "returns", ()) or ()
        )
        # others only need metadata or their inputs (e.g. adaptive average
        # pooling).  A pruned saved-tensor set must not require a symbol for
        # a result that the selected formula never reads.
        context["result"] = forward_symbols.get(node)
        if reported is not None and len(parsed.returns) == len(reported) + 1:
            # Values the forward reported beside its result answer to the
            # names the schema gives those results.
            for declaration, symbol in zip(parsed.returns[1:], reported):
                context[declaration.name] = symbol
        tensor_params = {
            name for name, value in context.items() if isinstance(value, Node)
        }
        env = dict(formula_env)
        # A derivative formula may read a saved tensor only for metadata
        # (e.g. the reduction backward re-expands the tangent to the input
        # shape) and never touch its value.  The rebuilt forward graph drops
        # every activation whose value no backward op consumes, so such
        # inputs resolve to no symbol here.  Rebuild one from the traced
        # shape/dtype instead: metadata reads succeed without re-saving the
        # activation, while a formula that emits an operation on the value
        # still fails the eval below and keeps the graph off the AOT route.
        for name, value in context.items():
            if not isinstance(value, Node) or value in forward_symbols:
                continue
            traced = _traced_value(graph_module, value)
            if traced is None or not hasattr(traced, "shape"):
                continue
            forward_symbols[value] = _AotNativeSymbol(
                builder,
                None,
                tuple(int(item) for item in traced.shape),
                getattr(traced, "dtype", None),
            )
        env.update(
            {
                name: forward_symbols.get(value) if isinstance(value, Node) else value
                for name, value in context.items()
            }
        )
        try:
            for arg_name, formula in formulas.items():
                target = context.get(arg_name)
                if not isinstance(target, Node):
                    continue
                translated = _aot_formula_python(formula, tensor_params)
                contribution = eval(translated, {"__builtins__": {}}, env)
                if not add_adjoint(target, contribution):
                    return None
        except (KeyError, NameError, NotImplementedError, TypeError, ValueError, RuntimeError) as exc:
            import os as _os
            if _os.environ.get("TP_AOT_DEBUG"):
                print(f"[aot] {op_name} formula failed: {type(exc).__name__}: {exc}")
                import traceback; traceback.print_exc()
            return None

    grad_positions: list[int] = []
    for index, node in enumerate(external_nodes):
        actual = runtime_inputs[index]
        if not getattr(actual, "requires_grad", False):
            continue
        contribution = adjoints.get(node)
        if contribution is None:
            continue
        builder._materialize(contribution)
        builder.graph.register_output(builder._force_cast(contribution).value)
        grad_positions.append(index)
    if not grad_positions:
        return None
    needed_saved_nodes = [
        node
        for node, symbol in zip(saved_nodes, saved_symbols)
        if getattr(symbol.value, "use_count", 0) != 0
    ]
    return builder.graph, grad_positions, needed_saved_nodes

def _copy_back_mutations(lowering: Any, inputs: Any, outputs: Any) -> None:
    for position, output_index in lowering._mutations:
        inputs[position].copy_(outputs[output_index])

class _AotNativeLowering:

    def __init__(
        self,
        graph_module: GraphModule,
        forward_graph: Any,
        backward_graph: Any,
        attribute_targets: list[str],
        grad_positions: list[int],
        constant_values: list[Any] | None = None,
        mutations: list[tuple[int, int]] | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.forward_graph = forward_graph
        self.backward_graph = backward_graph
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets = attribute_targets
        self.constant_values = list(constant_values or [])
        self.grad_positions = list(grad_positions)
        self._mutations = list(mutations or [])
        self.input_count = len(self.placeholders) + len(self.attribute_targets)
        self._tensorplay_codegen = "stax-aot-native"
        self._tensorplay_backward_codegen = "stax-aot-native"
        lowering = self
        from ....autograd import Function

        class _AotAutogradFunction(Function):
            _tensorplay_direct_backward = True

            @staticmethod
            def forward(ctx: Any, *inputs: Any) -> Any:
                # The custom function owns the only gradient edge for this
                # region.  Keep native forward operators outside the eager
                # autograd graph even when the fused apply path leaves grad
                # recording enabled around the Python callback.
                import tensorplay

                with tensorplay.no_grad():
                    outputs = lowering.forward_graph.execute(
                        [*inputs, *lowering.constant_values]
                    )
                    _copy_back_mutations(lowering, inputs, outputs)
                    # Buffer-update outputs trail the saved tensors; they are
                    # epilogue state, not backward inputs.
                    ctx.save_for_backward(
                        *inputs,
                        *lowering.constant_values,
                        *outputs[1 : len(outputs) - len(lowering._mutations)],
                    )
                return outputs[0]

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                grad_output = grad_outputs[0] if grad_outputs else None
                if grad_output is None:
                    return (None,) * lowering.input_count
                saved = list(ctx.saved_tensors)
                outputs = lowering.backward_graph.execute([*saved, grad_output])
                by_position = dict(zip(lowering.grad_positions, outputs))
                return tuple(by_position.get(index) for index in range(lowering.input_count))

        self._autograd_function = _AotAutogradFunction

    def _bind_inputs(self, *args: Any, **kwargs: Any) -> list[Any]:
        bound = self.graph_module.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        inputs = [
            bound.arguments[node.target if isinstance(node.target, str) else node.name]
            for node in self.placeholders
        ]
        inputs.extend(self.graph_module._get_attr(target) for target in self.attribute_targets)
        return inputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._bind_inputs(*args, **kwargs)
        import tensorplay

        if not tensorplay.is_grad_enabled() or not any(
            getattr(value, "requires_grad", False) for value in inputs
        ):
            outputs = self.forward_graph.execute([*inputs, *self.constant_values])
            _copy_back_mutations(self, inputs, outputs)
            return outputs[0]
        return self._autograd_function.apply(*inputs)

def _lower_aot_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool = True,
) -> _AotNativeLowering | None:
    """Build separate native forward/backward graphs at the AOT boundary."""

    try:
        import tensorplay

        native_module = getattr(tensorplay._C, "_stax", None)
        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if native_module is None or not hasattr(native_module.Graph, "execute"):
        return None
    if len(example_inputs) != len(graph_module.graph.placeholders):
        return None
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        return None
    if not tensorplay.is_grad_enabled():
        return None

    external_nodes = list(graph_module.graph.placeholders) + [
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    ]
    attribute_targets = [node.target for node in external_nodes if node.op == "get_attr"]
    runtime_inputs = list(example_inputs)
    runtime_inputs.extend(graph_module._get_attr(target) for target in attribute_targets)
    if not any(getattr(value, "requires_grad", False) for value in runtime_inputs):
        return None

    saved_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op in {"call_function", "call_method"}
        and not isinstance(node.meta.get("val"), (tuple, list))
    ]
    output_values = [
        value for output in graph_module.graph.outputs for value in _nodes(output.args)
    ]
    if len(output_values) != 1:
        return None
    public_node = output_values[0]
    forward_lowering = _lower_native(
        graph_module,
        example_inputs,
        use_fusion=use_fusion,
        extra_output_nodes=saved_nodes,
        save_autocast_inputs=True,
    )
    if forward_lowering is None:
        return None
    if len(runtime_inputs) + len(forward_lowering.constant_values) != len(
        forward_lowering.graph.inputs
    ):
        return None

    # Training BatchNorm updates running buffers during forward.  A compiler
    # trace must not perform that update a second time; restore non-gradient
    # capture path separates tracing state from the user execution state.
    snapshots: list[tuple[Any, Any]] = []
    seen_attributes: set[int] = set()
    try:
        for target in attribute_targets:
            value = graph_module._get_attr(target)
            if (
                isinstance(value, tensor_type)
                and not getattr(value, "requires_grad", False)
                and id(value) not in seen_attributes
            ):
                snapshots.append((value, value.detach().clone()))
                seen_attributes.add(id(value))
        with tensorplay.no_grad():
            forward_outputs = forward_lowering.graph.execute(
                [*runtime_inputs, *forward_lowering.constant_values]
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        if snapshots:
            with tensorplay.no_grad():
                for value, snapshot in snapshots:
                    value.copy_(snapshot)

    cast_count = len(forward_lowering.autocast_outputs)
    extra_specs = list(forward_lowering.saved_result_outputs)
    extra_count = sum(count for _node, count in extra_specs)
    if len(forward_outputs) != (
        1 + len(saved_nodes) + cast_count + extra_count + len(forward_lowering._mutations)
    ):
        return None
    runtime_values: dict[Node, Any] = {public_node: forward_outputs[0]}
    for index, node in enumerate(saved_nodes, start=1):
        runtime_values[node] = forward_outputs[index]
    runtime_cast_values = dict(zip(
        forward_lowering.autocast_outputs,
        forward_outputs[1 + len(saved_nodes) : 1 + len(saved_nodes) + cast_count],
    ))
    runtime_saved_results: dict[Node, tuple[Any, ...]] = {}
    cursor = 1 + len(saved_nodes) + cast_count
    for node, count in extra_specs:
        runtime_saved_results[node] = tuple(forward_outputs[cursor : cursor + count])
        cursor += count

    built = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        saved_nodes,
        runtime_values,
        runtime_cast_values,
        runtime_inputs,
        public_node,
        runtime_saved_results,
    )
    if built is None:
        return None
    _, grad_positions, needed_saved_nodes = built
    # The first graph is a shape/materialization graph.  Rebuild the forward
    # graph with only the values that the source-derived backward graph reads,
    # intermediate until backward.
    forward_lowering = _lower_native(
        graph_module,
        example_inputs,
        use_fusion=use_fusion,
        extra_output_nodes=needed_saved_nodes,
        save_autocast_inputs=True,
    )
    if forward_lowering is None or forward_lowering.autocast_outputs != list(runtime_cast_values):
        return None
    rebuilt = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        needed_saved_nodes,
        runtime_values,
        runtime_cast_values,
        runtime_inputs,
        public_node,
        runtime_saved_results,
    )
    if rebuilt is None:
        return None
    backward_graph, grad_positions, rebuilt_saved_nodes = rebuilt
    if rebuilt_saved_nodes != needed_saved_nodes:
        return None
    return _AotNativeLowering(
        graph_module,
        forward_lowering.graph,
        backward_graph,
        attribute_targets,
        grad_positions,
        constant_values=forward_lowering.constant_values,
        mutations=forward_lowering._mutations,
    )


#: Names this layer owns.  The driver re-exports them, so the
#: import surface of the package does not change: the ahead-of-time reverse pass.
__all__ = [
    "_AotNativeGraphBuilder",
    "_AotNativeLowering",
    "_aot_add_adjoint",
    "_aot_default_value",
    "_aot_derivative_specs",
    "_aot_formula_python",
    "_aot_schema_for",
    "_build_aot_backward",
    "_build_aot_formula_env",
    "_copy_back_mutations",
    "_lower_aot_native",
]
