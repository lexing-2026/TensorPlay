"""What a choice reads: the tensors it runs on, and the extents each is reduced over.

A choice is asked to run, to measure itself, and to say whether it fits, and
all three need the same things said once: which tensors it is reading, which
extent each of them is indexed by, and which of those extents a reduction
walks.  An operation reads them the same way a generated kernel does, so the
two kinds of choice are handed to the same enumeration without caring which
is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .ir import Layout

class KernelInputs:
    """What a template is being asked to run.

    The operands, their extents and their element types, kept together so a
    heuristic can size itself to the problem and a template can refuse a form
    it has no kernel for.

    The constructor is written out rather than declared, because what a record
    holds is not the same list for every kind: a product is told which of its
    operands are the matrices, a convolution is told the geometry that decides
    its result, and a reduction is told nothing beyond its own extents.  A
    declaration would have to name the union of all of them and every caller
    would have to supply all of them, so each record says what it takes and the
    rest is asked for by name.
    """

    def __init__(
        self,
        input_nodes: tuple = (),
        shapes: tuple = (),
        strides: tuple = (),
        dtypes: tuple = (),
        device: Any = None,
        operands: tuple = (),
        feed: tuple = (),
        probe_feed: tuple = (),
        scalars: dict | None = None,
        out_dtype: Any = None,
        extra: dict | None = None,
    ):
        self._input_nodes = tuple(input_nodes)
        self.shapes = tuple(shapes)
        self.strides = tuple(strides)
        self.dtypes = tuple(dtypes)
        self._device = device
        self.operands = tuple(operands)
        self.feed = tuple(feed)
        self.probe_feed = tuple(probe_feed)
        self._scalars = dict(scalars) if scalars else {}
        self._out_dtype = out_dtype
        self.extra = dict(extra) if extra else {}
        if not self.shapes and not self._input_nodes:
            raise AssertionError("expected at least one input node")

    @property
    def device_type(self) -> str | None:
        """The kind of device the first operand is on.

        A heuristic branches on this -- a tile table is not a tile table on a
        host -- so it is the first thing asked and it is asked by name.

        Read from the operands rather than from what the caller said, because a
        caller that describes a call it has only partly seen can name a device
        that is not the one the call will run on, and a heuristic given the wrong
        one chooses a tile table for hardware that is not there.
        """

        from .ir import get_device_type

        if self._input_nodes:
            return get_device_type(self._input_nodes[0])
        device = self._device
        return None if device is None else getattr(device, "type", None)

    @property
    def device(self):
        """Where the first operand is.

        Read from the operands for the same reason the kind of device is: what a
        caller said about a call it has not fully seen is not what the call will
        run on.
        """

        if self._input_nodes:
            return self._input_nodes[0].get_device()
        return self._device

    def device_name(self) -> str | None:
        """The device's own name, asked of the device rather than assumed.

        Templates that key their configuration on the architecture cannot
        assume which one they are on, so this asks; a device that will not
        name itself has none to give.
        """

        if self.device is None:
            return None
        import tensorplay as tp

        index = getattr(self.device, "index", None)
        if index is None:
            return None
        try:
            return tp.cuda.get_device_properties(index).name
        except (AttributeError, RuntimeError, TypeError):
            return None

    def _hinted(self, extents):
        """The extents with every symbol replaced by the value it stands for.

        A declared extent can be an expression, and a measurement cannot use an
        expression: it needs a number to launch with.  Where a value is known
        it is substituted here; where it is not, the expression is handed on
        unchanged, so what comes back is always something a caller can look at
        and be told what it is looking at.
        """

        from .loops import V

        sizevars = getattr(V, "sizevars", None)
        if sizevars is None:
            return tuple(extents)
        return tuple(tuple(sizevars.optimization_hints(e) for e in extents) for extents in extents)

    def shapes_symbolic(self) -> tuple:
        """The operands' extents as declared, expressions and all.

        Asked of the operands rather than of what the caller said, because a
        caller that was handed extents separately from the operands can hand
        over extents of something else, and every question about a product is a
        question about the operands' extents.
        """

        if self._input_nodes:
            return tuple(tuple(node.get_size()) for node in self._input_nodes)
        return self.shapes

    def shapes_hinted(self) -> tuple:
        """The operands' extents with every symbol resolved to its value."""

        return self._hinted(self.shapes_symbolic())

    def strides_symbolic(self) -> tuple:
        """The operands' strides as declared, expressions and all."""

        return self.strides

    def strides_hinted(self) -> tuple:
        """The operands' strides with every symbol resolved to its value."""

        return self._hinted(self.strides)

    def nodes(self, reorder: Sequence[int] | None = None) -> tuple:
        """The operands, optionally in another order.

        The order matters to a product, which names two of them, so a caller
        that wants them the other way round says so -- and a reorder that does
        not preserve the count is refused rather than quietly shortened.
        """

        if reorder is None:
            return self._input_nodes
        if len(self._input_nodes) != len(reorder):
            raise AssertionError(
                f"reorder length mismatch: {len(self._input_nodes)} vs {len(reorder)}"
            )
        return tuple(self._input_nodes[i] for i in reorder)

    def count(self) -> int:
        return len(self.shapes) or len(self._input_nodes)

    def dtype(self, idx: int = 0) -> Any:
        return self.dtypes[idx] if idx < len(self.dtypes) else None

    def out_dtype(self) -> Any:
        """The result's element type: the one declared, or inferred from the operands.

        A caller that knows the result's type says so, because a product's type
        is not always any operand's type -- narrowing happens in the operation
        rather than in its inputs.  When nobody said, it is taken from the
        first operand, which is the type a caller would have expected had they
        not had to think about it.
        """

        if self._out_dtype is not None:
            return self._out_dtype
        return self.dtypes[0] if self.dtypes else None

    def get_scalar(self, name: str) -> Any:
        """A number the call carried that is not an operand's extent."""

        if name not in self._scalars:
            raise KeyError(f"no scalar named {name!r} in this call")
        return self._scalars[name]

    @property
    def output_layout(self, flexible: bool = True):
        """The result's layout, either settled or still open.

        Which one is a question the caller has to answer rather than one this
        can answer: a layout that may still change and a layout that may not
        are for different moments, and picking the wrong one either forbids a
        change that was wanted or promises one that will not happen.  So it is
        asked for, and a kind of call that has no answer says so.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not know its result's layout"
        )

    def rank(self) -> int:
        return max((len(s) for s in self.shapes), default=0)

    def mnk(self) -> tuple:
        """The extents a product of these operands has."""

        raise NotImplementedError


class MMKernelInputs(KernelInputs):
    """The operands of a product, and where the two matrices are among them.

    The matrices are named by position rather than assumed.  A product is very
    often not the whole call: an add takes an accumulator to add into, a layer
    is handed a scale, a fused form is handed a destination.  So the operands
    are addressed from the end, which is where the matrices sit in the forms
    that have nothing in front, and said so explicitly by the forms that do.
    Naming them is what lets one product template serve all of them.

    A product of batches is one call rather than many: the extents in front of
    the matrices are the batch, and they are checked against each other rather
    than assumed equal, because a product of different batch extents is either
    a broadcast or a mistake and the caller is the one who knows which.
    """
    def __init__(
        self,
        input_nodes: tuple = (),
        scalars: dict | None = None,
        out_dtype: Any = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
        shapes: tuple = (),
        **kwargs: Any,
    ):
        # A product is very often not the whole call, so the two matrices are
        # named by position rather than assumed: an add takes an accumulator to
        # add into, a layer is handed a scale, a fused form is handed a
        # destination.  The positions are stated the way the call states them --
        # counting from the end, so a caller need not know how many there are --
        # and are resolved where they are used, so the record of what was asked
        # for stays separate from the arithmetic about it.
        #
        # The operands' extents are declared rather than left to the general
        # keyword form, because a record built for a product is asked for its
        # matrices' extents and a caller that offers them has to be able to
        # hand them over: a parameter reached only through the keyword form is
        # a parameter a caller cannot fill, and a product with no extents has
        # no rows, columns or contraction to answer for.
        super().__init__(
            input_nodes=input_nodes,
            shapes=shapes,
            scalars=scalars,
            out_dtype=out_dtype,
            **kwargs,
        )
        self._mat1_idx = mat1_idx
        self._mat2_idx = mat2_idx



    def matrix_indices(self) -> tuple:
        """Where the two matrices are, counting from the front.

        The positions are kept as the call stated them -- a negative index
        counts from the end, which is how a caller says "the last two" without
        having to know how many there are -- and resolved here, so the record
        of what was asked for and the arithmetic about it stay separate.
        """

        # Counted over the operands rather than over what the caller declared,
        # because a position is a position among the operands and a count of
        # something else would resolve it against the wrong list.
        count = len(self.shapes_symbolic())
        if count < 2:
            raise AssertionError("a product needs two operands")
        first, second = self._mat1_idx, self._mat2_idx
        if first < 0:
            first += count
        if second < 0:
            second += count
        if not (0 <= first < count):
            raise AssertionError(f"the first matrix is at {first} of {count} operands")
        if not (0 <= second < count):
            raise AssertionError(f"the second matrix is at {second} of {count} operands")
        return first, second

    def mat1mat2(self) -> tuple:
        """The two matrices, by the positions the call gave them."""

        first, second = self.matrix_indices()
        shapes = self.shapes_symbolic()
        return shapes[first], shapes[second]

    def matrix_dtypes(self) -> tuple:
        first, second = self.matrix_indices()
        return (
            self.dtypes[first] if self.dtypes else None,
            self.dtypes[second] if self.dtypes else None,
        )

    def batch(self) -> tuple:
        """The extents in front of the matrices, which the product batches over."""

        first, second = self.mat1mat2()
        leading = [int(e) for e in first[:-2]]
        other = [int(e) for e in second[:-2]]
        if len(leading) != len(other):
            raise NotImplementedError("operands batched over different ranks")
        for a, b in zip(leading, other):
            if a != b and 1 not in (a, b):
                raise NotImplementedError(f"batches of {a} and {b} meet in a product")
        return tuple(b if a == 1 else a for a, b in zip(leading, other))

    def mnk(self) -> tuple:
        """The three extents a product reduces over, in the order the table wants."""

        first, second = self.mat1mat2()
        m, k = int(first[-2]), int(first[-1])
        k2, n = int(second[-2]), int(second[-1])
        if k != k2:
            raise NotImplementedError(f"operands contracting {k} and {k2}")
        return m, n, k

    def mnk_symbolic(self) -> tuple:
        """The extents as written, for a grid sized before it is run."""

        return self.mnk()

    def mnk_hinted(self) -> tuple:
        """The extents as whole numbers, for a heuristic sizing itself."""

        return self.mnk()

    def batch_hinted(self) -> int:
        """The batch, or one when there is none."""

        batch = self.batch()
        return int(batch[0]) if batch else 1

    def result_dtype(self) -> Any:
        """The result's element type: the one asked for, or the first matrix's."""

        if self._out_dtype is not None:
            return self._out_dtype
        first, _second = self.matrix_indices()
        return self.dtypes[first] if self.dtypes else None

    def output_layout(self, flexible: bool = True) -> Layout:
        """Where the result lands: the batch, then the two free extents.

        Which kind of layout that is depends on when it is wanted rather than
        on what the operands are, so it is chosen by class here: a layout that
        may still change and one that may not are different things, and a flag
        carried through a constructor would only be the difference restated
        somewhere less visible.
        """

        # Imported here because the template package imports this one: a module
        # that reaches its package at import time and is reached by that
        # package's own modules cannot be imported at the top of the file.
        from .ir import FixedLayout, FlexibleLayout
        from .templates.ir import contiguous_stride

        m, n, _k = self.mnk()
        size = (*self.batch(), m, n)
        if flexible:
            # A layout that may still change is given the extents and left to
            # work out an order; one that may not is handed the order that
            # order implies, because by then there is nothing left to decide.
            return FlexibleLayout(self.device, self.result_dtype(), size)
        return FixedLayout(
            self.device, self.result_dtype(), size, contiguous_stride(size)
        )

    def is_contiguous(self) -> bool:
        return bool(self.extra.get("qualifies", False))


class ConvKernelInputs(KernelInputs):
    """The operands of a convolution: an activation and a weight.

    A convolution's product is over the input channels, but what it produces
    is positions, so the extents a configuration is sized against are counted
    differently from a product's: the positions of every image in the batch
    together, because a tile does not care which image a position came from.
    """

    def mnk(self) -> tuple:
        extra = self.extra
        rows = extra.get("conv_rows")
        cols = extra.get("out_channels")
        inner = extra.get("in_channels_per_group")
        if not (rows and cols and inner):
            raise NotImplementedError("a convolution whose geometry is unknown")
        return rows, cols, inner

    def mnk_symbolic(self) -> tuple:
        return self.mnk()

    @property
    def is_depthwise(self) -> bool:
        """Each input channel reduced on its own: not a product at all."""

        groups = int(self.extra.get("groups", 1) or 1)
        return groups > 1 and self.extra.get("in_channels_per_group") == 1

    def output_layout(self, flexible: bool = True) -> Layout:
        """Where the result lands: whatever batching there is, then positions.

        A convolution's result is its batch and its output positions, and the
        channels are the one axis a layout may be chosen over -- which is why
        asking for a channels-last answer is asking for a different kind of
        layout rather than for a different stride on the same one.
        """

        from .ir import FixedLayout, FlexibleLayout

        from .templates.ir import contiguous_stride

        size = list(self.extra.get("out_size") or self.shapes_hinted()[0])
        if flexible:
            return FlexibleLayout(self.device, self._out_dtype, size)
        return FixedLayout(self.device, self._out_dtype, size, contiguous_stride(size))

    def is_1x1(self) -> bool:
        kernel = tuple(self.extra.get("kernel_size") or ())
        stride = tuple(self.extra.get("stride") or ())
        padding = tuple(self.extra.get("padding") or ())
        return bool(kernel) and all(k == 1 for k in kernel) and all(
            s == 1 for s in stride
        ) and all(p == 0 for p in padding) and int(
            self.extra.get("groups", 1) or 1
        ) == 1 and not self.extra.get("transposed")

