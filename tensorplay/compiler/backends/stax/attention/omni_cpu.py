"""Attention on a processor, written as a shader rather than as tiles.

The kernels beside this one spread their work over blocks and read a mask
several positions at a time.  A processor has no blocks and cannot compare
several positions at once in a way that is worth a kernel, so what it is handed
is a program: one thread group per query and key block, each walking its own
part of the mask and combining what it found.  That is a different shape of
program from the other two, not a slower one, and which is right is a fact about
the device rather than a preference.

Two things here are not in the other paths because a processor has them and they
do not.  The first is that the split of the work is decided while the program
runs: how many queries and how many keys one group covers is not known when the
program is written, because it comes from what the mask allows and the mask is
read at run time.  So those two counts are left as symbols and given ranges that
say "more than one", which is what stops a comparison of them against one from
deciding they are equal.

The second is that a mask says which positions may be seen, and applying that is
not the same computation as deciding it.  Deciding is what the caller's program
does; applying is turning a value into minus infinity where the answer was no.
So the caller's program is rewritten to do both, because a kernel that read a
mask and then applied it would be a kernel with two passes over it where one
would do.
"""

from __future__ import annotations

import copy
from typing import Any

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.symbolic_shapes import ValueRanges
from ..ir import FixedLayout
from ..loops import V
from ..op_lowerings import empty_strided
from ..templates.select_algorithm import autotune_select_algorithm
from .omni_flash_attention import (
    build_subgraph_buffer,
    build_subgraph_module_buffer,
    contiguous_last_dim,
    create_placeholder,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    is_tensor_ir_node,
    maybe_realize,
)

#: A position this processor can compare several at a time.
#:
#: Below this the comparisons are done one at a time and the whole point of the
#: mask being ranges rather than comparisons is lost.  The answer is the
#: machine's rather than the host's, because the host may be a different machine
#: from the one the kernel will run on.
def check_cpu_supported() -> bool:
    from ..cpu_vec_isa import pick_vec_isa

    return pick_vec_isa().name != "default"


def _realize_duplicate_buffer_inputs(inputs: Any) -> Any:
    """One buffer per name, so two views of one buffer do not become two.

    The generated code names its pointer arguments by the buffer they came from,
    so two arguments that are the same memory under two names would be written
    as two arguments and read as if they were two.  Making a copy of the second
    is the fix: it costs one copy and makes the two genuinely separate, which is
    what the caller's two names said they were.
    """

    from ..ir import ExternKernel

    names = [inp.get_name() for inp in inputs]
    duplicate_names = {n for n in names if names.count(n) > 1}
    if not duplicate_names:
        return inputs

    return [
        ExternKernel.copy_input(inp) if inp.get_name() in duplicate_names else inp
        for inp in inputs
    ]


def lower_omni_attention_cpu(
    query: Any,
    key: Any,
    value: Any,
    subgraph: Any,
    block_mask: Any,
    scale: float,
    kernel_options: Any,
    score_mod_other_buffers: Any,
    mask_mod_other_buffers: Any,
) -> Any:
    """Write the pass, for a processor, as a program that reads its own mask.

    One value back rather than three: there is no running total to hand on,
    because there is no later pass to hand it to.  A program that wanted one
    would be asking for something this does not produce, so it is said here
    rather than answered with a number that means nothing.
    """

    from ..codegen.cpp_flex_attention_template import CppFlexAttentionTemplate

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        _,
        _,
        _,
        _,
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        sparse_q_block_size,
        sparse_kv_block_size,
        mask_graph,
    ) = block_mask

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Mixed query, key, and value dtype is not supported on this platform, "
            f"got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype}."
        )

    if kernel_options["OUTPUT_LOGSUMEXP"]:
        raise NotImplementedError(
            "On a processor this is inference only, and a running total is not "
            "something it produces yet."
        )
    if not check_cpu_supported():
        raise NotImplementedError(
            "This processor cannot compare several positions at once, which is "
            "what reading a mask as ranges needs."
        )

    fake_buffers: list = []

    # How much of the work one group takes is decided while the program runs,
    # because it comes from what the mask allows and the mask is read at run
    # time.  So the two counts are symbols now and numbers later, and the
    # template turns them into what they will be.
    shape_env = V.graph.sizevars.shape_env
    cur_q_split_size = shape_env.create_unbacked_symint().node.expr
    cur_kv_split_size = shape_env.create_unbacked_symint().node.expr

    # Both counts are more than one, which is what stops a comparison of either
    # against one from deciding they are equal: two groups that each took one
    # position are two groups, not one.
    shape_env.var_to_range[cur_q_split_size] = ValueRanges(2, None)
    shape_env.var_to_range[cur_kv_split_size] = ValueRanges(2, None)

    score_dtype = tp.float
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_q_split_size, cur_kv_split_size]),
            ("b", tp.int64, []),
            ("h", tp.int64, []),
            ("q_idx", tp.int64, [cur_q_split_size, 1]),
            ("kv_idx", tp.int64, [1, cur_kv_split_size]),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    if subgraph_buffer is not None:
        if isinstance(subgraph_buffer, list):
            for buf in subgraph_buffer:
                if buf is not None:
                    buf.freeze_layout()
        else:
            subgraph_buffer.freeze_layout()

    # The mask's own placeholders are the score's: the rewritten program below
    # takes the score as its first argument, so a value of it has to exist before
    # the mask is applied to one.
    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device(), size)
        for name, dtype, size in [
            ("score", score_dtype, [cur_q_split_size, cur_kv_split_size]),
            ("b", tp.int64, []),
            ("h", tp.int64, []),
            ("q_idx", tp.int64, [cur_q_split_size, 1]),
            ("kv_idx", tp.int64, [1, cur_kv_split_size]),
        ]
    ]

    def convert_mask_graph_module(mask_graph: Any) -> Any:
        """Make the caller's mask program apply the mask as well as decide it.

        What a caller writes says which positions may be seen.  What a kernel
        needs is a score with those positions turned into minus infinity -- and
        the difference is a second pass over the value that the first one's answer
        already says where to make.  So the caller's program is given the value
        and the comparison, and asked to do both.

        The value arrives first because a program's first argument is what it is
        called with, and the value is the thing being changed.  The mask's own
        answer is still computed the same way and in the same order, so a caller
        sees no difference other than that their program now has one more
        argument and returns the value rather than the yes or no.
        """

        gm = copy.deepcopy(mask_graph.graph_module)
        graph = gm.graph
        with graph.inserting_before(next(iter(graph.nodes))):
            qk_data_node = graph.placeholder("qk_data")

        output_node = None
        for node in graph.nodes:
            if node.op == "output":
                output_node = node
                break

        if output_node is None:
            raise AssertionError("output_node must not be None")
        mask_node = output_node.args[0]

        size_node = [cur_q_split_size, cur_kv_split_size]
        # What a position the mask excluded is worth: not zero, which would be a
        # value the caller could have meant, but the identity for exponentiation,
        # so that the total of a row nothing was read for comes to nothing rather
        # than to a count of the positions that were skipped.
        with graph.inserting_after(mask_node):
            full_node = graph.call_function(
                tp.full,
                args=(size_node, -float("inf")),
                kwargs={"dtype": score_dtype},
            )

        with graph.inserting_after(full_node):
            where_node = graph.call_function(
                tp.ops.tp.where, args=(mask_node, qk_data_node, full_node)
            )

        output_node.args = (where_node,)

        graph.lint()
        return tp.fx.GraphModule(gm, graph)

    converted_mask_graph_module = convert_mask_graph_module(mask_graph)

    mask_graph_buffer = build_subgraph_module_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers),
        converted_mask_graph_module,
    )

    # The two counts were made as symbols and are given real values while the
    # program runs.  Anything that was waiting on them being decided is not
    # waiting any more, and leaving it waiting would make the next value in the
    # region depend on this kernel having been written.
    pending = shape_env.pending_fresh_unbacked_symbols
    shape_env.pending_fresh_unbacked_symbols = [
        x for x in pending if x not in (cur_q_split_size, cur_kv_split_size)
    ]

    from ..ir import TensorBox

    buffer_list = (
        placeholder_inps
        + list(score_mod_other_buffers)
        + mask_graph_placeholder_inps
        + list(mask_mod_other_buffers)
    )
    for item in buffer_list:
        if isinstance(item, TensorBox):
            fake_buffers.append(item.data.data)

    # These kernels walk the last axis one element at a time, so a last axis that
    # is not contiguous is a last axis read as a gather.  Making it contiguous is
    # a copy, and a copy here is cheaper than a gather everywhere.
    query, key, value = map(contiguous_last_dim, [query, key, value])

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        _,
        _,
        _,
        _,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            _,
            _,
            _,
            _,
        ]
    )

    query, key, value = _realize_duplicate_buffer_inputs([query, key, value])
    if query.get_dtype() not in (tp.float, tp.bfloat16, tp.float16):
        raise NotImplementedError(
            "On a processor this is written for 32-bit floats and the two "
            f"narrower ones; found {query.get_dtype()}."
        )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    bq, hq, seq_len_q, qk_head_dim = query.get_size()
    _, hkv, seq_len_kv, v_head_dim = value.get_size()

    out_size = [bq, hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, query.get_stride())

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [bq, hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    choices: list = []
    input_nodes = [query, key, value, kv_num_blocks, kv_indices]
    if not full_kv_num_blocks:
        no_full_kv_block = True
    else:
        no_full_kv_block = False
        input_nodes += [full_kv_num_blocks]
        input_nodes += [full_kv_indices]

    has_other_buffer = False
    kernel_input_name_to_buffer = {}
    if score_mod_other_buffers or mask_mod_other_buffers:
        has_other_buffer = True
        for prefix, buffers in (
            ("score_others", score_mod_other_buffers),
            ("mask_others", mask_mod_other_buffers),
        ):
            kernel_input_name_to_buffer.update(
                {f"{prefix}_{i}": buf for i, buf in enumerate(buffers)}
            )
        input_nodes += [
            v
            for v in kernel_input_name_to_buffer.values()
            if not isinstance(v, sympy.Expr)
        ]

    skip_mask_score = kernel_options.get("SKIP_MASK_SCORE", False)
    sparse_kv_block_size = V.graph.sizevars.guard_int(sparse_kv_block_size)
    sparse_q_block_size = V.graph.sizevars.guard_int(sparse_q_block_size)

    # How much of the key axis one program takes when the axis is split.  Not the
    # number of blocks -- that is the mask's business -- but how much of a block's
    # worth of keys one program walks.
    partition_size = kernel_options.get("PARTITION_SIZE", 128)
    if not V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_q, sympy.Mul(kv_indices.get_size()[-2], sparse_q_block_size))
    ):
        raise AssertionError(
            "The query axis must fit in the blocks the mask gives it on the query "
            "side; a larger block mask would fit it."
        )
    if not V.graph.sizevars.evaluate_expr(
        sympy.Le(seq_len_kv, sympy.Mul(kv_indices.get_size()[-1], sparse_kv_block_size))
    ):
        raise AssertionError(
            "The key axis must fit in the blocks the mask gives it on the key "
            "side; a larger block mask would fit it."
        )

    CppFlexAttentionTemplate.add_choices(
        choices=choices,
        input_nodes=input_nodes,
        layout=layout,
        scale=scale,
        score_mod=None if skip_mask_score else subgraph_buffer,
        mask_mod=None if skip_mask_score else mask_graph_buffer,
        kv_block_size=sparse_kv_block_size,
        q_block_size=sparse_q_block_size,
        partition_size=partition_size,
        has_other_buffer=has_other_buffer,
        no_full_kv_block=no_full_kv_block,
        fake_buffers=fake_buffers,
        len_score_other=len(score_mod_other_buffers),
        len_mask_other=len(mask_mod_other_buffers),
        kernel_input_name_to_buffer=kernel_input_name_to_buffer,
        block_vars=(cur_q_split_size, cur_kv_split_size),
    )
    res, _ = autotune_select_algorithm(
        "omni_attention",
        choices,
        [query, key, value],
        layout,
    )

    res.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    res.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (res,)
