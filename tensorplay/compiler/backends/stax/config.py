"""The knobs the code generators read while emitting code.

A generator asks a question here that has two defensible answers: a layout
that is padded to a machine's transaction width costs memory it would not
otherwise need, and one that is not padded may split a warp's access across
two transactions.  Neither answer is right for every shape and every machine,
so the answer is a setting rather than a constant, and the default is the one
that was measured to be right oftenest.

The environment may override the default of the first setting, because whether
padding is wanted depends on what the whole run is trying to do rather than on
any one kernel.
"""

from __future__ import annotations

import contextlib
import os
import sys

# Whether a loop nest's strides are padded so that every access is aligned to
# the width of a memory transaction.  Off for a region whose extents are not
# known when the code is written, since padding a symbolic stride cannot be
# undone later.
comprehensive_padding = os.environ.get("TP_COMPREHENSIVE_PADDING", "1") == "1"

# Whether the extents of a matrix product are rounded up to a multiple of a
# transaction width, so that each row of one operand starts where a wide
# access expects.  Costs work proportional to what the rounding adds and buys
# the accesses being able to be wide, which is a trade worth making for a
# product large enough to be limited by memory and not for a small one.
shape_padding = os.environ.get("TP_SHAPE_PADDING", "1") == "1"

# Whether to round the extents up whatever else was decided.  For reading the
# effect of the rounding on its own, with the trade above not deciding it.
force_shape_pad: bool = False

# How many values a concatenation of pointwise work may hold before it is
# worth doing as one operation over all of them rather than one per value.
# Past this many the loop that walks them costs more than the operations.
max_pointwise_cat_inputs = 8

# Whether a reduction long enough to leave the machine idle is worth cutting
# into pieces that are reduced independently.  Turning this off makes every
# reduction a single walk, which is slower but makes the code easier to read.
split_reductions = os.environ.get("TP_SPLIT_REDUCTIONS", "1") == "1"

# A reduction shorter than this is cheaper to walk straight through than to
# set up the bookkeeping that splitting it would need.
unroll_reductions_threshold = int(
    os.environ.get("TP_UNROLL_REDUCTIONS_THRESHOLD", "8")
)

# Whether the loops are ordered only after fusion decisions have been made,
# rather than before.  A write whose index expressions are in a different
# order than the read they are matched against would otherwise look unlike it.
loop_ordering_after_fusion = (
    os.environ.get("TP_LOOP_ORDERING_AFTER_FUSION", "0") == "1"
)

# Whether the order the loops are walked in is worked out at all, rather than
# left as the order the body was written in.  On because walking a body in the
# order memory is laid out is what makes each step read what the step before
# left in hand; the order is a heuristic and has not been tuned.
pick_loop_orders = os.environ.get("TP_PICK_LOOP_ORDERS", "1") == "1"

# A tensor already laid out with its channels last keeps that layout: padding
# it would defeat the layout it was given.
pad_channels_last = False

# Whether the layouts of a region containing convolutions may be chosen rather
# than taken as they are given.  Off where the accelerator it targets does not
# gain from the layout it would choose.
layout_optimization = True

# Whether that choice is made whatever the region looks like, for the case
# where a measurement says it pays and the shape of the region would not say so.
force_layout_optimization = os.environ.get("TP_FORCE_LAYOUT_OPT", "0") == "1"

# Whether strides are padded when the extents they belong to are symbolic.
pad_dynamic_shapes = False

# Whether a result is padded even where it would not be padded on its own.  Off
# because a result is measured against the strides it was given, and padding one
# produces strides the caller was not promised.
pad_outputs = False

# The width a padded access is aligned to, in bytes.  A warp's largest memory
# transaction is 128 bytes, so aligning to that is the finest alignment that
# can still be reached by one transaction.
padding_alignment_bytes = 128

# The smallest stride worth padding.  Below this, padding grows the buffer by
# more than the alignment saves: with an alignment of 16 elements, a stride of
# 320 is at most five percent of the buffer, but a stride of eight is more
# than half of it, and the many small blocks that result are accessed less
# efficiently than the padding was worth.  The threshold is also what keeps
# padding away from a persistent reduction, whose inner extent is deliberately
# not a multiple of a tile: a stride past the threshold would be padded, the
# layout would stop being contiguous, and the reduction would stop being
# eligible to be persistent.
padding_stride_threshold = 1024


class _TestConfigs:
    """Switches that make the generated code check the generator.

    Each one has the emitter write a check of its own output into the output:
    a check the generated text can be wrong about, which is the only way to
    find out that it is.  They are off because a kernel that checks itself
    costs more than it is worth in a run that is not looking for that class of
    mistake.
    """

    #: Have an emitter that runs on the accelerator state the element type it
    #: believes a value has, as an assertion in the generated code.
    runtime_triton_dtype_assert = False

    #: Have the host emitter state the element type it believes a value has,
    #: as a compile-time assertion in the generated code.
    static_cpp_dtype_assert = False

    #: Have an emitter that runs on the accelerator state the shape it believes
    #: a value has, as an assertion in the generated code.
    runtime_triton_shape_assert = False
    #: Make a measured time deliberately wrong, to check that a decision which
    #: rests on a measurement actually does rest on it.  Which way it is
    #: distorted, and by how much, is written here rather than derived.  Empty
    #: means do not distort, which is a value of its own rather than the absence
    #: of one: whether a measurement may be faked is a question with two answers,
    #: and the answer to it is this.
    distort_benchmarking_result = os.environ.get(
        "TP_DISTORT_BENCHMARKING_RESULT", ""
    )

    #: Write an assertion into the generated code for every value whose memory
    #: the wrapper hands out, so that a use after free is caught where it
    #: happens rather than where it is noticed.  "log" writes the events
    #: instead of asserting on them.
    track_memory_lifecycle = None

    #: How many shapes a matmul may be measured over.  None means no limit, and
    #: a limit is what keeps an exhaustive search from running all night.
    max_mm_configs = None

    #: The same limit for the shapes a grouped matmul may be measured over.
    max_flex_configs = None

    #: Restrict which measured candidates may be chosen from, by name and by
    #: description.  Both are how a particular machine's answers are pinned
    #: when a measurement would otherwise differ from run to run.
    autotune_choice_name_regex = os.environ.get("TP_AUTOTUNE_CHOICE_NAME_REGEX")
    autotune_choice_desc_regex = os.environ.get("TP_AUTOTUNE_CHOICE_DESC_REGEX")

    #: Force a decomposition, or refuse one, regardless of what the shapes
    #: would otherwise decide.
    force_custom_op_decomposition = None

    #: Keep a kernel that produces several results in one group of pieces,
    #: rather than splitting them.  Off by default because a group this way
    #: constrains the shapes the pieces can have.
    force_extern_kernel_in_multi_template = False

    #: Refuse to narrow a reduction's list of measured candidates.  Off by
    #: default, since narrowing is what makes the search affordable.
    force_filter_reduction_configs = False

    #: Emit every piece on its own rather than grouping pieces that share
    #: nothing.  Off by default, since grouping is the point.
    force_no_impl_grouping = False

    #: Assume that putting independent work into one launch is cheaper than
    #: running the launches one after another.  True by default, because
    #: measuring the grouping itself is not always possible.
    assume_bucketing_reduces_latency = True

    #: Bisect on whether a custom backend is worth keeping.  Off by default,
    #: since the bisection runs the search twice.
    bisect_keep_custom_backend_for_inductor = False

    #: Bisect on where a graph should be split between the two graphs.  Off by
    #: default, for the same reason.
    bisect_pre_grad_graph = False

    #: Have a graph-safe generator ignore the fallback generator.  Off by
    #: default, since the fallback is what makes a graph compile at all.
    graphsafe_rng_func_ignores_fallback_random = False

    #: Read measurements from the shared library's own cache.  Off by default,
    #: since the two caches do not agree about what a name means.
    use_libtorch = False


test_configs = _TestConfigs()
test_configs.track_memory_lifecycle = False


#: Whether a run is checking that no value it computed came out as a
#: not-a-number or an infinity, which is what a kernel reading a value that was
#: never written looks like from the inside.
runtime_triton_nan_asserts = False

#: Whether the shape of every value an emitter produces is written into the
#: generated code as an assertion, which is a way of finding out whether the
#: emitter's idea of a shape is the shape the value has.
debug_index_asserts = False

#: Whether an index is checked against the extent it indexes at all, as opposed
#: to only when the emitter is already checking something else there.
assert_indirect_indexing = True

#: Whether a result's memory is checked at run time to be aligned to the
#: width a wide load moves.
alignment_asserts = False

#: Whether a result's shape and stride are checked at run time against
#: what the call was written for.
size_asserts = True

#: Whether each node records where in the lowering it was made.
debug_ir_traceback = False

#: Whether a zero of either sign is told apart in a maximum or a minimum.  Off
#: because the two zeros compare equal, so which one a maximum returns would
#: otherwise depend on the order the values were combined in.
strict_signed_zero = False

#: Whether the position a value is computed from is carried through the body as an
#: expression, rather than being recomputed at each use.
constant_and_index_propagation = True

#: Whether the bounds of every value are computed, as opposed to only those that
#: something asks about.  Computing them all makes the generated code tighter
#: and the run slower.
compute_all_bounds = True


#: Whether a value held in half precision is computed in single precision.  It
#: changes the numerics of the emitted code and not the type of the value, so
#: it is a setting of the backend rather than of the program.
codegen_upcast_to_fp32 = True


class _CppConfig:
    #: Write a wrong answer on purpose, to check that a test would notice.
    #: The value names the mistake: a relu that can return a negative number,
    #: or a logarithm that can return something other than its input's
    #: logarithm.  Left unset, which is the only way to ship.
    inject_relu_bug_TESTING_ONLY: str | None = None
    inject_log1p_bug_TESTING_ONLY: str | None = None

    """The knobs the host emitter reads while emitting.

    These are the settings where more than one answer is defensible: whether
    the thread count is fixed at compile time or asked of the runtime decides
    whether a loop may be split at all, how small a piece of work has to be
    before splitting it pays decides the same question again, and whether a
    kernel is named after what it computes or after where it came from decides
    only what a report says.  The defaults are the ones that were measured to
    be right oftenest.
    """

    #: Whether the thread count is asked of the runtime rather than fixed when
    #: the kernel is written.  A kernel that is handed its thread count has the
    #: same code whatever the machine runs it on; one that asks has a loop whose
    #: bounds the runtime chooses, which is what lets one compiled kernel serve
    #: machines of different widths.
    dynamic_threads = False

    #: The fewest elements one thread is given.  Below this the cost of starting
    #: the thread is more than the work it does.
    min_chunk_size = 512

    #: How many threads the kernels are written for, when the count is fixed.
    threads = 1

    #: Whether a kernel is named after what it computes rather than after the
    #: loop nest it came from.  A report that says what was computed is easier to
    #: read; a cached kernel is easier to find when it is named after where it
    #: was written.  Off, which is the same as naming it after neither.
    #:
    #: On, this says which operation the name is built from: ``"tp"`` for the
    #: one this project's namespace spells, ``"original_aten"`` for the one the
    #: graph was captured with before it was decomposed, and
    #: ``"inductor_node"`` for the graph node's own name.
    descriptive_names: bool | str = False

    #: Whether a reduction whose extent is past a threshold accumulates through
    #: a helper that keeps the running sum in a wider type.  The result is
    #: different -- it is the more accurate one -- and it costs.
    enable_loop_tail_vec = False

    #: Whether the emitted kernel wraps its body in a guard that checks the
    #: device, which is what keeps a kernel that is launched on the wrong device
    #: from reading the wrong memory.
    enable_kernel_context_guard = True

    #: Whether a kernel records how long it took.
    enable_kernel_profile = False

    #: Whether the kernel is written so that a compiler will inline it into its
    #: caller, which lets it keep values in registers across the call.
    force_inline_kernel = False

    #: Whether a loop whose bounds are already known is left out of the emitted
    #: code, on the grounds that a loop of one trip contributes nothing.
    no_redundant_loops = True

    #: How many loop nests are fused into one kernel horizontally.
    max_horizontal_fusion_size = 8

    #: Whether the tiling of a loop nest is chosen by measuring candidates or by
    #: the first one that fits.
    enable_tiling_heuristics = False

    #: Whether a log line is written when a kernel is emitted.
    inject_log = False

    #: Whether the decomposition of the hyperbolic tangent is used where the
    #: language has one of its own.
    use_decompose_tanh = False


cpp = _CppConfig()

# Whether claims about numbers that come from the data are checked when the
# code runs.  The claims cost a comparison per run, so they are worth turning
# off once the code has been shown not to need them.
scalar_asserts = os.environ.get("TP_SCALAR_ASSERTS", "1") == "1"

# Whether a value made only of constants is folded by running the operations
# at run time rather than while the code is being written.  Doing it at run
# time is what lets a constant be a value that is only known once the program
# is running, at the cost of doing the work on every run.
use_runtime_constant_folding = (
    os.environ.get("TP_USE_RUNTIME_CONSTANT_FOLDING", "0") == "1"
)

# Whether the result of a call that is run through to is assumed to start at a
# boundary that a wider access may assume.  Assuming it when it does not hold
# is faster and wrong; not assuming it costs a check on every access.
assume_unaligned_fallback_output = (
    os.environ.get("TP_ASSUME_UNALIGNED_FALLBACK_OUTPUT", "0") == "1"
)

# Whether work that would otherwise be done after a hand-written kernel can be
# done as part of it.  This needs to know which value the kernel wrote, and
# writing it in one place, so it only applies to some kernels.
epilogue_fusion_user_defined_triton_kernel = (
    os.environ.get("TP_EPILOGUE_FUSION_USER_DEFINED_TRITON_KERNEL", "1") == "1"
)

# Whether a warning about the program as given stops the compilation instead
# of being printed.  Such a warning means the result will not be what was
# presumably wanted, so treating it as fatal is how that gets noticed rather
# than read past in a log nobody reads.
raise_on_developer_warning = os.environ.get("TP_RAISE_ON_DEVELOPER_WARNING", "0") == "1"

# Once every remaining piece of work is bigger than this, distinguishing them
# by whether they fit under the memory high-water mark says nothing useful, and
# the order is decided by which piece unblocks its successors soonest instead.
size_threshold_for_succ_based_strategy = int(
    os.environ.get("TP_SIZE_THRESHOLD_FOR_SUCC_BASED_STRATEGY", "1048576")
)

# Whether the order of the work may be changed to need less memory.  A better
# order can mean more time, since what runs together changes.
reorder_for_peak_memory = os.environ.get("TP_REORDER_FOR_PEAK_MEMORY", "1") == "1"

# Whether to write down the order that was chosen, for a simulator to be run
# against as a check that does not require running the thing for real.
reorder_for_peak_memory_debug = (
    os.environ.get("TP_REORDER_FOR_PEAK_MEMORY_DEBUG", "0") == "1"
)

# Whether to work out whether a join actually makes things faster by running
# both versions, rather than deciding from what they share.  Measuring is
# accurate and costs the time of compiling and running each candidate, so it is
# off unless the program is being tuned for speed rather than compiled quickly.
benchmark_fusion = os.environ.get("TP_BENCHMARK_FUSION", "0") == "1"

# Whether work that adds to a value rather than overwriting it may be folded
# into the end of a template.  The two are written differently, so a template
# has to be one that can take the first kind.
epilogue_fusion_with_atomic_add = (
    os.environ.get("TP_EPILOGUE_FUSION_WITH_ATOMIC_ADD", "0") == "1"
)

# Which of the ways of writing code for the host this compilation is using.
# Anything but the compiled one has no kernel to measure.
cpu_backend = os.environ.get("TP_CPU_BACKEND", "cpp")

# Whether the candidates are measured at all.  Off means the operation's own
# kernel is used, which is correct everywhere and fast nowhere in particular; on
# means every candidate is measured and the fastest is used, which costs the
# measurement on every compile and saves it on every run after.
# How eagerly a value that is read more than once is written to memory.
#
# Rematerializing a value that several consumers read costs the work once per
# consumer, so a value whose body is large enough is stored instead.  The
# threshold on the body's size is left unset here so that a target with its own
# answer can supply one, and the read count bounds the fanout case where
# rematerializing would multiply the work rather than repeat it.
_realize_opcount_threshold_default = 30
realize_opcount_threshold: "int | None" = None
realize_reads_threshold = 4
realize_opusers_threshold = 5
# A body written for a processor tolerates being larger than one written for an
# accelerator, and putting a moderate expression in memory costs a whole buffer's
# worth of traffic, which is why the two have answers of their own.
realize_cpu_opcount_threshold = 50

#: Whether a tile shape is narrowed by walking the shape down rather than only
#: by measuring whole candidates.
coordinate_descent_tuning = os.environ.get("TP_COORDINATE_DESCENT_TUNING") == "1"

max_autotune = os.environ.get("TP_MAX_AUTOTUNE", "0") == "1"

# Whether products are measured even when the rest is not.  Separate because a
# product is the one call worth spending a measurement on: it runs inside every
# layer, so what it costs is paid by the whole model rather than by one line.
max_autotune_gemm = os.environ.get("TP_MAX_AUTOTUNE_GEMM", "0") == "1"

# Which backends a measured product may be chosen from, as a list to add to or
# take from rather than a switch.  Naming them is what makes a measurement
# answerable: a product measured against the library's own kernel and a product
# measured against a set of tiles are two different questions, and which one is
# being asked is said here rather than inferred from which switches are on.
max_autotune_gemm_backends = os.environ.get(
    "TP_MAX_AUTOTUNE_GEMM_BACKENDS", "FRAMEWORK,TRITON,CPP"
).upper()

# Which backends a measured convolution may be chosen from, for the same reason.
max_autotune_conv_backends = os.environ.get(
    "TP_MAX_AUTOTUNE_CONV_BACKENDS", "FRAMEWORK,TRITON,CPP"
).upper()

# Whether a product is compiled as soon as it is chosen, so that the measuring
# of the candidates overlaps with the measuring of the next thing.  Off because
# compiling a product needs a graph of its own, which is work done whether or not
# the product is ever chosen.
pipeline_max_autotune_gemm = (
    os.environ.get("TP_PIPELINE_GEMM_AUTOTUNING") == "1"
)

# Whether the time a candidate takes is read off a profiler trace rather than off
# a clock around it.  A trace says how long the device was busy, which is not the
# same as how long the launch took, and for a candidate that waits its two are far
# apart.  Off because a trace costs more to read than a clock costs to read.
profile_bandwidth_with_do_bench_using_profiling = (
    os.environ.get("TP_PROFILE_WITH_DO_BENCH_USING_PROFILING") == "1"
)

# Whether the answer must be the same on every machine.  On means nothing is
# measured and the operation's own kernel is used, because the only candidate
# whose answer does not depend on which machine compiled it is that one.
deterministic = os.environ.get("TP_DETERMINISTIC", "0") == "1"

# Whether measuring happens in a process of its own.  On costs a process per
# measurement and keeps a candidate that crashes from taking the compiler with
# it, which is the trade when a candidate is suspected of misbehaving.
autotune_in_subproc = os.environ.get("TP_AUTOTUNE_IN_SUBPROC", "0") == "1"

# Whether the product's candidates are searched exhaustively or by the table.
# Exhaustive costs far more to compile and finds configurations the table does
# not have, which is worth it for a shape that runs for a long time.
max_autotune_gemm_search_space = os.environ.get(
    "TP_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
)

# Whether the kernels written in the device dialect are measured at all.  They
# are off by default because a configuration that will not launch is not one
# that measured badly, and a search wide enough to include those spends its time
# finding that out; so a program that has not asked for measuring is given a
# single known-good configuration rather than a search.
cutedsl_enable_autotuning = os.environ.get(
    "TP_CUTEDSL_ENABLE_AUTOTUNING", "0"
) == "1"

# How long a candidate may spend being prepared before it is given up on.  A
# candidate that takes longer than this is one whose result would not be worth
# waiting for, since it would be paid on every run.
precompilation_timeout_seconds = int(
    os.environ.get("TP_PRECOMPILATION_TIMEOUT_SECONDS", str(60 * 60))
)

# How long the contracted axis has to be, relative to the others, before
# splitting it is worth offering.  A split of a short axis is a sum of small
# products, which is more work than the product was; the point of splitting is to
# do the parts side by side, and that needs parts worth doing.
decompose_k_threshold = int(os.environ.get("TP_DECOMPOSE_K_THRESHOLD", "128"))

# How many ways of splitting the contracted axis are available.  Zero means
# none, which is the same as saying the split template is not offered at all.
num_decompose_k_splits = int(os.environ.get("TP_NUM_DECOMPOSE_K_SPLITS", "4"))

# Where the measurements a learned rule is built from are written.  A path of
# "DEFAULT" means beside the program's own cache, and anything else is that path.
# Writing them is off unless this names one, because a record of every
# measurement is a large file that nothing reads unless a rule is being built.
autoheuristic_log_path = os.environ.get("TP_AUTOHEURISTIC_LOG_PATH", "None")

# Which calls a learned rule is collected for, and which are answered by one.
# A name is a call's own name, and "DEFAULT" means the ones that have no name of
# their own.  Collecting and using are separate because a rule has to be built
# before it can be used, and a program that only uses rules has nothing to
# collect.
def collect_autoheuristic(name: str) -> bool:
    """Whether measurements for a call are being kept for a rule to learn from."""
    if autoheuristic_log_path == "None":
        return False
    import os as _os

    enabled = _os.environ.get("TP_AUTOHEURISTIC_COLLECT", "DEFAULT")
    return enabled == "ALL" or name == enabled


def run_autoheuristic(name: str) -> bool:
    """Whether a call is answered by a rule, whether learned or being learned.

    Both halves of the same question: whether to consult what has been
    learned, and whether to learn from this call.  Learning and consulting
    are asked separately because one is usually wanted without the other.
    """

    return collect_autoheuristic(name) or use_autoheuristic(name)


def use_autoheuristic(name: str) -> bool:
    """Whether a call is answered by a learned rule rather than by measuring."""
    import os as _os

    enabled = _os.environ.get("TP_AUTOHEURISTIC_USE", "NONE")
    return enabled == "ALL" or (enabled != "NONE" and name == enabled)


# Whether a call whose kernel is one element wide is computed as a product.
# On by default because it has more ways of being computed than a call does; off
# for a program that wants every call to be a call.
conv_1x1_as_mm = os.environ.get("TP_CONV_1X1_AS_MM", "1") == "1"

# Which ways of computing a call are measured.  Empty means the operation's own
# kernel only, which is correct everywhere and fast nowhere in particular; naming
# the framework's own keeps the floor in the measurement, and naming a backend
# adds its templates to what is measured.
max_autotune_conv_backends = os.environ.get(
    "TP_MAX_AUTOTUNE_CONV_BACKENDS", "EAGER,TRITON"
)

# Which ways of computing a product may be measured.  Every one of them is a
# candidate when the measurement was asked for, and the operation's own kernel
# is only among them if it is named here or the measurement was not asked for --
# a measurement without it would be comparing a template against nothing.
max_autotune_backends = os.environ.get("TP_MAX_AUTOTUNE_BACKENDS", "EAGER,TRITON")

# Which candidates may be measured, by name and by description.  Empty means all
# of them.  A way of asking for one candidate to be measured without taking the
# others off the list, which is not the same as removing them.
autotune_choice_name_regex = os.environ.get("TP_AUTOTUNE_CHOICE_NAME_REGEX") or None
autotune_choice_desc_regex = os.environ.get("TP_AUTOTUNE_CHOICE_DESC_REGEX") or None

# Whether to measure the work folded into the end of a template, and how many
# ways of doing that are worth measuring.  Beyond that many, the fastest is
# taken on trust rather than measured.
benchmark_epilogue_fusion = os.environ.get("TP_BENCHMARK_EPILOGUE_FUSION", "0") == "1"
max_epilogue_benchmarked_choices = int(
    os.environ.get("TP_MAX_EPILOGUE_BENCHMARKED_CHOICES", "8")
)

# Whether work may be folded into the end of a prepared kernel, and into the
# start of one.  Both change what the prepared kernel has to be able to do, so
# both are off where nothing says otherwise.
epilogue_fusion = os.environ.get("TP_EPILOGUE_FUSION", "1") == "1"
prologue_fusion = os.environ.get("TP_PROLOGUE_FUSION", "1") == "1"

# How many pieces one launch may hold.  Past this the code is too large to
# fit what a program may hold, and a launch that does not fit is worse than
# two that each do.
max_fusion_size = int(os.environ.get("TP_MAX_FUSION_SIZE", "128"))

# How much two pieces must share before joining them is worth it.  Below this
# the join saves little and makes the launch larger.
score_fusion_memory_threshold = int(
    os.environ.get("TP_SCORE_FUSION_MEMORY_THRESHOLD", "0")
)

# Whether to look at pairs that share no value as well.  These are only joined
# where they walk the same axes, which saves nothing in memory, so the answer
# is usually no.
aggressive_fusion = os.environ.get("TP_AGGRESSIVE_FUSION", "0") == "1"

# How many values one launch may be given.  Nothing here, for a launch may be
# given as many as it needs.
max_fusion_unique_io_buffers = int(
    os.environ.get("TP_MAX_FUSION_UNIQUE_IO_BUFFERS", "8")
)

# Whether a piece walking fewer elements may be made to look as though it
# walked more, so that it can be joined to one that did.
expand_dimension_for_pointwise_nodes = (
    os.environ.get("TP_EXPAND_DIMENSION_FOR_POINTWISE_NODES", "0") == "1"
)

# Whether the axes may be walked in a different order, or a value read the
# other way round, once it is settled which pieces are joined.  Both can make a
# pair joinable that was not, at the cost of trying arrangements.
loop_reindexing_after_fusion = (
    os.environ.get("TP_LOOP_REINDEXING_AFTER_FUSION", "0") == "1"
)
loop_index_inversion_in_fusion = (
    os.environ.get("TP_LOOP_INDEX_INVERSION_IN_FUSION", "0") == "1"
)

# Whether to work out whether a set of independent pieces is faster together
# than apart by running both, how many ways of dividing one up are worth
# measuring, and whether only the pieces that compute anything are eligible.
# A piece whose positions come from the data cannot be measured on made-up
# values, since those may not be valid.
benchmark_combo_kernel = os.environ.get("TP_BENCHMARK_COMBO_KERNEL", "0") == "1"
combo_kernels_pointwise_only = (
    os.environ.get("TP_COMBO_KERNELS_POINTWISE_ONLY", "0") == "1"
)
combo_kernels_autotune = int(os.environ.get("TP_COMBO_KERNELS_AUTOTUNE", "0"))
combo_kernel_per_subkernel_blocks = (
    os.environ.get("TP_COMBO_KERNEL_PER_SUBKERNEL_BLOCKS", "0") == "1"
)
combo_kernel_compile_time_autotune = (
    os.environ.get("TP_COMBO_KERNEL_COMPILE_TIME_AUTOTUNE", "0") == "1"
)

#: How many differently shaped blocks may share one kernel.  Mixing shapes lets
#: a group with one awkward member still be launched as one kernel, at the cost
#: of the shapes not being uniform.
combo_kernel_allow_mixed_sizes = 1

#: Say a warning when a generated kernel reads two layouts at once, which is
#: worth knowing when a kernel is slower than it looks.
warn_mix_layout = os.environ.get("TP_WARN_MIX_LAYOUT") == "1"

# How many pieces one launch of independent work may hold.  Each one is a
# launch of its own otherwise, and beyond a point the gain is smaller than the
# cost of writing them all out.
combo_kernel_max_num_nodes = int(os.environ.get("TP_COMBO_KERNEL_MAX_NUM_NODES", "8"))


# ---------------------------------------------------------------------------
# 供代码生成器读取的开关
# ---------------------------------------------------------------------------

#: Have the generated code assert, as it runs, that no value it was handed is
#: a NaN.  Off by default: the assertion costs a comparison per element, and a
#: program whose arithmetic produces NaNs is usually one whose shapes were
#: wrong, which the shape assertions would have caught first.
nan_asserts = os.environ.get("TP_NAN_ASSERTS") == "1"

#: Let a buffer be written over in place once nothing else wants it.  This is
#: the larger of the two questions about a buffer's memory -- whether it may be
#: shared with another buffer at all -- and turning it off costs memory for
#: certainty.
allow_buffer_reuse = True

#: Work out the order values are allocated in, rather than allocating each as
#: it is asked for.  It is what lets two values share one buffer.
memory_planning = os.environ.get("TP_MEMORY_PLANNING", "1") == "1"

#: Which pool the values live in.  "intermediates" is the pool for values that
#: exist only between two launches; "cudagraphs" is for a graph captured whole,
#: where nothing may be handed back to the allocator at all.
memory_pool = os.environ.get("TP_MEMORY_POOL", "intermediates")

#: Write the generated code so that it can be compiled on its own, rather than
#: only called from a wrapper.  Off by default, since the separate compilation
#: costs a step and buys only the ability to look at the code on its own.
cpp_wrapper = os.environ.get("TP_CPP_WRAPPER", "0") == "1"

#: Mark the regions of a model in the generated code, so a profile says which
#: part of the model a launch came from.
annotate_training = os.environ.get("TP_ANNOTATE_TRAINING", "0") == "1"

#: Mark each call of the wrapper in a profile, so a profile says how much of
#: the time was spent entering it rather than in the launches it made.
profiler_mark_wrapper_call = False

#: Call a hook after every piece, so a run can watch values as they are
#: produced.  Off by default, since the hook is a call per piece.
generate_intermediate_hooks = False

#: How a value is measured when the decision rests on a measurement rather than
#: on what the code does.  An empty string means not measured at all.
profile_bandwidth = os.environ.get("TP_PROFILE_BANDWIDTH", "") != ""
profile_bandwidth_output = os.environ.get("TP_PROFILE_OUTPUT", None)

#: Which measurements a run may trust.  "cudagraphs" is off by default because
#: a captured graph hides the launches it replaced, so a launch that was
#: measured after capture was not measured at all.
autotune_cudagraph_benchmarking = (
    os.environ.get("TP_AUTOTUNE_CUDAGRAPH_BENCHMARKING", "0") == "1"
)

#: How many times a measured candidate is run before its time is believed, and
#: how many times before the measurement starts.  The first run of a kernel
#: pays for loading it, so its time says nothing about the kernel.
inductor_default_autotune_warmup = int(
    os.environ.get("TP_DEFAULT_AUTOTUNE_WARMUP", 25)
)
inductor_default_autotune_rep = int(os.environ.get("TP_DEFAULT_AUTOTUNE_REP", 100))

#: Which measurement a run uses.  The built-in one times a callable directly;
#: the experimental one reports what the profiler saw, which costs more and
#: says more.
use_experimental_benchmarker = (
    os.environ.get("TP_USE_EXPERIMENTAL_BENCHMARKER", "1") == "1"
)

#: Take the time from a profiler trace rather than from a timer.  Off by
#: default, since a trace is much more expensive to collect.
use_torch_profiler_benchmarker = (
    os.environ.get("TP_USE_PROFILER_BENCHMARKER", "0") == "1"
)

#: Build and run the harness that measures a kernel, rather than measuring it
#: where it already stands.  Off by default, since it duplicates the program.
benchmark_harness = False

#: Measure a kernel when it is written.  Off by default: it is a debugging
#: answer, and it makes every run slow.
benchmark_kernel = os.environ.get("TP_BENCHMARK_KERNEL", "0") == "1"

#: Give a value no type of its own, inferring it from the arithmetic applied
#: to it.  On by default, since an explicit type is one more thing to keep in
#: step with the arithmetic.
_use_fp64_for_unbacked_floats = True

#: Which counters are gathered.  An empty string gathers none.
enabled_metric_tables = os.environ.get("TP_ENABLED_METRIC_TABLES", "")

#: Split the program into graphs, so that each may be launched on its own.  On
#: by default, since a graph is the unit a capture takes.
graph_partition = os.environ.get("TP_GRAPH_PARTITION", "1") == "1"


class _AotIConfigs:
    """The switches that only mean anything to a graph exported ahead of time.

    An exported graph is run without the Python that built it, so what it may
    assume about its caller is narrower than what a directly-run one may.
    """

    #: Print a value as it is produced, at a level above zero.
    debug_intermediate_value_printer = 0

    #: Keep the launcher beside the graph rather than emitting it separately,
    #: so that a run may be inspected with the code that launches it.
    local_wrappers = True

    #: Use the runtime shipped with the build rather than one built alongside
    #: the graph.  Off by default, since the two are not interchangeable.
    use_runtime_cache = False

    #: Check, at run time, that every assumption the graph was compiled under
    #: still holds.  On by default, since a graph that outlives its assumptions
    #: is the failure this exists to catch.
    runtime_asserts = True

    #: Read the graph's own memory rather than passing what it allocated on to
    #: the runtime's allocator.  Off because a graph that outlives the
    #: allocator it was built against is the failure this avoids.
    allow_stack_allocation: bool = False

    #: Give each kernel its own allocation while it is being tuned, rather than
    #: sharing one across every kernel of a launch.  Costs memory, and keeps one
    #: kernel's tuning from being decided by another's footprint.
    autotune_per_kernel_alloc: bool = False

    #: Which kernels a listing may name.  Left unset, every kernel is named.
    filtered_kernel_names = os.environ.get("TP_FILTERED_KERNELS_TO_PRINT", None)

    #: Fold the constants a graph computes at run time, rather than computing
    #: them once when it is built.  Off because a graph run many times is
    #: cheaper to fold once, and a graph run once is not.
    use_runtime_constant_folding: bool = False


aot_inductor = _AotIConfigs()


class _TritonConfig:
    """The switches that only mean anything to the kernel-writing runtime.

    Everything here is read by code that hands a kernel to that runtime rather
    than by code that runs here, so each is either about what to write into the
    generated module or about where the runtime should put what it produces.
    """

    #: Give a user's kernel a name of its own rather than the name it was
    #: written under, so that two kernels written under one name do not collide
    #: in the runtime's cache.
    unique_user_kernel_names = (
        os.environ.get("TP_UNIQUE_USER_KERNEL_NAMES", "0") == "1"
    )

    #: Keep the machine code the runtime produced beside the cache entry, which
    #: is what a reader needs when the entry itself will not load.
    store_cubin = False

    #: Write down a trace of each launch as it runs.
    proton_profiling: bool = (
        os.environ.get("TP_TRITON_PROTON_PROFILING", "0") == "1"
    )
    #: Where the trace is written.  Left unset, it goes beside the cache.
    proton_output_dir: str | None = os.environ.get("TP_TRITON_PROTON_OUTPUT_DIR")
    #: Put the work groups of a trace in the order the machine ran them,
    #: rather than one after another, which is what makes a trace readable
    #: against the machine.
    proton_group_by_sm: bool = (
        os.environ.get("TP_TRITON_PROTON_GROUP_BY_SM", "1") == "1"
    )
    #: Start a trace file per launch, rather than one file for the whole run.
    proton_split_invocations: bool = (
        os.environ.get("TP_TRITON_PROTON_SPLIT_INVOCATIONS", "1") == "1"
    )
    #: Fold the tracks of the warps of a work group into one track for the
    #: group, so a trace shows what the group did rather than what each of its
    #: warps did.
    proton_per_cta_occupancy: bool = (
        os.environ.get("TP_TRITON_PROTON_PER_CTA_OCCUPANCY", "1") == "1"
    )

    #: Measure each pointwise launch's tuning, rather than only the tunings
    #: of the launches that are worth the measurement.
    autotune_pointwise = True

    #: Read a position as one number per element rather than as an offset into
    #: a flat range.  Off because a dense read is only shorter where the whole
    #: range is read, and it is longer everywhere else.
    dense_indexing = False

    #: Do not overlap a load with the arithmetic that consumes it when the
    #: sizes are not known until the launch runs.  On because overlapping
    #: needs the sizes.
    dynamic_disable_pipelining = True

    #: Write a pointer's range as a 32-bit number.  Off because a buffer
    #: larger than that addresses would be written to the wrong place.
    emit_pointer_range_32 = (
        os.environ.get("TP_EMIT_POINTER_RANGE_32", "1") == "1"
    )

    #: Build the descriptor for a block on the host rather than in the launch,
    #: which takes the descriptor's construction off the critical path.
    enable_host_side_tma = os.environ.get("TP_ENABLE_HOST_SIDE_TMA", "0") == "1"

    #: Let a launch start before the one it depends on has finished, which
    #: overlaps them.  Off because the overlap is only correct when the two
    #: write and read in a stated order.
    enable_pdl = os.environ.get("TP_ENABLE_PDL", "0") == "1"

    #: The smallest reduction block a scan may be given, below which the scan
    #: is not worth splitting.
    min_split_scan_rblock = 256

    #: Let a mix-order reduction use more than one stage, which shares memory
    #: between the stages and so can run out of it.
    mix_order_reduction_allow_multi_stages = (
        os.environ.get("TP_MIX_ORDER_REDUCTION_ALLOW_MULTI_STAGES", "1") == "1"
    )

    #: Keep a load out of the first-level cache.  Off because a launch that
    #: reads the same place twice wants it there.
    skip_l1_cache = os.environ.get("TP_SKIP_L1", "0") == "1"

    #: How much a tuning may spill before it is not worth using.  A launch
    #: that spills spills on every run, so a small allowance is right.
    spill_threshold: int = 16

    #: Transpose a descriptor whose layout disagrees with the block it
    #: describes, rather than reading the block the layout says.
    transpose_discontiguous_tensor_descriptor = True

    #: How many kernels one launch may hold.  Above one, several kernels share
    #: a launch and are told apart inside it, which is worth doing when a
    #: launch's fixed cost is a large part of what it costs.
    multi_kernel: int = int(os.environ.get("TP_MULTI_KERNEL", "0"))

    #: Write the graph down as it is lowered, which is how a shape that came
    #: out wrong is traced back to the node that made it wrong.
    debug_sync_graph = False

    #: Launch a captured graph rather than calling into it, so the launches
    #: themselves are what a profiler sees.
    cudagraphs = os.environ.get("TP_CUDAGRAPHS") == "1"

    #: Round an extent that is a multiple of sixteen up to a multiple of
    #: sixteen, which is what lets a wide load stay aligned.  On because a wide
    #: load that is not aligned is the slower of the two.
    divisible_by_16 = os.environ.get("TP_DIVISIBLE_BY_16", "1") == "1"

    #: Read through a block pointer rather than through a row of offsets.
    use_block_ptr = False

    #: Hand a block to the runtime as a descriptor it builds once, rather than
    #: as a pointer it walks per access.  Worth it for a block read many times
    #: and not worth the setup for one read once, so this leaves the choice to
    #: the code that knows how often the block is moved.
    use_tensor_descriptor = False

    #: Compute a narrower-than-float value in float and narrow it at the end,
    #: rather than computing it in its own type throughout.
    codegen_upcast_to_fp32 = True

    #: Let the warps of a launch cooperate on one reduction rather than each
    #: reducing a part of it and a later step combining the parts.
    cooperative_reductions = (
        os.environ.get("TP_COOPERATIVE_REDUCTIONS", "0") == "1"
    )

    #: Choose among the tunings while the graph is being built, rather than at
    #: the first launch.  Left unset, the choice is made at the first launch.
    autotune_at_compile_time: bool | None = (
        None
        if "TP_AUTOTUNE_AT_COMPILE_TIME" not in os.environ
        else os.environ["TP_AUTOTUNE_AT_COMPILE_TIME"] == "1"
    )

    #: Fuse two reductions that run over different axes into one launch.  On by
    #: default: the fused launch reads each element once, which neither of the
    #: two does alone.
    mix_order_reduction = os.environ.get("TP_MIX_ORDER_REDUCTION", "1") == "1"

    #: Where a mix-order reduction is split, when it is split at a fixed size
    #: rather than at a size chosen by measurement.  Left unset, it is not.
    mix_order_reduction_split_size: int | None = None

    #: Whether a mix-order reduction may be measured with more than one stage.
    mix_order_reduction_autotune_split_size: bool = (
        os.environ.get("TP_MIX_ORDER_REDUCTION_AUTOTUNE_SPLIT_SIZE", "0") == "1"
    )

    #: How many blocks a kernel may be split into.  None means as many as the
    #: tile calls for; one is a single block, two is one dimension of tiling,
    #: and three is experimental.
    max_tiles: int | None = None

    #: Prefer a tile with more dimensions, which makes an indexing expression
    #: simpler to write and therefore easier to recognise as one that could be
    #: a block pointer.
    prefer_nd_tiling: bool = False

    #: Look at which loads of a kernel sit next to each other in memory before
    #: choosing a tile, rather than choosing a tile and then seeing what the
    #: loads cost.
    coalesce_tiling_analysis: bool = (
        os.environ.get("TP_COALESCE_TILING_ANALYSIS", "1") == "1"
    )

    #: Put a reduction inside the tile rather than beside it, which is what lets
    #: a tile carry a partial result across its own blocks.
    tile_reductions: bool = False

    #: Write a matrix product out as loops rather than calling a prepared one.
    native_matmul: bool = os.environ.get("TP_NATIVE_MATMUL", "0") == "1"

    #: Let a tile hold one reduction inside another.
    nested_reduction: bool = os.environ.get("TP_NESTED_REDUCTION", "0") == "1"

    #: End a fusion where a wider tile would let the two sides be tiled
    #: together, rather than fusing them and giving both a narrower tile.
    tiling_prevents_pointwise_fusion: bool = True
    tiling_prevents_reduction_fusion: bool = True

    #: Allow a mix-order reduction to use more than one stage, which shares
    #: memory between the stages and so can run out of it.
    mix_order_reduction_non_strict_mode = False

    #: How many separate reads a mix-order reduction may make.  Zero says not
    #: to check, which leaves the number of reads unbounded.
    mix_order_reduction_max_reads = 10


triton = _TritonConfig()


class _TraceConfig:
    """The switches that decide how much of a run is written down."""

    #: Which launch the region being traced covers: the wrapper that made it,
    #: or the launch itself.
    scope = "all"

    #: Which regions are worth writing down.  A region is identified by the
    #: part of the model it came from.
    regex = None

    #: Where the trace is written.  Left unset, nothing is written.
    output_dir = None

    #: Put the record of where each value came from on the trace's timeline,
    #: beside the launches, rather than in the graph's own listing.
    provenance_tracking_to_timeline = (
        os.environ.get("TP_COMPILE_DEBUG_EXTEND", "0") == "1"
    )


trace = _TraceConfig()


# ---------------------------------------------------------------------------
# How wide a read may be assumed to be, and how a build is spread over threads
# ---------------------------------------------------------------------------

#: The number of reads past which a value's loads are written out one by one
#: rather than left to be coalesced, when nothing overrides it.  A value read
#: this many times is read often enough that writing the reads out is worth the
#: code it costs.
_realize_acc_reads_threshold_default = 8

#: Where the reads of a value are written out one by one, overriding the
#: default above.  Left unset, the default is used.
realize_acc_reads_threshold: int | None = None

#: The same threshold for a value on the host, where a read is a plain load and
#: there is no coalescing to hand it to, so the bar for writing reads out is
#: higher.
realize_cpu_acc_reads_threshold = 12

#: Where the reads are also written out once the value is read this many
#: elements, however few times it is read.  Left unset, size is not considered.
realize_acc_reads_size_threshold: int | None = None

#: How many builds run at once.  Resolved from the environment first, then from
#: the machine: one thread means every build stays on the calling thread, which
#: is what keeps stepping through a build possible.
if "TP_COMPILE_THREADS" in os.environ:
    compile_threads = int(os.environ["TP_COMPILE_THREADS"])
elif sys.platform == "win32":
    # Starting a thread per build is unreliable here, so a build is run on the
    # calling thread instead.
    compile_threads = 1
else:
    compile_threads = max(1, (os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# What a fused program is allowed to assume
# ---------------------------------------------------------------------------

#: Assume an input's memory is aligned to the width of a wide load, and write
#: the generated code on that assumption, copying an input that is not aligned
#: rather than writing a narrower load for it.  Most inputs are aligned, so the
#: narrow load is the rarer case and the copy is the rarer cost.
assume_aligned_inputs: bool = False

#: Assume every index fits in 32 bits, which lets a generated kernel use
#: narrower arithmetic throughout.  A program with an index that does not fit
#: is then wrong rather than slow, so this is off unless a caller knows the
#: extents are small.
assume_32bit_indexing: bool = False

#: Take a fusion only where the two pieces are independent enough that the
#: result does not depend on the order they run in.
combo_kernels = False

#: Write into a buffer that is still being read, rather than into one of its
#: own.  On because a program that cannot do this allocates twice as much.
inplace_buffers = True

#: Write down which node of the graph each line of generated code came from.
comment_origin = False

#: Say what the compiler is doing while it does it.
debug = False

#: A conversion to a narrower type is written as the rounding it performs,
#: rather than as the conversion the hardware does.  The two disagree on values
#: the hardware cannot represent, and writing the rounding makes that visible.
emulate_precision_casts: bool = (
    os.environ.get("TP_EMULATE_PRECISION_CASTS", "0") == "1"
)

#: Ignore every cache and compile afresh, which is what a measurement needs:
#: a number read out of a cache is the number that was measured then.

#: Whether compiled kernels should travel inside a cache entry.  Off by
#: default: it makes an entry carry every kernel it needs, which is what a
#: cache read on another machine requires and what a cache only ever read
#: where it was written does not.  ``None`` means "decided by the bundler".
bundle_triton_into_fx_graph_cache: bool | None = None

#: Whether a compiled kernel can be launched from a binary already on disk,
#: without going through the runtime's own compile step.
use_static_triton_launcher: bool = False
force_disable_caches: bool = False

#: How many pairs of groups to try fusing before giving up on a round of
#: grouping, which bounds the work one round can take when there are many
#: groups.
max_fusion_buffer_group_pairwise_attempts = 64

#: Take a runtime estimate for a matrix multiply from a measurement of it,
#: rather than from a formula, which is slower and more accurate.
runtime_estimations_mms_benchmark: bool = False


@contextlib.contextmanager
def patch(*args, **kwargs):
    """Set settings for the length of a block, then put them back.

    A setting that has to differ for one piece of work and not for the rest is
    changed around that piece rather than around the whole run, so what the
    setting is outside the block is whatever the caller had it.  A name that
    does not exist is an error rather than a new setting: a setting nobody
    reads is a setting that was meant for something else.

    The settings are given as a mapping, as alternating names and values, or as
    keyword arguments naming the settings directly.  A name may be written the
    way it is reached, so a setting in a namespace is patched as
    ``"triton.proton_profiling"`` or as ``triton.proton_profiling=False``.
    """

    if args and isinstance(args[0], dict):
        if len(args) > 1 or kwargs:
            raise TypeError(
                "settings are given as a mapping, or as names and values, not both"
            )
        pairs = list(args[0].items())
    else:
        if len(args) % 2 != 0:
            raise TypeError(
                "settings given by name must come in name-and-value pairs"
            )
        pairs = list(zip(args[::2], args[1::2])) + list(kwargs.items())

    saved = []
    try:
        for name, value in pairs:
            parts = name.split(".")
            target = sys.modules[__name__]
            for part in parts[:-1]:
                target = getattr(target, part)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise AttributeError(f"there is no setting named {name!r}")
            saved.append((target, leaf, getattr(target, leaf)))
            setattr(target, leaf, value)
        yield
    finally:
        # Put back in the reverse order the settings were applied, so that two
        # patches of the same setting nest the way they were written.
        for target, leaf, value in reversed(saved):
            setattr(target, leaf, value)


# ---------------------------------------------------------------------------
# How a launch is tuned, and what a measurement may assume
# ---------------------------------------------------------------------------

#: Keep the tunings that were measured, so a second run of the same shape
#: tunes nothing.  On because measuring is the expensive part.
autotune_local_cache: bool = True

#: Keep the tunings where another process can read them.  Left unset, the
#: answer is whatever the runtime's own remote cache decides, which is why it
#: is unset rather than off.
autotune_remote_cache: bool | None = None

#: Keep the raw binary of a kernel that is started without going through the
#: runtime's own launcher.  A cold load can then rehydrate from those bytes
#: instead of compiling again -- at the cost of carrying them in every cache
#: entry, which is why it is off.
keep_static_cubin_raw: bool = (
    os.environ.get("TP_KEEP_STATIC_CUBIN_RAW", "0") == "1"
)

#: Keep the measured answers together, as one entry, rather than one entry per
#: kernel -- which on a model of any size is a great many small files.  Only
#: available when the local cache is on, because the gathered answers are fed
#: to it.  ``True`` turns it on, ``False`` turns it off, ``None`` leaves the
#: decision to whatever the remote cache itself says.
bundled_autotune_remote_cache: bool | None = None

#: Tune each launch the first time it is seen and keep the answer, rather than
#: tuning it again for every new shape.
incremental_autotune: bool | None = False

#: Spend the slow passes on the tuning, including for launches that are only
#: pointwise.
max_autotune_pointwise = os.environ.get("TP_MAX_AUTOTUNE_POINTWISE") == "1"

#: Make a sum give the same answer whatever order it was computed in, and
#: make the answer not depend on how the work was split across threads.  Off
#: because both cost arithmetic the fast path would not otherwise do.
batch_invariant = os.environ.get("TP_BATCH_INVARIANT") == "1"

#: Measure a tuning against every direction it could move, rather than
#: accepting the first that improves.  Off because it multiplies the number of
#: measurements.
coordinate_descent_check_all_directions = (
    os.environ.get("TP_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS") == "1"
)

#: How far one tuning step may move a block size.
coordinate_descent_search_radius = int(
    os.environ.get("TP_COORDINATE_DESCENT_RADIUS", "1")
)

#: Widen the reduction block as the number of blocks shrinks, so a launch with
#: few blocks still has enough work per block to be worth launching.
dynamic_scale_rblock = os.environ.get("TP_DYNAMIC_SCALE_RBLOCK", "1") == "1"

#: Write the reduction in the order that gives the same answer for a signed
#: zero whatever order it is computed in, which is slower than the order that
## does not.
strict_signed_zero = False

#: Let the arithmetic be reorganised freely, which lets a multiply be folded
#: into an addition and a division become a reciprocal.  Off because the answer
#: is then not the answer the arithmetic says.
use_fast_math = os.environ.get("TP_USE_FAST_MATH") == "1"

#: Write into the launch that a run asked for deterministic arithmetic, so
#: that the record says the run wanted it rather than the launch assuming it.
write_are_deterministic_algorithms_enabled = (
    os.environ.get("TP_WRITE_ARE_DETERMINISTIC_ALGORITHMS_ENABLED", "1") == "1"
)

#: Which launches the bandwidth figures are written for, as a pattern.  Empty
#: means every launch.
profile_bandwidth_regex = ""

#: Whether the tuning is walked by hand rather than searched.
coordinate_descent_tuning = False

#: Whether the tuning is benchmarked at all.
benchmark_kernel = False

#: Whether a launch of several kernels is benchmarked as a whole.
benchmark_combo_kernel = False


def is_fbcode() -> bool:
    """Whether this build is one where the internal defaults do not apply.

    The defaults are chosen for a general build.  A build with its own
    conventions answers false here, so that the defaults stand rather than
    being second-guessed.
    """

    return False


class _EagerNumerics:
    """What the arithmetic does at the edges of what a type can hold.

    A value too large for its type becomes an infinity and a value too small
    becomes a denormal, and the two are the arithmetic's business rather than
    the hardware's.  These say whether they are written as the arithmetic
    would do them or as the hardware happens to.
    """

    #: A denormal is written as a zero instead of being kept.
    disable_ftz = False

    #: Take a float64 operation from the device library rather than from
    #: arithmetic.  Off because the arithmetic is faster where it exists, and
    #: the two are not always the same answer.
    use_project_libdevice = False


eager_numerics = _EagerNumerics()


#: Fall back to running a draw or a dropout eagerly.  Slow, and useful when a
#: fused result is suspected of being wrong and the eager one is known good.
fallback_random = False

#: Run a draw or a dropout the way the framework does when it is not being
#: compiled, while still letting the result be fused.  Faster, and differs
#: from the eager result in how a value is arranged within a thread.
align_random_eager = False

#: Which of the joins and splits to apply, and with what settings, before the
#: backward graph is built.  Empty applies none of them; the keys are the
#: names the rules are registered under.
pre_grad_fusion_options: dict[str, dict[str, Any]] = {}

#: Which of the batches of work to apply, and with what settings, after the
#: backward graph is built.  Empty applies none of them; the keys are the
#: names the rules are registered under.
post_grad_fusion_options: dict[str, dict[str, Any]] = {}


class _CudaConfig:
    """Which architecture and toolkit a kernel is compiled for.

    A program can be run on a machine other than the one it is built on, and a
    kernel written for one architecture will not run on another. So the
    architecture is something that can be said rather than something read off
    whatever device happens to be present, and saying nothing means the device
    decides.
    """

    #: The architecture to compile for, as the two digits the device reports
    #: them: "80" for one generation, "90" for the next.  Nothing means the
    #: device that will run it decides.
    arch: str | None = None
    #: The toolkit to compile against, as its version: "12.1" and so on.
    #: Nothing means the toolkit that is installed decides.
    version: str | None = None
    #: Where the compiler driver is, when it is not on the path.
    cuda_cxx: str | None = os.environ.get("TP_CUDA_CXX")


cuda = _CudaConfig()


#: Where a worker's output is written.  Empty sends it to the worker's own
#: stdout, which the parent then has to read and disentangle from its own; a
#: path sends each worker's output to its own file, which is what makes the
#: output of a failing worker readable at all.
worker_logpath: str = os.environ.get("TP_WORKER_LOGPATH", "")

#: Whether a worker's own output is dropped unless something went wrong.
#:
#: A worker that is told to be quiet still says so when it fails; what it
#: stops doing is narrating a successful compile, which is the parent's job
#: to summarise and which a pool of workers otherwise repeats once each.
worker_suppress_logging: bool = os.environ.get("TP_WORKER_SUPPRESS_LOGGING", "1") == "1"


class _RocmConfig:
    """Settings that only mean anything on the other vendor's hardware.

    Kept apart from the rest because a setting here is read on hardware that
    most machines are not: a program that reads one of these on the wrong device
    is reading a number that was never about it, so the values are gathered here
    where it is visible that they only apply there.
    """

    #: How much wider than it is tall the contraction has to be before making
    #: the right operand contiguous pays for the copy.  Nothing rather than one:
    #: no threshold is a threshold nothing passes, and a program on this hardware
    #: should not be rewriting its operands because of a number meant for another.
    contiguous_threshold: int | None = None


rocm = _RocmConfig()


#: How often a worker reports a job that is still running after it started.
#:
#: A parent waiting on a worker reads nothing while it waits, so a worker that
#: has stopped making progress looks the same as one that is being slow.  The
#: worker therefore says so on this interval for as long as a job is still
#: going, which leaves a record of which job was stuck rather than only that
#: something was.  Zero turns the reporting off.
compile_worker_watchdog_interval_seconds: int = int(
    os.environ.get("TP_COMPILE_WORKER_WATCHDOG_INTERVAL", 60)
)

#: How long a pool waits with nothing to do before telling its workers to stop
#: waiting for work.  A worker blocked reading its pipe forever holds memory
#: for a parent that has gone; this is how long it takes to notice.
quiesce_async_compile_time: int = 60


def get_worker_log_path() -> str | None:
    """Where a worker's output goes when no path was configured.

    Answered with nothing outside an internal build, so a worker's output
    goes to the worker's own stdout and the parent has to disentangle it from
    its own.  An internal build knows which job and which rank it is, and
    writes to the one place a job's workers are expected to write.
    """
    if not is_fbcode():
        return None
    job_name = os.environ.get("MAST_HPC_JOB_NAME")
    if job_name is None:
        return None
    return f"/logs/dedicated_log_compile_worker_rank{os.environ.get('ROLE_RANK', '0')}"


def decide_worker_start_method() -> str:
    """How a worker process is started.

    A worker generates code in a fresh interpreter, and there are three ways
    to get one.  Forks are cheapest and inherit a process that has already
    built things, which is both faster and the reason they are unsafe here.
    Spawning starts clean and pays for it.  A separate subprocess sits
    between them: a new interpreter, without paying to re-import.

    A name outside those three is refused rather than fallen back from: a
    start method that does not exist is a configuration that was meant to
    say something and does not.
    """
    start_method = os.environ.get("TP_WORKER_START", "subprocess")
    if start_method not in ("subprocess", "fork", "spawn"):
        raise AssertionError(f"Invalid start method: {start_method}")
    return start_method


#: Which of those three a worker process is started by.
worker_start_method: str = decide_worker_start_method()

#: Whether the pool of workers is told to go quiet at the end of each
#: compilation, rather than staying warm for the next one.
#:
#: A warm pool answers the next compilation faster; a quiet one holds no
#: memory between them.  Which is worth more depends on how often something
#: is compiled, so this is a choice rather than a fact.
quiesce_async_compile_pool: bool = (
    os.environ.get("TP_QUIESCE_ASYNC_COMPILE_POOL", "1") == "1"
)
