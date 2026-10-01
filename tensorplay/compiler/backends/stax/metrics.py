"""Counters describing what the code generators actually did.

A generator that changes a layout to suit the machine is making a decision
that is invisible in the output: the buffer it emits has the same extents and a
different stride, and nothing in the emitted code says so.  A counter is the
only place that decision shows up, which is what makes it possible to tell a
run that padded from a run that did not.
"""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache

from ....graph.experimental.sympy_functions import OrderedSet
from . import config
from .utils import get_benchmark_name

#: How many layouts had their strides padded to the transaction width.
num_comprehensive_padding = 0

#: How many kernels were emitted, across every backend.
generated_kernel_count = 0

#: How many of those were emitted as the vector form of a C++ piece, and how
#: many were the scalar form instead.  A run where nearly everything took the
#: vector form is a run where the shapes lined up with the machine's width.
generated_cpp_vec_kernel_count = 0

#: How many values were read out of a buffer, and how many were written, not
#: counting the reads that a value already in hand made unnecessary.  A kernel
#: that reads far more than it writes is doing its arithmetic on values it
#: could have kept.
num_load = 0
num_store = 0

#: How many times a write into a constant tensor was turned into a masked
#: write over the whole of it.  Worth counting because it replaces a cheap
#: write with an arithmetic one, which is a trade that only pays when the
#: count of the writes it removes is large.
num_matches_for_scatter_upon_const_tensor = 0

#: How many times a kernel was emitted more than once for the same call.
num_kernel_reuse = 0

#: How many values were combined into one, across every kernel.
num_reduction = 0

@dataclasses.dataclass
class CppOuterLoopFusedCount:
    """One outer-loop join: how many launches it held, and how many local buffers.

    A join of outer loops is only worth having if it kept sharing work, so the
    count of launches inside says how much was actually merged, and the count
    of local buffers says how much of the merged traffic stayed off the
    outermost memory.
    """

    inner_kernel_number: int
    local_buffer_number: int = 0


#: One entry per outer-loop join made, across every backend.
cpp_outer_loop_fused_inner_counts: list[CppOuterLoopFusedCount] = []

#: How many times a narrow floating point value had to be widened before being
#: computed on and narrowed again afterwards.  A run where this is large is a
#: run spending its time on conversions rather than on arithmetic.
cpp_to_dtype_count = 0

#: How many reductions were written so that each thread accumulates into its own.
#: A reduction that is not written this way is one where two threads may write
#: the same accumulator, so the count is what says whether that ever happened.
parallel_reduction_count = 0


#: The tables that have been registered, by the name each is asked for.
#: Recorded because a caller that asks for a table by a name that was never
#: registered has to be told so rather than handed an empty one.
REGISTERED_METRIC_TABLES: dict = {}

#: How many operations the region held before any of them were fused together,
#: so that the number that came out can be said against what went in.
ir_nodes_pre_fusion = 0

#: How many times a loop's order was changed to suit the machine, and how many
#: times a fusion of a reduction in a different order was refused.
num_loop_reordering = 0
rejected_mix_order_reduction_fusion: int = 0

#: How many reductions were written nested inside another reduction, and how
#: many were written in an order the operations were not given in.
codegen_nested_reduction: int = 0
codegen_mix_order_reduction: int = 0


def reset() -> None:
    """Zero every counter, so one run's numbers are not read as another's."""

    global num_comprehensive_padding, generated_kernel_count
    global generated_cpp_vec_kernel_count, cpp_outer_loop_fused_inner_counts
    global cpp_to_dtype_count
    global num_kernel_reuse, num_load, num_store, num_reduction

    num_comprehensive_padding = 0
    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    cpp_outer_loop_fused_inner_counts.clear()
    cpp_to_dtype_count = 0
    num_kernel_reuse = 0
    num_load = 0
    num_store = 0
    num_reduction = 0
    parallel_reduction_count = 0

def get_metric_fields() -> list[str]:
    return [field.name for field in dataclasses.fields(CachedMetricsDeltas)]


@dataclasses.dataclass
class MetricTable:
    table_name: str
    column_names: list[str]

    num_rows_added: int = 0

    def add_row(self, row_fn: Callable[[], dict[str, str | float | None]]) -> None:
        if self.table_name not in enabled_metric_tables():
            return

        row_dict = row_fn()
        if len(self.column_names) != len(row_dict):
            raise AssertionError(f"{len(self.column_names)} v.s. {len(row_dict)}")
        if OrderedSet(self.column_names) != OrderedSet(row_dict.keys()):
            raise AssertionError(
                f"{OrderedSet(self.column_names)} v.s. {OrderedSet(row_dict.keys())}"
            )

        bn = get_benchmark_name()
        # assert bn is not None
        row = [bn] + [row_dict[column_name] for column_name in self.column_names]
        if not all(isinstance(i, (str, float, type(None))) for i in row):
            raise AssertionError("expected all row values to be str, float, or None")
        self._write_row(row)

    def output_filename(self) -> str:
        return f"metric_table_{self.table_name}.csv"

    def write_header(self) -> None:
        filename = self.output_filename()
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(["model_name"] + self.column_names)

    def _write_row(self, row: list[str | float | None]) -> None:
        filename = self.output_filename()
        if self.num_rows_added == 0 and not os.path.exists(filename):
            self.write_header()

        self.num_rows_added += 1

        for idx, orig_val in enumerate(row):
            if isinstance(orig_val, float):
                new_val = f"{orig_val:.6f}"
            elif orig_val is None:
                new_val = ""
            else:
                new_val = orig_val
            row[idx] = new_val

        with open(filename, "a") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(row)

    @staticmethod
    def register_table(name: str, column_names: list[str]) -> None:
        table = MetricTable(name, column_names)
        REGISTERED_METRIC_TABLES[name] = table


def log_kernel_metadata(
    kernel_name: str, kernel_path: str, kernel_module_code: str
) -> None:
    """
    A utility to log kernel metadata. We may parse metadata from kernel source code here.

    It's fine to parse the generated kernel code here since the logging is
    disabled by default. It would hurt compilation time.
    """
    from .runtime.benchmark_report import get_kernel_category_by_source_code

    kernel_category = get_kernel_category_by_source_code(kernel_module_code)
    reduction_hint = _parse_reduction_hint(kernel_category, kernel_module_code)
    size_hints = _parse_size_hints(kernel_module_code, kernel_category)
    kernel_fn_code = _parse_kernel_fn_code(kernel_module_code)

    proper_kernel_fn_code = _parse_proper_kernel_fn_code(kernel_fn_code)

    # the line of code excluding the decortors
    kernel_line_of_code = _parse_kernel_line_of_code(proper_kernel_fn_code)

    get_metric_table("kernel_metadata").add_row(
        lambda: {
            "kernel_name": kernel_name,
            "kernel_path": kernel_path,
            "kernel_category": kernel_category,
            "size_hints": size_hints,
            "reduction_hint": reduction_hint,
            "line_of_code": kernel_line_of_code,
            "num_load": _count_pattern(proper_kernel_fn_code, "tl.load"),
            "num_store": _count_pattern(proper_kernel_fn_code, "tl.store"),
            "num_for_loop": _count_pattern(proper_kernel_fn_code, "for "),
            "num_atomic_add": _count_pattern(proper_kernel_fn_code, "tl.atomic_add"),
            "num_args": _count_args(proper_kernel_fn_code),
            "xnumel": _parse_numel(proper_kernel_fn_code, "xnumel"),
            "ynumel": _parse_numel(proper_kernel_fn_code, "ynumel"),
            "rnumel": _parse_numel(proper_kernel_fn_code, "rnumel"),
            "kernel_args_num_gb": _parse_kernel_args_num_gb(
                kernel_fn_code, kernel_category
            ),
        }
    )


def enabled_metric_tables() -> OrderedSet[str]:
    return enabled_metric_tables_impl(config.enabled_metric_tables)


def enabled_metric_tables_impl(config_str: str) -> OrderedSet[str]:
    enabled: OrderedSet[str] = OrderedSet()
    for name in config_str.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in REGISTERED_METRIC_TABLES:
            raise AssertionError(f"Metric table name {name} is not registered")
        enabled.add(name)
    return enabled


def is_metric_table_enabled(name: str) -> bool:
    return name in enabled_metric_tables()


def get_metric_table(name: str) -> MetricTable:
    if name not in REGISTERED_METRIC_TABLES:
        raise AssertionError(f"Metric table {name} is not defined")
    return REGISTERED_METRIC_TABLES[name]


def log_kernel_autotune_result(
    kernel_path: str, kernel_name: str, config: Config, latency: float
) -> None:
    get_metric_table("kernel_autotune").add_row(
        lambda: {
            "kernel_path": kernel_path,
            "kernel_name": kernel_name,
            "triton_config": str(config),
            "latency_ms": latency,
        }
    )


MetricTable.register_table(
    "slow_fusion",
    [
        "kernel1_path",
        "kernel1_latency",
        "kernel2_path",
        "kernel2_latency",
        "fused_kernel_path",
        "fused_kernel_latency",
        "slow_down_ratio",
    ],
)

MetricTable.register_table(
    "graph_stats",
    [
        "graph_id",
        "num_nodes_before_fusion",
        "num_nodes_after_fusion",
    ],
)

MetricTable.register_table(
    "persistent_red_perf",
    [
        "kernel0_path",
        "kernel1_path",
        "kernel2_path",
        "kernel3_path",
        "kernel0_latency",
        "kernel1_latency",
        "kernel2_latency",
        "kernel3_latency",
        "size_hints",
        "reduction_hint",
    ],
)

MetricTable.register_table(
    "fusion_failure_due_to_indexing_mismatch",
    [
        "pre_grad_graph_id",
        "post_grad_graph_id",
        "node1_name",
        "node2_name",
        "node1_debug_str",
        "node2_debug_str",
        "common_buffer_names",
        "failure_reason",
    ],
)

MetricTable.register_table(
    "kernel_metadata",
    [
        "kernel_name",
        "kernel_path",
        "kernel_category",  # pointwise/reduction/foreach etc.
        "size_hints",
        "reduction_hint",
        "line_of_code",
        "num_load",
        "num_store",
        "num_for_loop",
        "num_atomic_add",
        "num_args",
        # xyz numel can be different to size_hints since size_hints are rounded
        # up to the nearest power of 2.
        # The compiler kernel will burn in the xyz numel in kernel code for static
        # shape kernels.
        # Logging them will be helpful to find unaligned shape for reduction
        "xnumel",
        "ynumel",
        "rnumel",
        "kernel_args_num_gb",
    ],
)

MetricTable.register_table(
    "kernel_autotune",
    [
        "kernel_path",
        "kernel_name",
        "triton_config",
        "latency_ms",
    ],
)
