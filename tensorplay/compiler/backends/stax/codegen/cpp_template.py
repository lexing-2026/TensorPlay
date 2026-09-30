"""A template of the C++ route, as distinct from a schedule of it.

The scheduled route builds a kernel out of a nest it decides itself.  This one
builds it out of a program handed to it -- a template -- and what this class
holds is everything except the part that names the buffers, which is the part
the template produces by running.

So the same kernel gets built twice.  Once here, to find out what it will
contain: the shapes, the distances, the sizes that have to become names because
the caller is handed numbers and the kernel is not, and the source of the
benchmark that will measure it.  And once more at the end, when the buffers are
known and the kernel can be written against them.  The second time is not a
repeat of the first -- it is the only time the argument list exists.

Which is why the arguments are checked here rather than assumed.  A buffer named
in one place and a different buffer named in another would compile, run, and
read memory that was never written, and nothing about the failure would say so.
So the order the caller is handed is compared against the order that was asked
for, and a difference is said out loud.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import logging
import sys
from typing import Any, Callable, Iterable
from unittest.mock import patch

import sympy

from .. import config, ir
from ..autotune_process import CppBenchmarkRequest, TensorMeta
from ..loops import V
from ..utils import IndentedBuffer, Placeholder, unique
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel

log = logging.getLogger(__name__)


class CppTemplate(KernelTemplate):
    """One way of writing a kernel, as a program that writes it.

    Asking this for a kernel gives back a choice, not a kernel.  The choice knows
    what the kernel will read and write, and it can produce the kernel once the
    buffers are named -- which is why the name and the factory travel together
    rather than a finished thing that could not be finished.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list,
        layout: Any,
        num_threads: int,
        epilogue_creator: Callable | None = None,
    ) -> None:
        super().__init__(name)
        self.input_nodes = input_nodes
        self.index = next(self.index_counter)
        self.output_node: Any = ir.Buffer(
            name=f"buf_out{self.index}", layout=layout
        )
        self.layout = layout
        self.num_threads = num_threads
        self.epilogue_creator = epilogue_creator

    def generate(self, **kwargs: Any) -> CppTemplateCaller:
        """Run the body once to find out what the kernel will be, and say so.

        Everything the choice needs is worked out here: the source, the argument
        list, and a request that can measure it.  The kernel itself is not
        written -- it is written again later, against buffers whose names are
        then known -- so what comes back is a factory rather than a kernel.
        """

        kernel_name = f"cpp_{self.name}"
        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)),
            patch.object(ir.FlexibleLayout, "allow_indexing", True),
            V.graph.set_current_device(self.layout.device),
            CppTemplateKernel(
                kernel_name=kernel_name, num_threads=self.num_threads
            ) as kernel,
        ):
            code = kernel.render(self, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        # The order the caller will hand arguments in has to be the order the
        # body asked for them in.  A body that read a buffer after writing
        # another would produce the same code either way, and would be handed
        # the wrong one.
        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        if isinstance(self.output_node, Iterable):
            expected_args.extend([node.get_name() for node in self.output_node])
        else:
            expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )
        # extra_args are only used for benchmarking, not compiled kernel correctness
        extra_args = V.graph.sizevars.optimization_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # Cast the size hint from int to ctypes.c_ulonglong explicitly
        # since in cpp kernel, we bind it to C long
        extra_args = tuple(ctypes.c_ulonglong(x) for x in extra_args)

        kernel_hash_name = f"cpp_{self.name}_{self.index}"

        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: Any,
            flag_template_buffer_has_other_users: bool,
            epilogue_nodes: list | None = None,
        ) -> Any:
            """Write the kernel against buffers whose names are now known.

            Not the same call as the one above: that one worked out what the
            kernel would be, and this one is it.  The difference is the buffer --
            there the output was a placeholder to measure against, and here it is
            the one the caller will read.
            """

            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME), num_threads=self.num_threads
            )
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node[0].get_layout()
            if isinstance(self.output_node, Iterable)
            else self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def header(self) -> IndentedBuffer:
        """What the emitted source has to be able to name.

        Only when profiling was asked for: the rest is in the prefix every
        emitted source already carries, and repeating it would be a second place
        where a change to the prefix had to be made.
        """

        res = IndentedBuffer()
        res.writeline('#include "tensorplay/GeneratedCode.h"')
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        return res

    def render(self, **kwargs: Any) -> str:
        raise NotImplementedError
