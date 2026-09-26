"""The layer that reaches a loop region.

A loop runtime describes a call before anything is compiled, so this is where the
result specification, the probe and the selection live.  They are not on the base
because the only caller that needs them is this one.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

from .choice import KernelTemplateChoice, make_ktc_generator
from .common import KernelTemplate
from .heuristics import TemplateConfigHeuristics
from .ir import KernelInputs
from .params import KernelTemplateParams

class LoopTemplate(KernelTemplate):
    """A template reached through a loop region.

    The loop runtime describes a call before anything is compiled, so this
    layer adds the two things that description needs: the result the call
    produces, which the template knows and its caller would otherwise have to
    infer, and the probe its candidates are measured on, which is built from
    the same description rather than from the region's real tensors.
    """

    def __init__(self, name: str, hash: str | None = None):
        super().__init__(name, hash)
        self.heuristics = TemplateConfigHeuristics()

    # identity ------------------------------------------------------------
    def emitter(self) -> str:
        """The source a configuration is turned into, for change detection."""

        raise NotImplementedError

    @property
    def uid(self) -> str:
        digest = hashlib.sha1(self.emitter().encode()).hexdigest()[:16]
        return f"{self.name}:{digest}"

    # results -------------------------------------------------------------
    def out_specs(self, meta: dict) -> tuple:
        """The layouts this call produces."""

        raise NotImplementedError

    # configurations ------------------------------------------------------
    #: Which inputs class describes this template's operands.
    inputs_class: type = KernelInputs

    def inputs_for(self, meta: dict) -> KernelInputs:
        sizes = tuple(meta.get("operand_sizes") or ())
        return self.inputs_class(
            shapes=sizes,
            dtypes=tuple([meta.get("operand_dtype")] * len(sizes)),
            device=meta.get("device"),
            operands=tuple(meta.get("operand_specs") or ()),
            feed=tuple(meta.get("feed") or ()),
            probe_feed=tuple(meta.get("probe_feed") or ()),
            extra=meta,
        )

    def configurations(self, out_specs: tuple, meta: dict) -> Iterator[KernelTemplateParams]:
        """The configurations worth considering for this call."""

        inputs = self.heuristics.adjust_kernel_inputs(self.inputs_for(meta), self.name)
        return self.heuristics.get_template_configs(inputs, self.name)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit.

        ``plain_launch`` is what the region would run with no template at all.
        It is handed in rather than reached for because for one configuration
        it *is* the answer: the operation is a candidate like any other, and
        the only way it can be is by being given the thing it replaces.
        """

        raise NotImplementedError

    # enumeration ---------------------------------------------------------
    def choices(self, meta: dict) -> list:
        """Every configuration that applies to this call."""

        return self.collect(meta, lambda choice, plain: choice.resolve(plain))

    def collect(self, meta: dict, build) -> list:
        """Build the applicable choices, leaving out the ones that do not fit.

        The configurations come from the heuristic, the overrides are what
        this call imposes on all of them, and the pairing is the one the
        deferred-choice generator owns -- so there is a single way for a
        configuration to become a choice, whichever template asked for it.
        """

        specs = self.out_specs(meta)
        inputs = self.heuristics.adjust_kernel_inputs(self.inputs_for(meta), self.name)
        overrides = dict(self.heuristics.get_extra_kwargs(inputs, self.name))
        if specs:
            overrides.setdefault("out_size", tuple(specs[0].size))
            overrides.setdefault("out_dtype", specs[0].dtype)
        out = []
        for choice in make_ktc_generator(
            self,
            self.configurations(specs, meta),
            {},
            overrides,
            specs[0] if specs else None,
            inputs,
        ):
            self.maybe_append_choice(out, choice, build)
        return out

    def maybe_append_choice(self, choices: list, choice, build) -> Any:
        try:
            if build(choice, None) is None:
                raise NotImplementedError(
                    f"{self.name} configuration {choice.params!r} does not fit"
                )
        except NotImplementedError as error:
            return error
        choices.append(choice)
        return None

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates."""

        return None

    def select(self, meta: dict, build) -> tuple:
        """Choose among this call's configurations and return the winner.

        The choice is made while the region is compiled, so the kernel a
        region runs is settled before it ever runs.  The operator itself is
        always among the candidates, which is what makes measuring safe: the
        worst a measurement can conclude is that the operator was already the
        best of them.
        """

        choices = self.collect(meta, build)
        if not choices:
            return None, None
        if len(choices) == 1 or not meta.get("bench", True):
            return build(choices[0], None), choices[0].params
        from ..runtime.stax_autotune import bench_candidates

        built = {}

        def materialise(candidate, plain):
            launcher = build(candidate, plain)
            built[candidate] = launcher
            return launcher

        feed = meta.get("probe_feed")
        if feed is None:
            return build(choices[0], None), choices[0].params
        best, _launch, _time = bench_candidates(
            materialise, choices, feed, rounds=meta.get("rounds", 2)
        )
        if best is None:
            best = choices[0]
        return built.get(best) or build(best, None), best.params
