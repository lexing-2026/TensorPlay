"""One template, one configuration, and the choice they make -- eventually.

A call is described long before anything is compiled, so the pairing is deferred.
The deferral is the point: a configuration that turns out not to fit is remembered as
not fitting rather than rebuilt on every visit.
"""

from __future__ import annotations

from typing import Any, Iterator

from .common import KernelTemplate
from .ir import ChoiceCaller, KernelInputs, Layout
from .params import DictKernelTemplateParams, KernelTemplateParams

class KernelTemplateChoice:
    """One template, one configuration, and the choice they make -- eventually.

    The kernel is built the first time the choice is asked for and then kept,
    including when the build turns out not to apply: a configuration that does
    not fit this call is an answer, not a retry.
    """

    def __init__(
        self,
        template: KernelTemplate,
        params: KernelTemplateParams,
        extra_kwargs: dict[str, Any],
        layout: Layout | None,
        inputs: KernelInputs,
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = dict(extra_kwargs)
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}

    @property
    def choice(self) -> ChoiceCaller | None:
        """The built choice, or ``None`` when this configuration does not fit."""

        if not hasattr(self, "_choice"):
            kwargs = self.params.to_kwargs()
            try:
                self._choice = self.template.choice_or_none(
                    **kwargs, **self.extra_kwargs
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice

    @property
    def key(self) -> tuple:
        """What identifies this choice across processes."""

        return (
            self.template.uid,
            repr(self.params.to_serializeable_dict()),
            repr(self.layout),
            str(self.inputs.device),
        )

    def resolve(self, plain_launch):
        """The launcher for this configuration, or ``None`` when it does not fit."""

        if not hasattr(self, "_resolved"):
            self._resolved = True
            try:
                self._choice = self.template.generate_for(
                    self.params,
                    (self.layout,) if self.layout else (),
                    self.inputs.extra,
                    plain_launch,
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None and self._choice is not plain_launch:
                self._choice.bind(lambda *a, _c=self._choice, **k: _c.to_callable()(*a, **k))
        return self._choice

    def __repr__(self) -> str:
        return f"KernelTemplateChoice({self.template.name}, {self.params!r})"
def make_ktc_generator(
    template: KernelTemplate,
    cs: Iterator[KernelTemplateParams],
    extra_kwargs: dict[str, Any],
    overrides: dict[str, Any],
    layout: Layout | None,
    inputs: KernelInputs,
) -> Iterator[KernelTemplateChoice]:
    """One deferred choice per configuration, with the overrides folded in.

    The overrides are what a caller imposes on every configuration -- the
    parts of the call that are not a choice -- and folding them into the
    parameters here means a template reads one dict.
    """

    for params in cs:
        merged = {**params.to_kwargs(), **overrides}
        yield KernelTemplateChoice(
            template=template,
            params=DictKernelTemplateParams(merged),
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
