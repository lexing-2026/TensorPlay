"""The base every kernel template shares.

Four things and no more: an identity that survives a restart, a digest of the source
it emits, a refusal that comes back as a refusal rather than an exception, and
generation.  Everything a particular caller needs belongs to that caller's layer.
"""

from __future__ import annotations

from typing import Any

from .ir import ChoiceCaller

class KernelTemplate:
    """One operation, implemented by kernels chosen from a configuration space.

    A subclass says what to do with one configuration in ``generate``, and
    everything else here is the discipline around it: a refusal is caught and
    reported rather than propagated, an identity is stable across processes so
    a stored decision can be recognised, and a source digest lets a stored
    decision be thrown away when the kernel it named has changed.
    """

    def __init__(self, name: str, hash: str | None = None):
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        """A stable identity, so a stored decision can be found again.

        Every template is unique in the system, and the identity has to
        survive a restart, so nothing about it may depend on where it was
        defined.
        """

        return self.name

    @property
    def src_hash(self) -> str | None:
        """A digest of the source this template emits, when it has one.

        A template that emits a kernel can say what the kernel was, so a
        stored decision is not reused once the kernel it names has moved.
        """

        return self._hash

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The choice for one configuration, or ``None`` when it does not fit."""

        collected: list[ChoiceCaller] = []
        error = self.maybe_append_choice(collected, **kwargs)
        if error is None and len(collected) == 1:
            return collected[0]
        return None

    def maybe_append_choice(
        self, choices: list[ChoiceCaller], **kwargs: Any
    ) -> NotImplementedError | None:
        """Add the choice for one configuration, or report why there is none.

        Refusing is a normal answer and comes back as the refusal rather than
        as an exception, so a caller can offer a whole table and keep the
        ones that apply.
        """

        try:
            choices.append(self.generate(**kwargs))
        except NotImplementedError as error:
            return error
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """The choice one configuration describes."""

        raise NotImplementedError
