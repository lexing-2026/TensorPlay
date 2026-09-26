"""The operation itself, as a choice that can hold its own against a kernel.

The thing every template is measured against is not a fallback taken when measurement
fails; it is one of the things being measured, and it wins whenever no kernel beats
it.  So it is registered here, beside the templates, and answers the same questions
they do.
"""

from __future__ import annotations

from typing import Any, Callable

from .ir import ChoiceCaller, ExternChoiceCaller

class ExternKernelChoice:
    """An operation that can hold its own against a kernel, as a choice.

    The operation every template is measured against is not a fallback taken
    when measurement fails -- it is one of the things being measured, and it
    wins whenever no kernel beats it.  So it is registered here, beside the
    templates, under a name codegen can refer to, and it answers the same
    questions they do; a caller can hand a list of either to the same
    enumeration without caring which is which.

    Each instance registers once under its name.  Registering the same
    callable twice is tolerated, because a module can be initialised twice in
    one process; registering a *different* callable under a name already taken
    is refused, because that is a collision rather than a re-registration.
    """

    _registry: dict[str, "ExternKernelChoice"] = {}

    def __init__(
        self,
        kernel: Callable[..., Any],
        name: str | None = None,
        *,
        has_out_variant: bool = True,
        op_overload: Any = None,
        use_fallback_kernel: bool = False,
        kernel_creator: Callable[..., Any] | None = None,
    ):
        name = name or getattr(kernel, "__name__", None) or "extern"
        if kernel is not None and not callable(kernel):
            raise AssertionError("an extern choice must wrap something callable")
        # ``None`` means the kernel is the framework's own and is resolved by
        # whatever launches it, which is the case for an operation the compiler
        # calls by name rather than by function.  The name is still the identity
        # that matters here, so the collision check below stands either way.
        existing = ExternKernelChoice._registry.get(name)
        if existing is not None and existing.kernel is not kernel:
            raise AssertionError(f"duplicate extern choice: {name}")
        self.name = name
        self.kernel = kernel
        self.has_out_variant = has_out_variant
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator
        ExternKernelChoice._registry[name] = self

    # -- the part that makes it usable wherever a template is ------------
    @property
    def uid(self) -> str:
        return self.name

    @property
    def src_hash(self) -> str | None:
        return None

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The operation itself, as the choice it always is."""

        return ExternChoiceCaller(
            name=self.name,
            layout=kwargs.get("layout"),
            description="the operation itself",
            launcher=self.kernel,
        )

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        choices.append(self.choice_or_none(**kwargs))
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        return self.choice_or_none(**kwargs)

    @classmethod
    def lookup(cls, name: str) -> "ExternKernelChoice | None":
        return cls._registry.get(name)

    def __repr__(self) -> str:
        return f"ExternKernelChoice({self.name})"
