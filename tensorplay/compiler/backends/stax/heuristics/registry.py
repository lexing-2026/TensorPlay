"""One place every rule about which configurations are worth trying is registered.

A rule is asked two questions that do not have the same answer: which
configurations of a template are worth measuring, and what a generated kernel
should be configured with.  The two are kept in one table because they are asked
the same way -- by name, by device, and sometimes by the operation -- and because
a program that has one rule registered but not the other is a program whose
configuration depends on which file the rule was written in.

Registration is by decorator rather than by a call at the point of definition so
that a rule lives next to the numbers it is a rule about: a table of tile sizes
and the rule that reads the table belong together, and separating them is how a
table and the rule that reads it come to disagree.

A lookup cascades.  A rule registered for a template on a device is asked first;
failing that, one registered for that template on any device; then for that
device whatever operation; then the one that applies everywhere.  So a rule that
is not device-specific is written once, and a rule that is specific overrides it
without having to restate what it did not mean to change.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

if True:  # typing-only import kept out of the runtime path
    from collections.abc import Iterator

    from .template.base import TemplateConfigHeuristics

log = logging.getLogger(__name__)

#: Every rule that has been registered, by what it was registered for.  The three
#: parts of the key are the template or name, the device, and the operation, any
#: of which may be absent to mean "whatever is in the others".
_HEURISTIC_REGISTRY: dict[tuple[str | None, ...], Any] = {}

#: The same table under the name it had when only templates were registered.  An
#: alias rather than a copy, so that clearing one clears the other.
_TEMPLATE_HEURISTIC_REGISTRY = _HEURISTIC_REGISTRY

#: Rules that have already been made, by what they were made for.  A rule is asked
#: once per configuration table per call, and making it twice would make two
#: objects that ought to agree about the same numbers.
_HEURISTIC_CACHE: dict[tuple[str | None, ...], Any] = {}


def _register(key: tuple[str | None, ...], target: Any) -> None:
    _HEURISTIC_REGISTRY[key] = target


def _lookup(name: str, device_type: str, op_name: str | None) -> Any | None:
    """The rule for this name on this device for this operation, or nothing.

    Asked with the operation first, then the device, then neither, so that the
    most specific rule that was registered is the one that answers: a rule
    written for one operation beats one written for the template, and a rule
    written for one device beats one written for all of them.
    """

    keys = [
        (name, device_type, op_name),
        (name, None, op_name),
        (name, device_type, None),
        (name, None, None),
    ]
    for key in keys:
        if key in _HEURISTIC_REGISTRY:
            return _HEURISTIC_REGISTRY[key]
    return None


# -- templates --------------------------------------------------------------


def register_template_heuristic(
    template_name: str,
    device_type: str | None,
    register: bool = True,
    op_name: str | None = None,
) -> Any:
    """Register a rule about which configurations of a template are worth trying.

    ``device_type`` of ``None`` means the rule applies wherever nothing more
    specific does.  ``op_name`` of ``None`` means the same for the operation.
    ``register`` says whether to register at all, which is how a rule that only
    applies on some hardware is written in one place and registered in another:
    the class exists either way, so a subclass elsewhere can still be declared.
    """

    def decorator(cls):
        if register:
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            _register(key, cls)
            log.info(
                "Registered template heuristic: %s for template_name=%s, "
                "device_type=%s, op_name=%s",
                cls.__name__, template_name, device_type, op_name,
            )
        return cls

    return decorator


def get_template_heuristic(
    template_name: str, device_type: str, op_name: str
) -> "TemplateConfigHeuristics":
    """The rule for this template on this device for this operation.

    Made once and then remembered, because a rule holds the numbers it was built
    with and building it twice would hold two copies that could come to disagree
    about a table.  Where nothing was registered, the rule that applies
    everywhere is used and said out loud, since a template quietly getting no
    rule at all is a template being measured against nothing.
    """

    cache_key: tuple[str | None, ...] = (template_name, device_type, op_name)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(template_name, device_type, op_name)

    if heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        log.error(
            "No template heuristic found - template_name=%s, device_type=%s, "
            "op_name=%s. Available: %s. Using fallback.",
            template_name, device_type, op_name, list(_HEURISTIC_REGISTRY.keys()),
        )
        return _Base()

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


def get_registered_heuristic_class(
    template_name: str, device_type: str, op_name: str
) -> None | type["TemplateConfigHeuristics"]:
    """The rule itself rather than one made from it, for a caller that wants the class.

    Asked by a caller that is about to make the rule itself -- to read what it
    was registered as, or to register a different one beside it.
    """

    return _lookup(template_name, device_type, op_name)


# -- generated kernels ------------------------------------------------------


class CodegenConfigHeuristics:
    """The rule about what a generated kernel should be configured with.

    A base rather than nothing, so that a kind of kernel with no rule registered
    still answers every question with something: the questions are asked of every
    generated kernel, and a rule that has to be written before any of them can be
    compiled is a rule that will be written for one and not the others.
    """

    def get_configs(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError


def register_codegen_heuristic(
    name: str,
    device_type: str | None = None,
    register: bool = True,
) -> Any:
    """Register a rule about how a generated kernel is configured on a device."""

    def decorator(cls):
        if register:
            key: tuple[str | None, ...] = (name, device_type, None)
            _register(key, cls)
            log.info(
                "Registered codegen heuristic: %s for name=%s, device_type=%s",
                cls.__name__, name, device_type,
            )
        return cls

    return decorator


def get_codegen_heuristic(name: str, device_type: str) -> CodegenConfigHeuristics:
    """The rule for this kind of kernel on this device.

    Made once and then remembered, for the same reason template rules are.
    """

    cache_key: tuple[str | None, ...] = (name, device_type, None)
    if cache_key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[cache_key]

    heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        from . import triton_codegen as _triton_codegen

        del _triton_codegen
        heuristic_class = _lookup(name, device_type, None)

    if heuristic_class is None:
        raise ValueError(
            f"No codegen heuristic found - name={name}, device_type={device_type}. "
            f"Available: {list(_HEURISTIC_REGISTRY.keys())}"
        )

    instance = heuristic_class()
    _HEURISTIC_CACHE[cache_key] = instance
    return instance


# -- utilities --------------------------------------------------------------


def clear_registry() -> None:
    """Forget every registered rule and every rule already made.

    For a test that registers a rule and must not leave it behind, and for a
    program that has registered a rule which turns out to be wrong: a rule that
    stays registered is a rule that keeps answering after it was found to be
    wrong, which is the harder half of the mistake to notice.
    """

    _HEURISTIC_REGISTRY.clear()
    _HEURISTIC_CACHE.clear()


@contextlib.contextmanager
def override_template_heuristics(
    device_type: str,
    template_op_pairs: list[tuple[str, str]],
    override_heuristic_class: type["TemplateConfigHeuristics"] | None = None,
) -> "Iterator[None]":
    """Use different rules for the enclosed work, and then go back.

    What a rule says depends on what is being measured, and a measurement
    sometimes wants to be taken against a different rule than the one the program
    would otherwise use -- to ask what would happen with a narrower search, or to
    ask the same question on a device the table is not for.  The previous rules
    are put back rather than cleared, so that an override inside another override
    does not leave the outer one undone.
    """

    if override_heuristic_class is None:
        from .template.base import TemplateConfigHeuristics as _Base

        override_heuristic_class = _Base
    original_entries: dict[tuple[str | None, ...], Any] = {}
    new_keys: list[tuple[str | None, ...]] = []
    _HEURISTIC_CACHE.clear()
    try:
        for template_name, op_name in template_op_pairs:
            key: tuple[str | None, ...] = (template_name, device_type, op_name)
            original_entries[key] = _HEURISTIC_REGISTRY.get(key)
            _HEURISTIC_REGISTRY[key] = override_heuristic_class
            new_keys.append(key)
        yield
    finally:
        for key in new_keys:
            if original_entries.get(key) is None:
                _HEURISTIC_REGISTRY.pop(key, None)
            else:
                _HEURISTIC_REGISTRY[key] = original_entries[key]
        _HEURISTIC_CACHE.clear()


__all__ = [
    "CodegenConfigHeuristics",
    "clear_registry",
    "get_codegen_heuristic",
    "get_registered_heuristic_class",
    "get_template_heuristic",
    "override_template_heuristics",
    "register_codegen_heuristic",
    "register_template_heuristic",
]
