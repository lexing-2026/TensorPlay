"""The rules about which configurations of a template are worth trying.

One module per kind of template, and one registration per device the numbers
differ on.  A template that has no rule of its own gets the rule that applies
wherever nothing more specific does, so that a new template is measurable before
anyone has written numbers for it -- measured against the general numbers, which
is a real answer, rather than against nothing, which is not.

This package names nothing at module level, because the registry reaches back into
it for the base class; taking that name eagerly would have the two half-built
whenever either was used.
"""

__all__ = [
    "TemplateConfigHeuristics",
    "clear_registry",
    "get_registered_heuristic_class",
    "get_template_heuristic",
    "override_template_heuristics",
    "register_template_heuristic",
]


def __getattr__(name):
    """The names above, reached through the registry without importing it early.

    Asked for by whoever needs them rather than imported at module level, so that
    the package can be named without the registry being part-built at the time.
    """

    if name in (
        "clear_registry",
        "get_registered_heuristic_class",
        "get_template_heuristic",
        "override_template_heuristics",
        "register_template_heuristic",
    ):
        from .. import registry

        return getattr(registry, name)
    if name == "TemplateConfigHeuristics":
        from .base import TemplateConfigHeuristics

        return TemplateConfigHeuristics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
