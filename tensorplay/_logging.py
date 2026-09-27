"""Internal logging helpers, and the registrations that build the loggers.

Only what has a consumer inside this library lives here.  The loggers
themselves are ordinary ``logging`` loggers named by dotted path, and anything
that wants one names it; what is here is the small number of behaviours those
loggers are asked for that plain logging does not do on its own.
"""

from __future__ import annotations

import functools


__all__ = [
    "warning_once",
]


@functools.cache
def warning_once(logger_obj, *args, **kwargs) -> None:
    """Warn, but only the first time this exact warning is reached.

    A warning that is emitted on every pass stops being read after the first
    time, which is the same as not warning at all -- except that it costs
    something on every pass.  So a warning whose text is fixed is remembered
    and not repeated.

    What is remembered is the whole call, so two places that warn with the same
    text are one warning between them, not two.  That is the right reading
    when each warning is written to be unique, which is the assumption here;
    where it is not, the warning needs to say something that distinguishes it.
    """

    logger_obj.warning(*args, **kwargs)
