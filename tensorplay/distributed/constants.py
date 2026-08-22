# Ported from torch/distributed/constants.py. tp's default timeouts live in
# Python (the store layer is pure Python), so the C++ fallbacks are absent.
from datetime import timedelta

__all__ = ["default_pg_timeout", "default_pg_nccl_timeout"]

# Default process group wide timeout, if applicable.
# This only applies to the non-nccl backends
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout: timedelta = timedelta(minutes=30)

# Separate timeout for PGNCCL mainly because it's always been that way in the
# C++ layer, but until recently there was one default that applied across all
# backends in the python layer.
default_pg_nccl_timeout: timedelta | None = None
