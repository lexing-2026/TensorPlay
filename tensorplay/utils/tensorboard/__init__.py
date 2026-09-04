"""Summary logging: ``tensorplay.utils.tensorboard``.

Requires the ``tensorboard`` package; all event-file encoding, framing and
flushing is delegated to its writer stack, so this module only provides the
high-level API surface over it::

    from tensorplay.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir="runs")
    writer.add_scalar("train/loss", 0.42, step=1)

The resulting run directories open with the standard
``tensorboard --logdir runs`` command.
"""

import re

import tensorboard

if not hasattr(tensorboard, "__version__") or tuple(
    int(part) for part in re.findall(r"\d+", tensorboard.__version__.split("+")[0])[:2]
) < (1, 15):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del re
del tensorboard

from .writer import FileWriter, SummaryWriter
from tensorboard.summary.writer.record_writer import RecordWriter

__all__ = ["FileWriter", "RecordWriter", "SummaryWriter"]
