import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.distributed._tools.mem_tracker import MemTracker
from tensorplay.distributed._tools.memory_tracker import MemoryTracker
from tensorplay.distributed._tools.runtime_estimator import RuntimeEstimator
from tensorplay.distributed._tools.sac_estimator import SACEstimator
from tensorplay.utils._dispatch import _get_current_dispatch_mode


def _model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_memory_tracker_samples_every_operator():
    model = _model()
    tracker = MemoryTracker()
    tracker.start_monitor(model)
    try:
        model(tp.randn(3, 8)).sum()
    finally:
        tracker.stop()
    assert tracker._op_index > 0


def test_runtime_estimator_sees_dispatched_operators():
    model = _model()
    with RuntimeEstimator() as estimator:
        assert _get_current_dispatch_mode() is estimator
        model(tp.randn(3, 8)).sum()
    assert estimator.total_runtime > 0.0
    assert _get_current_dispatch_mode() is None


def test_sac_estimator_records_operator_metadata():
    model = _model()
    with SACEstimator() as estimator:
        model(tp.randn(3, 8)).sum()
    assert len(estimator._sac_metadata) > 0 or estimator.sac_mod_stats


def test_mem_tracker_is_active_while_entered():
    model = _model()
    tracker = MemTracker()
    with tracker:
        assert _get_current_dispatch_mode() is tracker
        model(tp.randn(3, 8)).sum()
    assert _get_current_dispatch_mode() is None
    snapshot = tracker.get_tracker_snapshot("peak")
    assert snapshot
