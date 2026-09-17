"""Tests for the monitoring primitives: events, handlers and stats."""

import time

import tensorplay as tp
from tensorplay import monitor


def test_event_roundtrip_and_handlers():
    seen = []
    handle = monitor.register_event_handler(seen.append)
    try:
        event = monitor.Event(
            name="unit.test", timestamp=time.time(), data={"a": 1.0, "b": 2})
        monitor.log_event(event)
        assert len(seen) == 1
        assert seen[0].name == "unit.test"
        assert seen[0].data["a"] == 1.0
    finally:
        monitor.unregister_event_handler(handle)
    monitor.log_event(
        monitor.Event(name="unit.test", timestamp=time.time(), data={}))
    # the unregistered handler must not see further events
    assert all(e.name == "unit.test" for e in seen)
    assert len(seen) == 1


def test_stat_window_aggregations():
    stat = monitor.Stat(
        "unit.latency", [monitor.MEAN, monitor.COUNT, monitor.MAX,
                         monitor.MIN, monitor.SUM, monitor.VALUE],
        window_size=10.0)
    for v in (3.0, 4.0, 8.0):
        stat.add(v)
    assert stat.count() == 3
    by_type = {r.type: r.value for r in stat.get()}
    assert by_type[monitor.COUNT] == 3
    assert by_type[monitor.SUM] == 15.0
    assert by_type[monitor.MEAN] == 5.0
    assert by_type[monitor.MAX] == 8.0
    assert by_type[monitor.MIN] == 3.0
    assert by_type[monitor.VALUE] == 8.0


def test_stat_window_export_event():
    seen = []
    handle = monitor.register_event_handler(seen.append)
    try:
        stat = monitor.Stat("unit.export", [monitor.COUNT, monitor.MEAN],
                            window_size=0.01)
        stat.add(2.0)
        stat.add(4.0)
        # the tiny window rolls over on the next add
        time.sleep(0.02)
        stat.add(1.0)
        exports = [e for e in seen if e.name == monitor.STAT_EVENT]
        assert exports, "window rollover must emit a stat event"
        data = exports[-1].data
        assert data["unit.export.COUNT"] == 2.0
        assert data["unit.export.MEAN"] == 3.0
    finally:
        monitor.unregister_event_handler(handle)


def test_stat_max_samples_cap():
    stat = monitor.Stat("unit.capped", [monitor.COUNT],
                        window_size=10.0, max_samples=2)
    for v in range(10):
        stat.add(float(v))
    assert stat.count() == 2
    assert stat.get()[0].value == 2.0
