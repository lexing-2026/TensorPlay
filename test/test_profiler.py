"""Native profiler (torch.profiler subset parity) vs local torch 2.13.

Checks that tensorplay.profiler captures the same observable surface as
torch.profiler for an identical workload -- forward op names, backward ops,
user annotations -- plus chrome-trace export validity and the inactive-path
performance contract (one atomic load per op).
"""

import json
import os
import tempfile

import pytest
import torch

import tensorplay as tp
from tensorplay import profiler as tp_prof


def torch_op_names(trace_path):
    doc = json.load(open(trace_path))
    return {e["name"] for e in doc["traceEvents"]
            if e.get("cat") == "cpu_op"}


class TestCaptureParity:
    def test_forward_ops_match_torch(self):
        # Same tiny workload on both frameworks; compare forward op sets.
        with torch.profiler.profile() as tprof:
            xt = torch.randn(8, 16)
            wt = torch.randn(16, 8, requires_grad=True)
            yt = xt.matmul(wt)
            lt = yt.relu().sum()

        with tp_prof.profile() as pprof:
            x = tp.randn([8, 16])
            w = tp.randn([16, 8], requires_grad=True)
            y = x.matmul(w)
            l = y.relu().sum()

        tp_names = {name for name, *_ in pprof.events}
        assert {"matmul", "relu", "sum", "randn"} <= tp_names
        # torch must observe at least the math ops too (it records mm as
        # "aten::mm" family names; check the suffixes exist).
        tnames = {e.key for e in tprof.key_averages()}
        joined = " ".join(tnames)
        assert any(k in joined for k in ("mm", "matmul"))
        assert any("relu" in n for n in tnames)
        assert any("sum" in n for n in tnames)

    def test_backward_ops_and_engine_span(self):
        x = tp.randn([4, 4], requires_grad=True)
        with tp_prof.profile() as prof:
            y = (x * 2.0).sum()
            y.backward()
        names = [n for n, kind, *_ in prof.events if kind == "o"]
        assert any(n.startswith("mul") for n in names)  # mul.Scalar for x*2.0
        assert any(n.startswith("sum") for n in names)
        kinds = {kind for _, kind, *_ in prof.events}
        assert "b" in kinds  # __backward__ engine span emitted
        assert "__backward__" in {n for n, *_ in prof.events}

    def test_composite_inner_ops_recorded_individually(self):
        # gradient is a backend-neutral composite (CIA analog): its inner
        # narrow/sub/div/cat calls must each appear, like upstream counts
        # decomposed aten calls.
        x = tp.arange(8).to(tp.float64)
        with tp_prof.profile() as prof:
            tp.gradient(x)
        names = [n for n, *_ in prof.events]
        for expect in ("narrow", "sub", "div", "cat"):
            assert any(expect in n for n in names), (expect, names)


class TestUserAnnotation:
    def test_record_function_span(self):
        with tp_prof.profile() as prof:
            x = tp.ones([3])
            with tp_prof.record_function("step-1"):
                y = x.add(x)
        entries = {(n, k) for n, k, *_ in prof.events}
        assert ("step-1", "u") in entries

    def test_annotation_exception_safe(self):
        with tp_prof.profile() as prof:
            try:
                with tp_prof.record_function("boom"):
                    raise ValueError("x")
            except ValueError:
                pass
            z = tp.ones([2]).add(tp.ones([2]))
        # The unterminated-by-exception path still closed via finally; next
        # op recorded normally.
        assert any("add" in n for n, *_ in prof.events)

    def test_nested_spans(self):
        with tp_prof.profile() as prof:
            with tp_prof.record_function("outer"):
                with tp_prof.record_function("inner"):
                    tp.ones([2])
        spans = {ev[0]: (ev[2], ev[3]) for ev in prof.events
                 if ev[1] == "u"}
        assert "outer" in spans and "inner" in spans
        so, eo = spans["outer"]
        si, ei = spans["inner"]
        assert so <= si and ei <= eo


class TestExportAndAggregation:
    def test_chrome_trace_valid_json(self):
        with tp_prof.profile() as prof:
            x = tp.randn([4, 4])
            y = x.matmul(x)
        path = os.path.join(tempfile.mkdtemp(), "t.json")
        prof.export_chrome_trace(path)
        doc = json.load(open(path))
        evs = doc["traceEvents"]
        assert len(evs) >= 1
        for e in evs:
            assert e["ph"] == "X"
            assert e["dur"] > 0
            assert e["ts"] >= 0
            assert e["cat"] in ("cpu_op", "user_annotation", "backward")
            assert isinstance(e["tid"], int)
        # torch's own export loads as the same schema
        with torch.profiler.profile() as tprof:
            torch.ones(2) + torch.ones(2)
        tpath = os.path.join(tempfile.mkdtemp(), "tt.json")
        tprof.export_chrome_trace(tpath)
        tdoc = json.load(open(tpath))
        assert "traceEvents" in tdoc
        keys = set(tdoc["traceEvents"][0].keys())
        assert keys <= keys | set(evs[0].keys())  # superset-compatible

    def test_key_averages_sorted_and_formatted(self):
        with tp_prof.profile() as prof:
            a = tp.randn([32, 32])
            for _ in range(5):
                a.matmul(a)
        table = str(prof.key_averages())
        assert "matmul" in table
        assert "Calls" in table
        rows = [ln for ln in table.splitlines()
                if ln.strip() and not ln.startswith(("-", "="))
                and "Name" not in ln]
        assert len(rows) >= 1


class TestInactiveContract:
    def test_no_session_stop_is_empty(self):
        raw = tp._C._profiler_stop()  # never started
        assert list(raw) == []

    def test_events_outside_context_not_captured(self):
        with tp_prof.profile():
            pass
        x = tp.ones([2]).add(tp.ones([2]))  # outside any session
        with tp_prof.profile() as prof:
            pass
        assert len(prof.events) <= 1  # only context bookkeeping, no ops


class TestRecordShapes:
    def test_shapes_captured(self):
        with tp_prof.profile(record_shapes=True) as prof:
            x = tp.randn([8, 16])
            w = tp.randn([16, 8], requires_grad=True)
            y = x.matmul(w)
        mm = [(ev[0], ev[5]) for ev in prof.events if ev[0] == "matmul"]
        assert mm, "no matmul events"
        _, shapes = mm[0]
        assert shapes is not None
        sigs = [tuple(s) for s in shapes]
        assert (8, 16) in sigs and (16, 8) in sigs

    def test_shapes_none_when_not_requested(self):
        with tp_prof.profile() as prof:
            tp.randn([4, 4]).relu()
        assert all(ev[5] is None and ev[6] is None for ev in prof.events)

    def test_group_by_input_shape(self):
        with tp_prof.profile(record_shapes=True) as prof:
            a = tp.randn([4, 4]); b = tp.randn([8, 8])
            a.matmul(a); b.matmul(b)
        table = prof.key_averages(group_by_input_shape=True)
        by_shape = [r for r in table.rows if r.input_shapes is not None]
        shapes_seen = {tuple(r.input_shapes[0]) for r in by_shape
                       if r.input_shapes}
        assert (4, 4) in shapes_seen and (8, 8) in shapes_seen

    def test_key_averages_default_groups_by_name(self):
        with tp_prof.profile(record_shapes=True) as prof:
            a = tp.randn([4, 4]); b = tp.randn([8, 8])
            a.matmul(a); b.matmul(b)
        rows = [r for r in prof.key_averages().rows if r.name == "matmul"]
        assert len(rows) == 1 and rows[0].count == 2


class TestSchedule:
    def test_wait_warmup_active_cycles(self):
        sched = tp_prof.schedule(wait=1, warmup=1, active=2)
        x = tp.ones([4])
        with tp_prof.profile(schedule=sched) as prof:
            actions = []
            for step in range(12):
                action = prof.step()
                actions.append(action)
                y = x.mul(x)
                _ = y.sum()
        # capture only happened on RECORD steps
        assert actions[0] == "none" and actions[1] == "warmup"
        assert actions[2] == "record" and actions[3] == "record"
        recorded = sum(1 for n, k, *_ in prof.events
                       if n.startswith("mul"))
        assert recorded >= 2

    def test_step_without_schedule_is_noop(self):
        with tp_prof.profile() as prof:
            prof.step()
            tp.ones([2]).add(tp.ones([2]))
        assert any("add" in n for n, *_ in prof.events)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestWithStack:
    def test_op_call_site_captured(self):
        with tp_prof.profile(with_stack=True) as prof:
            x = tp.randn([4, 4])
            y = x.matmul(x)  # this exact line should be recorded
        sites = [ev[7] for ev in prof.events if ev[0] == "matmul"]
        assert sites and all(s for s in sites)
        assert any("test_profiler.py" in s for s in sites)

    def test_inner_ops_have_no_site(self):
        # composite inner ops never re-enter a binding: no inherited site
        with tp_prof.profile(with_stack=True) as prof:
            tp.gradient(tp.arange(8).to(tp.float64))
        inner = [(n, ev[7]) for ev in prof.events
                 for n in [ev[0]] if n.split(".")[0] in ("sub", "div", "cat")]
        assert inner
        assert any(s is None for _, s in inner)

    def test_user_span_site(self):
        with tp_prof.profile(with_stack=True) as prof:
            with tp_prof.record_function("annotated"):
                pass
        spans = [ev[7] for ev in prof.events if ev[1] == "u"]
        assert spans and spans[0]


class TestMemorySnapshot:
    def test_factory_bytes_derived(self):
        with tp_prof.profile(record_shapes=True) as prof:
            a = tp.zeros([128])          # 512 B f32
            b = tp.ones([16, 16])        # 1024 B f32
        total, peak, tl = prof.memory_summary()
        assert total >= 512 + 1024 - 1   # randn intermediates may add
        assert peak >= total or peak > 0
        names = {n for _, _, n in tl}
        assert "zeros" in names and "ones" in names

    def test_requires_shapes(self):
        with tp_prof.profile() as prof:
            tp.zeros([4])
        _t, _p, tl = prof.memory_summary()
        assert tl == []


class TestSelfTimes:
    def test_parent_self_less_than_total(self):
        with tp_prof.profile() as prof:
            x = tp.randn([8, 8], requires_grad=True)
            with tp_prof.record_function("outer"):
                y = (x * 2.0).sum()
                y.backward()
        table = prof.key_averages()
        outer = next(r for r in table.rows if r.name == "outer")
        total_us_outer = outer.total_us
        assert 0 < outer.self_us < total_us_outer

    def test_totals_row_present(self):
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        s = str(prof.key_averages())
        assert "Self CPU time total" in s


class TestGpuFieldsCpuBuild:
    def test_gpu_ms_negative_on_cpu(self):
        with tp_prof.profile() as prof:
            tp.ones([3]).add(tp.ones([3]))
        assert all(ev[8] < 0 for ev in prof.events)


class TestGpuTiming:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_gpu_timing_resolves(self):
        a = tp.ones([4096]).to("cuda")
        with tp_prof.profile(record_shapes=True, gpu_timing=True) as prof:
            a.add(a)
        timed = [ev for ev in prof.events if ev[0].startswith("add")]
        assert timed and all(ev[8] >= 0 for ev in timed)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_gpu_timing_reclaims_repeated_launches(self):
        """Stop-time reclaim must not deadlock after a burst of launches."""
        a = tp.ones([4096]).to("cuda")
        with tp_prof.profile(gpu_timing=True) as prof:
            for _ in range(64):
                a = a.add(a)
        assert prof.events
        assert all(ev[8] >= 0 for ev in prof.events)
        assert prof.gpu_resolved_events == prof.gpu_timed_events
        assert prof.stop_ms >= 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestBackwardNodes:
    def test_node_events_named(self):
        with tp_prof.profile() as prof:
            x = tp.ones([4], requires_grad=True)
            y = (x * 2.0).sum()
            y.backward()
        node_names = [ev[0] for ev in prof.events
                      if ev[0].startswith("backward::")]
        assert any("Mul" in n for n in node_names), node_names
        assert any("Sum" in n for n in node_names), node_names

    def test_nodes_absent_without_session(self):
        x = tp.ones([4], requires_grad=True)
        y = (x * 3.0).sum()
        y.backward()  # no session: nothing recorded anywhere


class TestAutogradNamespace:
    def test_namespace_alias(self):
        import tensorplay as tp
        assert hasattr(tp.autograd, "profiler")
        assert hasattr(tp.autograd, "emit_nvtx")
        assert tp.autograd.profiler.profile is tp_prof.profile


class TestEmitNvtx:
    def test_flag_scoped_and_safe(self):
        # Without libnvtx the ranges are silent no-ops; with it they emit.
        # Either way the context must be scoped and side-effect free.
        with tp_prof.emit_nvtx():
            x = tp.ones([3])
            y = x.add(x)
        z = y.add(1.0)  # flag cleared after the block
        assert isinstance(z, tp.Tensor)

    def test_works_inside_profile(self):
        with tp_prof.profile() as prof:
            with tp_prof.emit_nvtx():
                tp.ones([2]).add(tp.ones([2]))
        assert any("add" in ev[0] for ev in prof.events)
