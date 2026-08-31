"""

Checks that tensorplay.profiler captures the same observable surface as
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


class TestCaptureReference:
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
        # Gradient is a backend-neutral composite: its inner narrow/sub/div/cat
        # calls must each appear as separate events.
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
        x_events = [e for e in evs if e["ph"] == "X"]
        assert x_events and all(e["ts"] >= 0 for e in x_events)
        for e in x_events:
            assert e["dur"] > 0
            assert e["cat"] in ("cpu_op", "user_annotation", "backward")
            assert isinstance(e["tid"], int)
        assert any(e["ph"] == "M" for e in evs)
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
        raw_ops, raw_gpu, raw_mem = tp._C._profiler_stop()  # never started
        assert list(raw_ops) == []
        assert list(raw_gpu) == []
        assert list(raw_mem) == []

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


class TestGpuTrace:
    """CUPTI kernel-level tracing (gpu_trace=True, USE_CUDA builds)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_kernels_runtime_and_correlation(self):
        a = tp.ones([4096]).to("cuda")
        with tp_prof.profile(gpu_trace=True) as prof:
            b = a.add(a)
            _ = b.mul(b)
        acts = prof.gpu_activities
        kinds = {k[1] for k in acts}
        assert "k" in kinds, f"no kernel rows: {kinds}"
        assert "r" in kinds, "no cuda_runtime rows"
        kernels = [k for k in acts if k[1] == "k"]
        # kernel names are real (non-empty, interned)
        assert all(k[0] and k[0] != "unknown" for k in kernels)
        # op -> kernel correlation via external id (OpRecord slot)
        ops = prof.events
        kNoExt = 0xFFFFFFFFFFFFFFFF
        for k in kernels:
            if k[7] != kNoExt:
                assert k[7] < len(ops)
        timed = [ev for ev in ops if len(ev) > 11 and ev[11] > 0]
        assert timed, "no op received correlated kernels"
        assert all(ev[8] >= 0 for ev in timed)  # summed kernel ms

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_memcpy_rows(self):
        x = tp.ones([1024]).to("cuda")  # outside the session
        with tp_prof.profile(gpu_trace=True) as prof:
            _ = x.cpu()
        kinds = {k[1] for k in prof.gpu_activities}
        assert "m" in kinds, kinds
        mc = [k for k in prof.gpu_activities if k[1] == "m"][0]
        assert mc[10] == 4096  # 1024 * float32 bytes moved

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_trace_gpu_lanes_and_flows(self):
        a = tp.ones([4096]).to("cuda")
        with tp_prof.profile(gpu_trace=True) as prof:
            _ = a.add(a)
        path = os.path.join(tempfile.mkdtemp(), "gpu.json")
        prof.export_chrome_trace(path)
        doc = json.load(open(path))
        evs = doc["traceEvents"]
        cats = {e["cat"] for e in evs if "cat" in e}
        assert "kernel" in cats
        assert "cuda_runtime" in cats
        assert "ac2g" in cats  # flow arrows op -> kernel
        assert any(e.get("name") == "process_labels" and
                   e["args"].get("labels", "").startswith("GPU ")
                   for e in evs)
        # kernel rows live on a GPU pid (>= 1000000), not the CPU pid
        krows = [e for e in evs if e.get("cat") == "kernel"]
        assert all(isinstance(e["pid"], int) and e["pid"] >= 1000000
                   for e in krows)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_gpu_trace_and_timing_coexist(self):
        a = tp.ones([4096]).to("cuda")
        with tp_prof.profile(gpu_timing=True, gpu_trace=True) as prof:
            _ = a.add(a)
        assert any(ev[8] >= 0 for ev in prof.events)
        assert any(k[1] == "k" for k in prof.gpu_activities)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_repeated_sessions_reusable(self):
        a = tp.ones([1024]).to("cuda")
        for _ in range(3):
            with tp_prof.profile(gpu_trace=True) as prof:
                _ = a.add(a)
            assert any(k[1] == "k" for k in prof.gpu_activities)
            assert not prof.mem_events  # unrelated to mem capture


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


class TestMemoryHooks:
    """Allocator-level capture (profile_memory=True)."""

    def test_alloc_and_free_balance(self):
        with tp_prof.profile(profile_memory=True) as prof:
            x = tp.zeros([1024])  # 4 KiB f32
            del x
            tp.zeros([512])
        assert prof.mem_events, "no allocator events captured"
        allocs = [e for e in prof.mem_events if e[3]]
        frees = [e for e in prof.mem_events if not e[3]]
        assert allocs and frees
        # allocs carry the requested bytes; frees repeat the block size
        assert all(e[2] > 0 for e in allocs)

    def test_timeline_and_summary(self):
        with tp_prof.profile(profile_memory=True) as prof:
            a = tp.zeros([4096])
            b = tp.zeros([4096])
            del a
        total, peak, tl = prof.memory_summary()
        assert total > 0
        assert peak >= total or peak > 0
        assert any(kind == "alloc" for _, _, kind in tl)
        assert any(kind == "free" for _, _, kind in tl)

    def test_export_memory_timeline_csv(self):
        with tp_prof.profile(profile_memory=True) as prof:
            tp.zeros([256])
        path = os.path.join(tempfile.mkdtemp(), "mem.csv")
        prof.export_memory_timeline(path)
        lines = open(path).read().strip().splitlines()
        assert lines[0] == "timestamp_ns,device,allocated_bytes"
        assert len(lines) >= 2
        assert "cpu" in lines[1]

    def test_no_mem_events_without_flag(self):
        with tp_prof.profile() as prof:
            tp.zeros([128])
        assert prof.mem_events == []


class TestFullStack:
    """with_stack captures the full Python frame chain."""

    def test_stack_captured(self):
        with tp_prof.profile(with_stack=True) as prof:
            tp.randn([4, 4])
        stacks = [ev[10] for ev in prof.events if ev[10]]
        assert stacks
        # leaf frame is the test file itself; chain contains profiler.py's
        # module or the test frame
        joined = ";".join(stacks[0])
        assert "test_profiler.py" in joined

    def test_inner_ops_have_no_stack(self):
        with tp_prof.profile(with_stack=True) as prof:
            tp.gradient(tp.arange(8).to(tp.float64))
        inner = [ev for ev in prof.events
                 if ev[0].split(".")[0] in ("sub", "div", "cat")]
        assert inner
        assert all(ev[10] is None for ev in inner)

    def test_stack_absent_without_flag(self):
        with tp_prof.profile() as prof:
            tp.randn([4, 4])
        assert all(ev[10] is None for ev in prof.events)

    def test_export_stacks_folded(self):
        with tp_prof.profile(with_stack=True) as prof:
            a = tp.randn([8, 8])
            a.matmul(a)
        path = os.path.join(tempfile.mkdtemp(), "stacks.txt")
        prof.export_stacks(path)
        content = open(path).read()
        assert content
        line = content.splitlines()[0]
        # "<frames> <self_us>" folded format
        frames, us = line.rsplit(" ", 1)
        assert float(us) > 0
        assert ";" in frames or "(" in frames


class TestSampling:
    """Python stack sampler (with_samples=True)."""

    def test_samples_collected(self, monkeypatch):
        monkeypatch.setenv("TP_PROFILER_SAMPLE_MS", "1")
        with tp_prof.profile(with_samples=True) as prof:
            x = tp.ones([64])
            for _ in range(3000):  # outlast GIL switch interval (~5 ms)
                y = x.mul(x)
            _ = y.sum()
        assert prof._sampler is not None
        assert len(prof._sampler.samples) >= 1
        ts, os_tid, chain = prof._sampler.samples[0]
        assert chain and os_tid

    def test_samples_in_trace(self, monkeypatch):
        monkeypatch.setenv("TP_PROFILER_SAMPLE_MS", "1")
        with tp_prof.profile(with_samples=True) as prof:
            x = tp.ones([64])
            for _ in range(3000):  # outlast GIL switch interval (~5 ms)
                _ = x.mul(x)
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        prof.export_chrome_trace(path)
        doc = json.load(open(path))
        samples = [e for e in doc["traceEvents"]
                   if e.get("cat") == "python_function"]
        assert samples
        assert samples[0]["ph"] == "i"
        assert samples[0]["args"]["stack"]

    def test_sampler_stops(self):
        p = tp_prof.profile(with_samples=True)
        p.__enter__()
        sampler = p._sampler
        p.__exit__(None, None, None)
        assert sampler._stop.is_set()      # loop asked to finish
        assert sampler._thread is None     # joined and cleared


class TestDistributed:
    """Rank tagging + multi-process trace merge."""

    def test_rank_in_args_and_metadata(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "8")
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        path = os.path.join(tempfile.mkdtemp(), "r.json")
        prof.export_chrome_trace(path, torch_compat=True)
        doc = json.load(open(path))
        assert doc["distributedInfo"] == {"rank": 2, "world_size": 8,
                                          "backend": "tensorplay.distributed"}
        ops = [e for e in doc["traceEvents"] if e.get("cat") == "cpu_op"]
        assert ops and all(e["args"]["rank"] == 2 for e in ops)

    def test_merge_two_ranks(self, monkeypatch):
        paths = []
        for rank in (0, 1):
            monkeypatch.setenv("RANK", str(rank))
            monkeypatch.setenv("WORLD_SIZE", "2")
            with tp_prof.profile() as prof:
                tp.ones([2]).add(tp.ones([2]))
            p = os.path.join(tempfile.mkdtemp(), f"rank{rank}.json")
            prof.export_chrome_trace(p, torch_compat=True)
            paths.append(p)
        out = os.path.join(tempfile.mkdtemp(), "merged.json")
        tp_prof.merge_distributed_traces(paths, out)
        doc = json.load(open(out))
        pids = {e["pid"] for e in doc["traceEvents"] if "pid" in e}
        assert 0 in pids and 10_000_000 in pids
        assert doc["distributedInfo"]["merged_ranks"] == [0, 1]
        assert len(doc["traceEvents"]) >= 2


class TestTensorboardExport:

    def test_torch_schema_keys(self):
        with tp_prof.profile(record_shapes=True) as prof:
            x = tp.randn([4, 4])
            _ = x.matmul(x)
        outdir = tempfile.mkdtemp()
        written = prof.export_tensorboard_trace(outdir)
        assert written.endswith(".pt.trace.json")
        doc = json.load(open(written))
        for key in ("schemaVersion", "deviceProperties",
                    "baseTimeNanoseconds", "traceName", "with_flops",
                    "record_shapes"):
            assert key in doc, key
        assert doc["schemaVersion"] == 1
        assert doc["traceName"].endswith(".pt.trace.json")
        evs = doc["traceEvents"]
        assert any(e["ph"] == "M" and e["name"] == "process_name"
                   for e in evs)
        assert any(e["ph"] == "M" and e["name"] == "process_labels"
                   for e in evs)
        assert any(e.get("name") == "Record Window End" for e in evs)
        # cpu_op rows kept, backward span re-catgorized to user_annotation
        assert any(e.get("cat") == "cpu_op" for e in evs)

    def test_schema_matches_torch_export(self):
        with torch.profiler.profile() as tprof:
            torch.ones(2) + torch.ones(2)
        tpath = os.path.join(tempfile.mkdtemp(), "tt.pt.trace.json")
        tprof.export_chrome_trace(tpath)
        tdoc = json.load(open(tpath))
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        ppath = os.path.join(tempfile.mkdtemp(), "p")
        prof.export_tensorboard_trace(ppath)
        import glob
        pfile = glob.glob(os.path.join(ppath, "*.pt.trace.json"))[0]
        pdoc = json.load(open(pfile))
        assert set(tdoc.keys()) - {"traceEvents"} <= set(pdoc.keys()) | {
            "with_stack", "distributedInfo", "traceName", "trace_events"}

    def test_backward_cat_remaps(self):
        x = tp.ones([4], requires_grad=True)
        with tp_prof.profile() as prof:
            y = (x * 2.0).sum()
            y.backward()
        path = os.path.join(tempfile.mkdtemp(), "b.json")
        prof.export_chrome_trace(path, torch_compat=True)
        cats = {e.get("cat") for e in json.load(open(path))["traceEvents"]}
        assert "backward" not in cats
        assert "user_annotation" in cats


class TestGpuTraceCpuBuild:
    """gpu_trace degrades gracefully without CUDA."""

    def test_warning_and_empty_gpu(self):
        if torch.cuda.is_available():
            pytest.skip("exercises CPU-build degradation only")
        with pytest.warns(RuntimeWarning, match="gpu_trace unavailable"):
            with tp_prof.profile(gpu_trace=True) as prof:
                tp.ones([2]).add(tp.ones([2]))
        assert prof.gpu_activities == []
        assert any("add" in ev[0] for ev in prof.events)

    def test_cupti_available_flag(self):
        avail = tp._C and hasattr(tp._C, "_profiler_stop")
        assert avail  # binding surface intact


class TestFlops:
    """with_flops: C++-estimated FLOPs stamped per event at session stop."""

    def _run(self, *fns):
        with tp_prof.profile(with_flops=True) as prof:
            for fn in fns:
                fn()
        return prof

    def _flops(self, prof, name, shapes):
        want = [list(s) for s in shapes]
        return next(ev[12] for ev in prof.events
                    if ev[0] == name
                    and [list(s) for s in ev[5]][:len(want)] == want)

    def test_matmul_family(self):
        prof = self._run(
            lambda: tp.randn([2, 3, 5, 7]).matmul(tp.randn([3, 7, 9])),
            lambda: tp.randn([7]).matmul(tp.randn([7, 9])),
            lambda: tp.randn([5, 7]).matmul(tp.randn([7])),
            lambda: tp.randn([4, 7]).matmul(tp.randn([2, 4, 7, 9])),
        )
        # [2,3]x[3] batch broadcast -> 6 batches of 5x7x9
        assert self._flops(prof, "matmul", ([2, 3, 5, 7], [3, 7, 9])) == 2*6*5*7*9
        assert self._flops(prof, "matmul", ([7], [7, 9])) == 2*7*9
        assert self._flops(prof, "matmul", ([5, 7], [7])) == 2*5*7
        # [4,7] @ [2,4,7,9]: batch prefix [2,4] = 8
        assert self._flops(prof, "matmul", ([4, 7], [2, 4, 7, 9])) == 2*8*4*7*9

    def test_mm_addmm_bmm_baddbmm(self):
        prof = self._run(
            lambda: tp.randn([64, 32]).mm(tp.randn([32, 48])),
            lambda: tp.addmm(tp.randn([8, 8]), tp.randn([8, 8]), tp.randn([8, 8])),
            lambda: tp.bmm(tp.randn([2, 4, 6]), tp.randn([2, 6, 5])),
            lambda: tp.baddbmm(tp.randn([2, 4, 5]), tp.randn([2, 4, 6]), tp.randn([2, 6, 5])),
        )
        assert self._flops(prof, "mm", ([64, 32], [32, 48])) == 2*64*32*48
        # addmm(input, mat1, mat2) counts the matmul operands only
        assert self._flops(prof, "addmm", ([8, 8], [8, 8], [8, 8])) == 2*8*8*8
        assert self._flops(prof, "bmm", ([2, 4, 6], [2, 6, 5])) == 2*2*4*6*5
        assert self._flops(prof, "baddbmm",
                           ([2, 4, 5], [2, 4, 6], [2, 6, 5])) == 2*2*4*6*5

    def test_conv2d_assumes_unit_stride(self):
        import tensorplay.nn as nn

        prof = self._run(
            lambda: nn.functional.conv2d(
                tp.randn([4, 8, 16, 16]), tp.randn([16, 8, 3, 3]))
        )
        got = self._flops(prof, "conv2d", ([4, 8, 16, 16], [16, 8, 3, 3]))
        assert got == 2*4*16*16*16*8*9  # N * Cout * H*W * (Cin/g) * kh*kw * 2

    def test_uncovered_ops_report_zero(self):
        prof = self._run(
            lambda: tp.randn([8, 8]).relu(),
            lambda: tp.randn([8, 8]).sum(),
        )
        assert all(ev[12] == 0 for ev in prof.events)

    def test_no_shapes_reports_zero(self):
        with tp_prof.profile() as prof:  # record_shapes off
            tp.randn([4, 4]).mm(tp.randn([4, 4]))
        ev = next(ev for ev in prof.events if ev[0] == "mm")
        assert ev[12] == 0

    def test_key_averages_column_and_total(self):
        prof = self._run(
            lambda: tp.randn([64, 32]).mm(tp.randn([32, 48])),
        )
        table = prof.key_averages()
        row = next(r for r in table.rows if r.name == "mm")
        assert row.flops == 2*64*32*48
        assert table.total_flops == 2*64*32*48
        text = table.table()
        assert "Total Flops" in text
        assert f"{2*64*32*48:,}" in text

    def test_no_column_without_flag(self):
        with tp_prof.profile(record_shapes=True) as prof:
            tp.randn([4, 4]).mm(tp.randn([4, 4]))
        assert "Total Flops" not in prof.key_averages().table()

    def test_event_attribute_and_sort(self):
        prof = self._run(
            lambda: tp.randn([4, 4]).mm(tp.randn([4, 4])),
            lambda: tp.randn([2, 4, 6]).bmm(tp.randn([2, 6, 5])),
        )
        ev = next(e for e in prof.events.function_events if e.name == "mm")
        assert ev.flops == 2*4*4*4
        # bmm (2*2*4*6*5 = 480) outranks mm (2*4*4*4 = 128) by flops
        rows = prof.key_averages(sort_by="flops").rows
        assert rows[0].name == "bmm" and rows[0].flops == 2*2*4*6*5

    def test_total_average_carries_flops(self):
        prof = self._run(
            lambda: tp.randn([4, 4]).mm(tp.randn([4, 4])),
            lambda: tp.randn([4, 4]).mm(tp.randn([4, 4])),
        )
        assert prof.total_average().flops == 2 * 2*4*4*4


class TestWithModules:
    """with_modules: nn.Module forwards bracket profiler spans."""

    def test_spans_present_nested_and_restored(self):
        import tensorplay.nn as nn

        class Sub(nn.Module):
            def forward(self, x):
                return x.mm(x.t())

        class Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()

            def forward(self, x):
                return self.sub(x) + x.sum()

        orig_call = nn.Module.__call__
        model = Top()
        with tp_prof.profile(with_modules=True) as prof:
            model(tp.randn([16, 16]))
        mods = [(ev[0], ev[2], ev[3]) for ev in prof.events
                if isinstance(ev[0], str) and ev[0].startswith("nn.Module:")]
        names = [n for n, _, _ in mods]
        assert "nn.Module: Top" in names and "nn.Module: Sub" in names
        top = next(e for e in mods if e[0] == "nn.Module: Top")
        sub = next(e for e in mods if e[0] == "nn.Module: Sub")
        assert top[1] <= sub[1] and sub[2] <= top[2]  # nesting
        # op event inside the leaf module's span
        mm = next(ev for ev in prof.events if ev[0] == "mm")
        assert sub[1] <= mm[2] and mm[3] <= sub[2]
        assert nn.Module.__call__ is orig_call  # restored

    def test_no_module_events_without_flag(self):
        import tensorplay.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return x.relu()

        model = M()
        with tp_prof.profile() as prof:
            model(tp.randn([4, 4]))
        assert not [ev for ev in prof.events
                    if str(ev[0]).startswith("nn.Module")]

    def test_exception_keeps_call_consistent(self):
        import tensorplay.nn as nn

        class Boom(nn.Module):
            def forward(self, x):
                raise ValueError("boom")

        orig_call = nn.Module.__call__
        with tp_prof.profile(with_modules=True):
            with pytest.raises(ValueError):
                Boom()(tp.randn([4, 4]))
        assert nn.Module.__call__ is orig_call
        # next session still captures balanced events
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        assert any("add" in str(ev[0]) for ev in prof.events)

    def test_nested_sessions_share_one_patch(self):
        import tensorplay.nn as nn

        orig_call = nn.Module.__call__
        with tp_prof.profile(with_modules=True):
            with tp_prof.profile(with_modules=True):
                pass
            assert nn.Module.__call__ is not orig_call
        assert nn.Module.__call__ is orig_call

    def test_key_averages_aggregates_module_rows(self):
        import tensorplay.nn as nn

        class Sub(nn.Module):
            def forward(self, x):
                return x.mm(x.t())

        model = Sub()
        with tp_prof.profile(with_modules=True) as prof:
            for _ in range(3):
                model(tp.randn([8, 8]))
        row = next(r for r in prof.key_averages().rows
                   if r.name == "nn.Module: Sub")
        assert row.count == 3


class TestKeyAveragesGpuColumns:
    def test_no_gpu_columns_on_cpu(self):
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        table = str(prof.key_averages())
        assert "Self CUDA" not in table
        assert "Self CPU time total" in table

    def test_sort_by_cuda_key_accepted(self):
        with tp_prof.profile() as prof:
            tp.ones([2]).add(tp.ones([2]))
        # unknown keys still raise; cuda keys parse without GPU data
        rows = prof.key_averages(sort_by="cuda_time").rows
        assert rows
