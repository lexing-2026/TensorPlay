"""Chrome trace serialization for profiler events."""

from __future__ import annotations

import gzip
import json
import os
import socket
import time
import uuid

from ._utils import dtype_name, rank_world


CPU_PID = 0
GPU_PID_BASE = 1_000_000
NO_EXTERNAL_ID = 0xFFFFFFFFFFFFFFFF


def _device_properties():
    properties = []
    try:
        import tensorplay as tp

        if not getattr(tp.cuda, "is_available", lambda: False)():
            return properties
        for index in range(tp.cuda.device_count()):
            item = tp.cuda.get_device_properties(index)
            properties.append(
                {
                    "id": index,
                    "name": getattr(item, "name", "unknown"),
                    "totalGlobalMem": getattr(item, "total_memory", 0),
                    "computeMajor": getattr(item, "major", 0),
                    "computeMinor": getattr(item, "minor", 0),
                    "maxThreadsPerBlock": getattr(item, "max_threads_per_block", 0),
                    "maxThreadsPerMultiprocessor": getattr(item, "multi_processor_count", 0),
                    "regsPerMultiprocessor": getattr(item, "regs_per_multiprocessor", 0),
                    "warpSize": getattr(item, "warp_size", 32),
                }
            )
    except Exception:
        return []
    return properties


def _open_trace(path):
    path = os.fspath(path)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def _build_document(event_list, trace_events, torch_compat, path):
    rank, world = rank_world()
    document = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
    }
    if not torch_compat:
        return document

    document.update(
        {
            "schemaVersion": 1,
            "deviceProperties": _device_properties(),
            "displayTimeUnit": "ms",
            "baseTimeNanoseconds": event_list.base_ns,
            "with_flops": False,
            "record_shapes": any(len(event) > 5 and event[5] is not None for event in event_list),
            "with_stack": any(len(event) > 10 and event[10] for event in event_list),
            "traceName": os.fspath(path),
            "host_name": socket.gethostname(),
            "trace_id": uuid.uuid4().hex,
        }
    )
    try:
        import tensorplay._C as native

        runtime_version = native._cuda.get_version()
        driver_version = native._cuda.get_driver_version()
    except Exception:
        runtime_version = driver_version = 0
    try:
        import tensorplay._C as native

        cupti_version = native._profiler_cupti_version()
    except Exception:
        cupti_version = 0
    if runtime_version:
        document["cuda_runtime_version"] = runtime_version
    if driver_version:
        document["cuda_driver_version"] = driver_version
    if cupti_version:
        document["cupti_version"] = cupti_version
    if rank is not None:
        document["distributedInfo"] = {
            "rank": rank,
            "world_size": world if world is not None else 0,
            "backend": "tensorplay.distributed",
        }
    return document


def export_chrome_trace(
    event_list,
    path,
    torch_compat=False,
    gpu_activities=None,
    mem_events=None,
    samples=None,
):
    """Write collected CPU, device, memory and sample events as JSON."""
    gpu_activities = event_list.gpu_activities if gpu_activities is None else gpu_activities
    mem_events = event_list.mem_events if mem_events is None else mem_events
    samples = event_list.samples if samples is None else samples

    tids = {}
    lane_meta = {}

    def lane(raw_tid, pid=CPU_PID, name=None):
        key = (pid, raw_tid)
        if key not in tids:
            tids[key] = len(tids)
        lane_id = tids[key]
        if name is not None:
            lane_meta.setdefault((pid, lane_id), name)
        return lane_id

    category_for_kind = {
        "o": "cpu_op",
        "u": "user_annotation",
        "b": "backward",
    }
    if torch_compat:
        category_for_kind["b"] = "user_annotation"
    category_for_gpu_kind = {
        "k": "kernel",
        "m": "gpu_memcpy",
        "s": "gpu_memset",
        "r": "cuda_runtime",
        "d": "cuda_driver",
    }

    trace_events = []
    op_slots = {}
    rank, _world = rank_world()

    def relative_time(timestamp_ns):
        return (timestamp_ns - event_list.base_ns) / 1e3

    for slot, event in enumerate(event_list):
        if len(event) < 5:
            continue
        name, kind, start_ns, end_ns, thread_id = event[:5]
        shapes = event[5] if len(event) > 5 else None
        dtypes = event[6] if len(event) > 6 else None
        site = event[7] if len(event) > 7 else None
        gpu_ms = event[8] if len(event) > 8 else None
        out_bytes = event[9] if len(event) > 9 else None
        kernel_count = event[11] if len(event) > 11 else 0
        thread_lane = lane(thread_id)
        args = {}
        if shapes is not None:
            args["Input Dims"] = [list(shape) for shape in shapes]
            if dtypes is not None:
                args["Input type"] = [dtype_name(dtype) for dtype in dtypes]
        if site:
            args["Call site"] = site
        if gpu_ms is not None and gpu_ms >= 0:
            args["GPU duration (ms)"] = round(gpu_ms, 4)
            if kernel_count:
                args["Kernels"] = kernel_count
        if out_bytes is not None and out_bytes > 0:
            args["Output alloc (bytes)"] = out_bytes
        if rank is not None:
            args["rank"] = rank

        op_slots[slot] = (start_ns, thread_lane)
        trace_events.append(
            {
                "name": name,
                "cat": category_for_kind.get(kind, "cpu_op"),
                "ph": "X",
                "ts": relative_time(start_ns),
                "dur": max((end_ns - start_ns) / 1e3, 0.001),
                "pid": CPU_PID,
                "tid": thread_lane,
                "args": args,
            }
        )

    flows = []
    for activity in gpu_activities:
        (
            name,
            kind,
            start_ns,
            end_ns,
            device,
            stream,
            correlation,
            external_id,
            os_tid,
            _cbid,
            nbytes,
            copy_kind,
            _value,
        ) = activity
        args = {"correlation": correlation}
        if external_id is not None and external_id != NO_EXTERNAL_ID:
            args["External id"] = external_id
        if nbytes:
            args["bytes"] = nbytes
        if kind == "m":
            args["copy kind"] = int(copy_kind)
        if kind in ("r", "d"):
            row_pid = CPU_PID
            row_tid = lane(("api", os_tid), CPU_PID, f"thread {os_tid}")
        else:
            row_pid = GPU_PID_BASE + max(device, 0)
            row_tid = lane(
                ("stream", device, stream),
                row_pid,
                f"stream {stream}",
            )

        trace_events.append(
            {
                "name": name,
                "cat": category_for_gpu_kind.get(kind, "kernel"),
                "ph": "X",
                "ts": relative_time(start_ns),
                "dur": max((end_ns - start_ns) / 1e3, 0.001),
                "pid": row_pid,
                "tid": row_tid,
                "args": args,
            }
        )
        if (
            kind not in ("r", "d")
            and external_id is not None
            and external_id != NO_EXTERNAL_ID
            and external_id in op_slots
        ):
            op_ts, op_tid = op_slots[external_id]
            flows.append(
                (
                    correlation,
                    op_ts,
                    op_tid,
                    CPU_PID,
                    start_ns,
                    row_tid,
                    row_pid,
                )
            )

    for ts, _ptr, nbytes, allocated, is_cuda, device, _stream, thread_id in mem_events:
        trace_events.append(
            {
                "name": "[memory]",
                "cat": "user_annotation",
                "ph": "X",
                "ts": relative_time(ts),
                "dur": 0.001,
                "pid": CPU_PID,
                "tid": lane(thread_id),
                "args": {
                    "Bytes": nbytes,
                    "Allocation Type": "alloc" if allocated else "free",
                    "Device": f"cuda:{device}" if is_cuda else "cpu",
                },
            }
        )

    for ts, os_tid, chain in samples:
        if not chain:
            continue
        trace_events.append(
            {
                "name": chain[0],
                "cat": "python_function",
                "ph": "i",
                "s": "t",
                "ts": relative_time(ts),
                "pid": CPU_PID,
                "tid": lane(("api", os_tid), CPU_PID, f"thread {os_tid}"),
                "args": {"stack": chain},
            }
        )

    for correlation, op_ts, op_tid, op_pid, gpu_ts, gpu_tid, gpu_pid in flows:
        trace_events.append(
            {
                "ph": "s",
                "id": correlation,
                "pid": op_pid,
                "tid": op_tid,
                "ts": relative_time(op_ts),
                "cat": "ac2g",
                "name": "ac2g",
            }
        )
        trace_events.append(
            {
                "ph": "f",
                "id": correlation,
                "pid": gpu_pid,
                "tid": gpu_tid,
                "ts": relative_time(gpu_ts),
                "cat": "ac2g",
                "name": "ac2g",
                "bp": "e",
            }
        )

    used_gpu_pids = sorted(
        {event["pid"] for event in trace_events if isinstance(event.get("pid"), int) and event["pid"] >= GPU_PID_BASE}
    )
    first_ts = min(
        (event["ts"] for event in trace_events if "ts" in event),
        default=0.0,
    )
    metadata = []

    def add_metadata(name, pid, tid, key, value):
        metadata.append(
            {
                "ph": "M",
                "name": name,
                "ts": first_ts,
                "pid": pid,
                "tid": tid,
                "args": {key: value},
            }
        )

    add_metadata("process_name", CPU_PID, 0, "name", "python")
    add_metadata("process_labels", CPU_PID, 0, "labels", "CPU")
    add_metadata("process_sort_index", CPU_PID, 0, "sort_index", CPU_PID)
    for pid in used_gpu_pids:
        add_metadata("process_name", pid, 0, "name", "python")
        add_metadata("process_labels", pid, 0, "labels", f"GPU {pid - GPU_PID_BASE}")
        add_metadata("process_sort_index", pid, 0, "sort_index", 5_000_000 + pid)
    for (pid, thread_id), name in sorted(lane_meta.items()):
        add_metadata("thread_name", pid, thread_id, "name", name)
        add_metadata("thread_sort_index", pid, thread_id, "sort_index", thread_id)

    end_ts = max(
        (event["ts"] + event.get("dur", 0) for event in trace_events if event.get("ph") == "X" and "ts" in event),
        default=0.0,
    )
    trace_events = metadata + trace_events
    trace_events.append(
        {
            "ph": "i",
            "s": "g",
            "name": "Record Window End",
            "pid": "",
            "tid": "",
            "ts": end_ts + 1.0,
        }
    )
    document = _build_document(event_list, trace_events, torch_compat, path)
    with _open_trace(path) as file:
        json.dump(document, file)
    return os.fspath(path)


def _load_trace(path):
    if os.fspath(path).endswith(".gz"):
        with gzip.open(path, "rt") as file:
            return json.load(file)
    with open(path) as file:
        return json.load(file)


def merge_distributed_traces(paths, out_path):
    """Merge per-process trace files into one trace document."""
    merged = {
        "traceEvents": [],
        "displayTimeUnit": "ns",
        "distributedInfo": {"merged_ranks": []},
    }
    for path in paths:
        document = _load_trace(path)
        rank = document.get("distributedInfo", {}).get("rank")
        if rank is None:
            rank = len(merged["distributedInfo"]["merged_ranks"])
        merged["distributedInfo"]["merged_ranks"].append(rank)
        pid_base = rank * 10_000_000
        for event in document.get("traceEvents", []):
            copied = dict(event)
            if "pid" in copied:
                try:
                    copied["pid"] = pid_base + int(copied["pid"])
                except (TypeError, ValueError):
                    pass
            merged["traceEvents"].append(copied)
    merged["traceName"] = os.fspath(out_path)
    with _open_trace(out_path) as file:
        json.dump(merged, file)
    return os.fspath(out_path)


__all__ = ["export_chrome_trace", "merge_distributed_traces"]
