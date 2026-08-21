import gc
import threading

import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


def test_stream_context_restores_state_and_is_thread_local():
    default = tp.cuda.default_stream()
    assert tp.cuda.current_stream() == default

    selected = tp.cuda.Stream()
    before = tp.cuda.current_stream()
    with tp.cuda.stream(selected):
        assert tp.cuda.current_stream() == selected
    assert tp.cuda.current_stream() == before

    observed = []

    def worker():
        observed.append(tp.cuda.current_stream().cuda_stream)
        worker_stream = tp.cuda.Stream()
        tp.cuda.set_stream(worker_stream)
        observed.append(tp.cuda.current_stream() == worker_stream)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert observed == [0, True]
    assert tp.cuda.current_stream() == before


def test_real_event_timing_and_stream_wait():
    producer = tp.cuda.Stream()
    consumer = tp.cuda.Stream()
    start = tp.cuda.Event(enable_timing=True)
    end = tp.cuda.Event(enable_timing=True)

    with tp.cuda.stream(producer):
        start.record()
        tp.cuda._sleep(100_000_000)
        end.record()

    assert not end.query()
    consumer.wait_event(end)
    consumer.synchronize()
    assert end.query()
    assert start.elapsed_time(end) > 0.0


def test_stream_aware_allocator_prevents_early_reuse():
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
    tp.cuda.reset_peak_memory_stats()
    # Global allocator stats may retain blocks from earlier failing tests
    # (pytest keeps their tracebacks, and frames keep tensors alive), so all
    # assertions below are expressed as deltas of this baseline.
    baseline_allocated = tp.cuda.memory_allocated()
    baseline_reserved = tp.cuda.memory_reserved()

    # Warm enough allocator blocks before queuing work. A cold cudaMalloc may
    # synchronize the device, which would hide the cross-stream lifetime this
    # test is intended to exercise.
    warm = [tp.empty((1024, 1024), device="cuda") for _ in range(3)]
    warm_result = warm[0] + warm[1]
    tp.cuda.synchronize()
    del warm_result, warm
    gc.collect()
    tp.cuda.synchronize()

    producer = tp.cuda.Stream()
    consumer = tp.cuda.Stream()
    finished = tp.cuda.Event()
    with tp.cuda.stream(producer):
        source = tp.ones((1024, 1024), device="cuda")
        source_ptr = source.data_ptr()
        tp.cuda._sleep(500_000_000)
        result = source + source
        finished.record()
        del source

    # Pointer comparison is meaningful only while producer work is known to
    # remain outstanding; avoid a timing-dependent false failure on fast GPUs.
    assert not finished.query()

    with tp.cuda.stream(consumer):
        replacement = tp.empty((1024, 1024), device="cuda")
        replacement_ptr = replacement.data_ptr()

    # The source block still participates in queued producer work, so the
    # caching allocator must not hand it to an unrelated stream yet.
    assert replacement_ptr != source_ptr
    producer.synchronize()
    assert result[0, 0].item() == 2.0

    assert tp.cuda.memory_allocated() > baseline_allocated
    assert tp.cuda.memory_reserved() >= tp.cuda.memory_allocated()
    del replacement, result
    gc.collect()
    tp.cuda.synchronize()
    assert tp.cuda.memory_allocated() == baseline_allocated
    assert tp.cuda.memory_reserved() > baseline_reserved
    tp.cuda.empty_cache()
    assert tp.cuda.memory_reserved() == baseline_reserved


def test_same_stream_allocator_reuses_ordered_block_without_event_fence():
    tp.cuda.synchronize()
    tp.cuda.empty_cache()
    stream = tp.cuda.Stream()
    finished = tp.cuda.Event()

    with tp.cuda.stream(stream):
        source = tp.ones((1024, 1024), device="cuda")
        source_ptr = source.data_ptr()
        tp.cuda._sleep(500_000_000)
        result = source + source
        finished.record()
        del source
        replacement = tp.empty((1024, 1024), device="cuda")
        replacement_ptr = replacement.data_ptr()

    assert not finished.query()
    assert replacement_ptr == source_ptr
    stream.synchronize()
    assert result[0, 0].item() == 2.0


def test_pinned_memory_nonblocking_copy_keeps_host_storage_alive():
    tp.cuda.synchronize()

    # Queue the H2D copy behind visible work. Dropping the source immediately
    # must leave its pinned allocation pending instead of returning it to the
    # host cache while DMA can still read it.
    source = tp.ones((1024 * 1024,), pin_memory=True)
    assert source.device.is_cpu()
    assert source.is_pinned()
    source_ptr = source.data_ptr()
    stream = tp.cuda.Stream()
    finished = tp.cuda.Event()
    with tp.cuda.stream(stream):
        tp.cuda._sleep(500_000_000)
        result = source.to("cuda", non_blocking=True)
        finished.record()
    assert not finished.query()

    del source
    assert not finished.query()
    replacement = tp.empty((1024 * 1024,), pin_memory=True)
    assert replacement.data_ptr() != source_ptr

    stream.synchronize()
    assert result[0].item() == 1.0


def test_nonblocking_device_to_pinned_host_copy_and_record_stream():
    tp.cuda.synchronize()
    source = tp.full((1024 * 1024,), 3.0, device="cuda")
    destination = tp.empty((1024 * 1024,), pin_memory=True)
    stream = tp.cuda.Stream()
    finished = tp.cuda.Event()
    with tp.cuda.stream(stream):
        tp.cuda._sleep(500_000_000)
        destination.copy_(source, non_blocking=True)
        finished.record()
    assert not finished.query()
    stream.synchronize()
    assert destination[0].item() == 3.0

    # record_stream accepts the public Stream wrapper and prevents the CUDA
    # allocator from reusing a block before that stream reaches the record.
    guarded = tp.empty((1024 * 1024,), device="cuda")
    guarded_ptr = guarded.data_ptr()
    guarded_finished = tp.cuda.Event()
    with tp.cuda.stream(stream):
        tp.cuda._sleep(500_000_000)
        guarded_finished.record()
    guarded.record_stream(stream)
    del guarded
    assert not guarded_finished.query()
    replacement = tp.empty((1024 * 1024,), device="cuda")
    assert replacement.data_ptr() != guarded_ptr
    stream.synchronize()


def test_cuda_allocation_tracks_requested_device():
    requested = tp.cuda.current_device()
    tensor = tp.empty((8,), device=f"cuda:{requested}")
    assert tensor.device.index == requested


@pytest.mark.skipif(tp.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_non_current_device_dispatch_restores_device():
    original = tp.cuda.current_device()
    other = 1 if original == 0 else 0
    tensor = tp.ones((16,), device=f"cuda:{other}")
    result = tensor + 1
    assert result.device.index == other
    assert result[0].item() == 2.0
    assert tp.cuda.current_device() == original
