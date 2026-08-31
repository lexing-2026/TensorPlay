import pickle
import threading
import unittest

from tensorplay.futures import Future, collect_all, wait_all


class TestFuture(unittest.TestCase):
    def test_result_and_nonblocking_value(self):
        future = Future[int]()
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            future.value()

        future.set_result(5)
        self.assertTrue(future.done())
        self.assertEqual(future.wait(), 5)
        self.assertEqual(future.value(), 5)

        with self.assertRaisesRegex(RuntimeError, "only be marked completed once"):
            future.set_result(6)

    def test_exception_is_raised(self):
        future = Future[int]()
        future.set_exception(ValueError("failed"))

        with self.assertRaisesRegex(ValueError, "failed"):
            future.wait()
        with self.assertRaisesRegex(ValueError, "failed"):
            future.value()

    def test_exception_object_can_be_a_result(self):
        future = Future[Exception]()
        value = ValueError("ordinary value")
        future.set_result(value)
        self.assertIs(future.wait(), value)

    def test_completer_exception_is_propagated(self):
        future = Future[int]()

        def complete():
            raise ValueError("completion failed")

        future._completer = complete
        with self.assertRaisesRegex(ValueError, "completion failed"):
            future.wait()
        self.assertTrue(future.done())

    def test_non_exception_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "not an Exception"):
            Future().set_exception("not an error")

    def test_devices_require_explicit_indices(self):
        with self.assertRaisesRegex(ValueError, "indices"):
            Future(devices=["cpu"])
        self.assertIsInstance(Future(devices=["cpu:0"]), Future)

    def test_native_unwrap_function(self):
        future = Future()
        value = ValueError("unwrapped")
        future.set_result(value)

        def unwrap(result):
            raise result

        future._set_unwrap_func(unwrap)
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            future.wait()

    def test_wait_from_another_thread(self):
        future = Future[int]()
        started = threading.Event()
        values = []

        def waiter():
            started.set()
            values.append(future.wait())

        thread = threading.Thread(target=waiter)
        thread.start()
        self.assertTrue(started.wait(1))
        self.assertFalse(future.done())
        future.set_result(9)
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(values, [9])

    def test_then_and_callback_order(self):
        future = Future[int]()
        order = []

        def first(item):
            order.append(1)
            item.add_done_callback(lambda _: order.append(3))

        future.add_done_callback(first)
        chained = future.then(lambda item: item.value() + 1)
        future.add_done_callback(lambda _: order.append(2))
        future.set_result(4)

        self.assertEqual(order, [1, 3, 2])
        self.assertEqual(chained.wait(), 5)

    def test_then_wraps_callback_errors(self):
        future = Future[int]()
        chained = future.then(lambda _: (_ for _ in ()).throw(ValueError("bad")))
        future.set_result(1)

        with self.assertRaisesRegex(RuntimeError, "bad"):
            chained.wait()

    def test_chained_then(self):
        future = Future[int]()
        chained = future
        for _ in range(8):
            chained = chained.then(lambda item: item.wait() + 1)

        future.set_result(1)
        self.assertEqual(chained.wait(), 9)

    def test_callback_errors_do_not_break_completion(self):
        future = Future[int]()

        def fail(_):
            raise ValueError("callback failure")

        future.add_done_callback(fail)
        future.set_result(3)
        self.assertEqual(future.wait(), 3)

    def test_lazy_completion_runs_once(self):
        calls = []
        future = Future[int]()

        def complete():
            calls.append(1)
            future.set_result(7)

        future._completer = complete
        values = []
        threads = [threading.Thread(target=lambda: values.append(future.wait())) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(1)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(calls, [1])
        self.assertEqual(values, [7] * 4)

    def test_collect_all_and_wait_all(self):
        futures = [Future[int]() for _ in range(3)]
        combined = collect_all(futures)
        futures[2].set_result(3)
        futures[0].set_result(1)
        self.assertFalse(combined.done())
        futures[1].set_result(2)

        self.assertIs(combined.wait()[0], futures[0])
        self.assertEqual(wait_all(futures), [1, 2, 3])

    def test_collect_all_propagates_first_input_error(self):
        first = Future[int]()
        second = Future[int]()
        combined = collect_all([first, second])
        second.set_exception(ValueError("second"))
        first.set_exception(ValueError("first"))

        with self.assertRaisesRegex(ValueError, "first"):
            combined.wait()

    def test_empty_and_invalid_collections(self):
        self.assertEqual(collect_all([]).wait(), [])
        self.assertEqual(wait_all([]), [])
        with self.assertRaisesRegex(RuntimeError, "Future can't be None"):
            collect_all([None])

    def test_future_is_not_serializable(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be serialized"):
            pickle.dumps(Future())


if __name__ == "__main__":
    unittest.main()
