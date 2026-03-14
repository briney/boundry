"""Tests for boundry._parallel module."""

import pytest

from boundry._parallel import (
    WorkPool,
    _suppress_worker_warnings,
)


def _double(x):
    """Top-level function for pickling in spawn workers."""
    return x * 2


def _raise_keyboard_interrupt(x):
    """Top-level function that raises KeyboardInterrupt."""
    raise KeyboardInterrupt


def _raise_on_third(x):
    """Top-level function that raises on x == 3."""
    if x == 3:
        raise ValueError("boom")
    return x * 2


# ------------------------------------------------------------------
# WorkPool
# ------------------------------------------------------------------


class TestWorkPool:
    """Tests for WorkPool context manager."""

    def test_sequential_when_workers_one(self):
        """WorkPool with max_workers=1 has no active pool."""
        with WorkPool(1) as pool:
            assert not pool.active
            assert pool.max_workers == 1

    def test_map_sequential_fallback(self):
        """map() runs sequentially when pool is not active."""
        with WorkPool(1) as pool:
            results = pool.map(
                lambda x: x * 2, [1, 2, 3]
            )
            assert results == [2, 4, 6]

    def test_active_when_workers_gt_1(self):
        """WorkPool with max_workers>1 creates a pool."""
        with WorkPool(2) as pool:
            assert pool.active

    def test_map_preserves_order(self):
        """map() returns results in input order."""
        with WorkPool(2) as pool:
            results = pool.map(_double, [1, 2, 3, 4])
            assert results == [2, 4, 6, 8]

    def test_pool_cleaned_up_on_exit(self):
        """Pool is shut down after context exit."""
        pool = WorkPool(2)
        pool.__enter__()
        assert pool.active
        pool.__exit__(None, None, None)
        assert not pool.active

    def test_submit_raises_when_no_pool(self):
        """submit() raises when pool is not active."""
        with WorkPool(1) as pool:
            with pytest.raises(RuntimeError, match="not active"):
                pool.submit(lambda: None)

    def test_force_shutdown(self):
        """_force_shutdown() tears down the pool immediately."""
        pool = WorkPool(2)
        pool.__enter__()
        assert pool.active
        pool._force_shutdown()
        assert not pool.active
        assert pool._pool is None

    def test_force_shutdown_noop_when_no_pool(self):
        """_force_shutdown() is safe to call when pool is None."""
        pool = WorkPool(1)
        pool.__enter__()
        assert not pool.active
        pool._force_shutdown()  # should not raise
        assert pool._pool is None

    def test_exit_on_keyboard_interrupt(self):
        """__exit__ calls _force_shutdown on KeyboardInterrupt."""
        pool = WorkPool(2)
        pool.__enter__()
        assert pool.active
        pool.__exit__(KeyboardInterrupt, None, None)
        assert not pool.active

    def test_exit_normal_waits(self):
        """__exit__ with no exception calls shutdown(wait=True)."""
        pool = WorkPool(2)
        pool.__enter__()
        assert pool.active
        pool.__exit__(None, None, None)
        assert not pool.active

    def test_map_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt from a worker task propagates."""
        with WorkPool(1) as pool:
            # Sequential fallback — KeyboardInterrupt propagates directly
            with pytest.raises(KeyboardInterrupt):
                pool.map(_raise_keyboard_interrupt, [1])


# ------------------------------------------------------------------
# on_complete callback
# ------------------------------------------------------------------


class TestWorkPoolOnComplete:
    """Tests for WorkPool.map() on_complete callback."""

    def test_callback_called_per_task_sequential(self):
        """on_complete is called once per task in sequential mode."""
        count = [0]

        def bump():
            count[0] += 1

        with WorkPool(1) as pool:
            results = pool.map(_double, [1, 2, 3], on_complete=bump)
        assert results == [2, 4, 6]
        assert count[0] == 3

    def test_callback_called_per_task_parallel(self):
        """on_complete is called once per task in parallel mode."""
        count = [0]

        def bump():
            count[0] += 1

        with WorkPool(2) as pool:
            results = pool.map(
                _double, [1, 2, 3, 4], on_complete=bump
            )
        assert sorted(results) == [2, 4, 6, 8]
        assert count[0] == 4

    def test_no_callback_is_noop(self):
        """map() works without a callback (default None)."""
        with WorkPool(1) as pool:
            results = pool.map(_double, [5, 6])
        assert results == [10, 12]

    def test_callback_not_called_on_error(self):
        """on_complete is not called for tasks that raise."""
        count = [0]

        def bump():
            count[0] += 1

        with WorkPool(1) as pool:
            with pytest.raises(ValueError):
                pool.map(
                    _raise_on_third,
                    [1, 2, 3],
                    on_complete=bump,
                )
        # First two succeed, third raises before callback
        assert count[0] == 2


# ------------------------------------------------------------------
# Worker warning suppression
# ------------------------------------------------------------------


class TestSuppressWorkerWarnings:
    """Tests for _suppress_worker_warnings."""

    def test_sets_warning_filters(self):
        import warnings

        original_filters = warnings.filters[:]
        try:
            _suppress_worker_warnings()
            # Should have added at least 2 filters
            assert len(warnings.filters) >= len(original_filters) + 2
        finally:
            warnings.filters[:] = original_filters
