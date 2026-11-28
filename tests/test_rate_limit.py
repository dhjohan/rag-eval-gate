from __future__ import annotations

import random
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.providers.rate_limit import (
    DailyQuota,
    ProviderLimiter,
    TokenBucket,
    TokenBucketTPM,
    compute_backoff,
    parse_retry_after,
)


class FakeClock:
    """Clock whose ``sleep`` advances time, so tests never actually wait."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_bucket_allows_a_full_burst_then_throttles() -> None:
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=2, clock=clock, sleep=clock.sleep)

    assert bucket.acquire() == 0.0
    assert bucket.acquire() == 0.0
    assert bucket.acquire() == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)


def test_bucket_refills_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    clock.now += 1.0
    assert bucket.acquire() == 0.0


def test_bucket_capacity_defaults_to_one_minute_of_traffic() -> None:
    bucket = TokenBucket(120, clock=FakeClock(), sleep=lambda _: None)
    assert bucket.capacity == 120.0


def test_bucket_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(0)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(60, capacity=1).acquire(5)


def test_tpm_bucket_carries_oversized_batches_as_debt() -> None:
    clock = FakeClock()
    bucket = TokenBucketTPM(60000, clock=clock, sleep=clock.sleep)  # burst 7500
    # Two 9000-token batches: 7500 burst + 10500 paced at 1000/s = 10.5s.
    assert bucket.acquire(9000) == pytest.approx(1.5)
    assert bucket.acquire(9000) == pytest.approx(9.0)
    assert clock.now == pytest.approx(10.5)


def test_tpm_bucket_caps_a_single_wait_at_62_seconds() -> None:
    clock = FakeClock()
    bucket = TokenBucketTPM(60000, clock=clock, sleep=clock.sleep)  # burst 7500
    # 100k tokens against a 7.5k burst at 1000/s would need 92.5s, capped at 62s.
    assert bucket.acquire(100_000) == pytest.approx(62.0)
    assert clock.now == pytest.approx(62.0)
    # The unpaid remainder stays on the bucket as negative debt.
    assert bucket.tokens < 0


class _SlotClock:
    """Fake clock whose ``sleep`` never advances time.

    Every waiter therefore reserves its slot at ``now == 0``; the barrier
    makes sure all waiters have reserved before any of them "wakes", so the
    test observes the true concurrent behaviour deterministically.
    """

    def __init__(self, parties: int) -> None:
        self.now = 0.0
        self.barrier = threading.Barrier(parties)
        self.durations: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.durations.append(seconds)
        self.barrier.wait()


def _run_concurrently(bucket_wait: Callable[[], float], parties: int = 2) -> list[float]:
    results: list[float] = []

    def waiter() -> None:
        results.append(bucket_wait())

    threads = [threading.Thread(target=waiter) for _ in range(parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_bucket_gives_concurrent_waiters_non_overlapping_slots() -> None:
    clock = _SlotClock(parties=2)
    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=clock.sleep)
    assert bucket.acquire() == 0.0  # drain the single token

    results = _run_concurrently(bucket.acquire)

    # Both waiters needed 1s of tokens, but the second one queued behind the
    # first's absolute slot (t=1s) instead of sleeping the same 1s and waking
    # at the same time: its slot ends at t=2s, i.e. it slept 2s from t=0.
    assert sorted(results) == [pytest.approx(1.0), pytest.approx(2.0)]
    assert sorted(clock.durations) == [pytest.approx(1.0), pytest.approx(2.0)]


def test_tpm_bucket_gives_concurrent_waiters_non_overlapping_slots() -> None:
    clock = _SlotClock(parties=2)
    bucket = TokenBucketTPM(60000, clock=clock, sleep=clock.sleep)  # burst 7500, 1000/s

    results = _run_concurrently(lambda: bucket.acquire(9000))

    # First waiter: (9000 - 7500) / 1000 = 1.5s, slot ends at t=1.5s. Second
    # waiter queues behind it: slot ends at 1.5 + 9.0 = 10.5s, slept from t=0.
    assert sorted(results) == [pytest.approx(1.5), pytest.approx(10.5)]
    assert sorted(clock.durations) == [pytest.approx(1.5), pytest.approx(10.5)]


def test_release_refunds_the_day_that_was_charged(cache: SqliteCache) -> None:
    now = [datetime(2026, 11, 23, 23, 59, 59, tzinfo=UTC)]
    limiter = ProviderLimiter(
        "p", cache, rpd=5, now=lambda: now[0], sleep=lambda _: None
    )
    outcome = limiter.acquire()
    assert cache.get_quota_used("p", "2026-11-23") == 1

    now[0] = datetime(2026, 11, 24, 0, 0, 1, tzinfo=UTC)
    limiter.release(outcome.window)

    assert cache.get_quota_used("p", "2026-11-23") == 0
    assert cache.get_quota_used("p", "2026-11-24") == 0


def test_backoff_grows_then_caps() -> None:
    assert compute_backoff(0, base_s=1.0, cap_s=30.0, jitter=0.0) == 1.0
    assert compute_backoff(1, base_s=1.0, cap_s=30.0, jitter=0.0) == 2.0
    assert compute_backoff(3, base_s=1.0, cap_s=30.0, jitter=0.0) == 8.0
    assert compute_backoff(10, base_s=1.0, cap_s=30.0, jitter=0.0) == 30.0


def test_backoff_with_jitter_stays_within_bounds() -> None:
    rng = random.Random(7)
    for attempt in range(5):
        raw = compute_backoff(attempt, base_s=1.0, cap_s=30.0, jitter=0.0)
        jittered = compute_backoff(attempt, base_s=1.0, cap_s=30.0, jitter=0.5, rng=rng)
        assert raw * 0.5 <= jittered <= raw


def test_backoff_rejects_negative_attempt() -> None:
    with pytest.raises(ValueError, match="attempt"):
        compute_backoff(-1)


def test_parse_retry_after_seconds() -> None:
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 0 ") == 0.0
    assert parse_retry_after("-3") == 0.0


def test_parse_retry_after_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=60)
    parsed = parse_retry_after(format_datetime(when, usegmt=True))
    assert parsed is not None
    assert 50 <= parsed <= 70


def test_parse_retry_after_garbage() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None


def test_daily_quota_uses_provider_timezone(cache: SqliteCache) -> None:
    # 07:00 UTC on the 22nd is still 23:00 on the 21st in Los Angeles.
    fixed = datetime(2026, 11, 22, 7, 0, tzinfo=UTC)
    quota = DailyQuota(
        "gemini_lite",
        cache,
        rpd=10,
        timezone="America/Los_Angeles",
        now=lambda: fixed,
    )
    assert quota.window_date() == "2026-11-21"


def test_daily_quota_remaining_tracks_usage(cache: SqliteCache) -> None:
    fixed = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
    quota = DailyQuota("p", cache, rpd=3, now=lambda: fixed)
    assert quota.remaining() == 3
    assert quota.try_consume() is True
    assert quota.remaining() == 2
    quota.release()
    assert quota.remaining() == 3


def test_limiter_denies_when_daily_quota_is_exhausted(cache: SqliteCache) -> None:
    limiter = ProviderLimiter(
        "p",
        cache,
        rpm=600,
        rpd=1,
        clock=FakeClock(),
        sleep=lambda _: None,
    )
    assert limiter.acquire().allowed is True
    denied = limiter.acquire()
    assert denied.allowed is False
    assert denied.reason == "daily_quota_exhausted"


def test_bucket_does_not_double_credit_after_wait_on_advancing_clock() -> None:
    # Simulate a real clock that keeps advancing during sleep. Without the fix,
    # the second acquire would refill a full token immediately because _last was
    # never advanced past the original time.
    clock = FakeClock()
    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=clock.sleep)

    # Drain the bucket; acquire sleeps 1.0s, advancing the clock to t=1.0.
    assert bucket.acquire() == 0.0
    assert bucket.acquire() == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)

    # Immediately after the wait, the bucket should still be empty: the 1.0s of
    # refill was already credited during the wait, not granted again on refill.
    assert bucket.acquire() == pytest.approx(1.0)


def test_bucket_guard_against_backward_clock() -> None:
    calls: list[float] = []

    def clock() -> float:
        calls.append(calls[-1] if calls else 0.0)
        return calls[-1]

    def sleep(seconds: float) -> None:
        calls.append(calls[-1] + seconds)

    bucket = TokenBucket(60, capacity=1, clock=clock, sleep=sleep)
    assert bucket.acquire() == 0.0
    # Clock does not advance during sleep, so the bucket is empty again.
    assert bucket.acquire() == pytest.approx(1.0)
    # _last was advanced by the wait, so a non-advancing clock does not
    # spuriously refill on the next call.
    assert bucket.acquire() == pytest.approx(1.0)


def test_compute_backoff_validates_cap_s() -> None:
    with pytest.raises(ValueError, match="cap_s"):
        compute_backoff(0, base_s=1.0, cap_s=0.0)
    with pytest.raises(ValueError, match="cap_s"):
        compute_backoff(0, base_s=1.0, cap_s=-5.0)


def test_compute_backoff_does_not_overflow_for_large_attempt() -> None:
    assert compute_backoff(1000, base_s=1.0, cap_s=30.0, jitter=0.0) == 30.0


def test_parse_retry_after_rejects_non_finite(cache: SqliteCache) -> None:
    assert parse_retry_after("inf") is None
    assert parse_retry_after("-inf") is None
    assert parse_retry_after("nan") is None


def test_limiter_releases_quota_on_bucket_interrupt(cache: SqliteCache) -> None:
    fixed = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    limiter = ProviderLimiter(
        "p",
        cache,
        rpm=1,
        rpd=2,
        clock=FakeClock(),
        sleep=lambda _: None,
        now=lambda: fixed,
    )
    # Consume the quota and the single bucket token.
    assert limiter.acquire().allowed is True

    # Make the bucket raise during acquire; the quota slot must be returned.
    def boom() -> float:
        raise KeyboardInterrupt("interrupted")

    limiter.bucket.acquire = boom  # type: ignore
    with pytest.raises(KeyboardInterrupt):
        limiter.acquire()

    assert cache.get_quota_used("p", "2026-11-23") == 1
