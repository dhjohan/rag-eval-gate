from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.providers.rate_limit import (
    DailyQuota,
    ProviderLimiter,
    TokenBucket,
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
