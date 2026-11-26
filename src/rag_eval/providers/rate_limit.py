"""Per-provider throttling: minute-level token buckets and daily quotas.

Free tiers fail in two different ways, so they are modelled separately:

* **RPM** is bursty and recoverable by waiting, so a token bucket smooths it.
* **RPD** is a hard wall that resets on a provider-specific timezone boundary,
  so it is persisted in SQLite and checked *before* spending a request.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

from rag_eval.cache.sqlite_cache import SqliteCache


@dataclass(frozen=True)
class AcquireOutcome:
    """Result of asking a provider for permission to make one request."""

    allowed: bool
    reason: str | None = None
    waited_s: float = 0.0
    window: str | None = None


class TokenBucket:
    """Classic token bucket, clock-injectable so tests never actually sleep."""

    def __init__(
        self,
        rate_per_minute: int,
        *,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_second = rate_per_minute / 60.0
        # Default burst is one full minute of traffic, never less than one token.
        self.capacity = float(capacity if capacity is not None else max(1, rate_per_minute))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    @property
    def tokens(self) -> float:
        with self._lock:
            return self._tokens

    def _refill_locked(self) -> None:
        now = self._clock()
        if now <= self._last:
            # Time already accounted for (e.g. explicitly credited during a
            # previous wait); avoid double-crediting on a real or advancing clock.
            return
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; return the seconds waited."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.capacity:
            raise ValueError(f"requested {tokens} tokens but capacity is {self.capacity}")
        with self._lock:
            self._refill_locked()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            deficit = tokens - self._tokens
            wait_s = deficit / self.rate_per_second
            # Credit the wait explicitly and advance ``_last`` so the elapsed
            # time during ``sleep`` is not double-counted on the next refill.
            self._tokens = min(self.capacity, self._tokens + wait_s * self.rate_per_second)
            self._tokens -= tokens
            self._last = max(self._clock(), self._last + wait_s)
        self._sleep(wait_s)
        return wait_s


class TokenBucketTPM:
    """Token bucket that paces *token* consumption, not request count.

    Embedding endpoints enforce tokens-per-minute, so a request-count bucket is
    not enough: a few large sub-batches can blow the TPM budget even at low RPM.
    The burst capacity is deliberately small (<= 25k) so back-to-back requests
    cannot dump a full minute's budget at ``t=0``.
    """

    def __init__(
        self,
        tpm: int,
        *,
        max_burst_tokens: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tpm <= 0:
            raise ValueError("tpm must be positive")
        self.rate_per_second = tpm / 60.0
        # Keep the initial burst to ~1/8 of the minute budget (<= 25k), so the
        # first minute can never exceed burst + tpm (~90k for an 80k budget).
        burst_cap = max_burst_tokens if max_burst_tokens is not None else 25_000
        self.capacity = float(max(1, min(burst_cap, tpm // 8)))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    @property
    def tokens(self) -> float:
        with self._lock:
            return self._tokens

    def _refill_locked(self) -> None:
        now = self._clock()
        if now <= self._last:
            return
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; return the seconds waited."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        # Charge the full request even when it exceeds the burst: the balance
        # may go negative, and later requests refill that debt first. Clamping
        # the charge to ``capacity`` let sustained oversized batches exceed TPM.
        request = float(tokens)
        with self._lock:
            self._refill_locked()
            if self._tokens >= request:
                self._tokens -= request
                return 0.0
            deficit = request - self._tokens
            # Cap one wait so a huge batch cannot stall the caller for minutes;
            # any unpaid remainder stays on the bucket as debt.
            wait_s = min(deficit / self.rate_per_second, 62.0)
            self._tokens += wait_s * self.rate_per_second - request
            self._last = max(self._clock(), self._last + wait_s)
        self._sleep(wait_s)
        return wait_s


class DailyQuota:
    """Persisted per-provider daily request budget."""

    def __init__(
        self,
        provider: str,
        cache: SqliteCache,
        *,
        rpd: int | None,
        timezone: str = "UTC",
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.rpd = rpd
        self.tz = ZoneInfo(timezone)
        self._now = now or (lambda: datetime.now(UTC))

    def window_date(self) -> str:
        return self._now().astimezone(self.tz).date().isoformat()

    def try_consume(self, window: str | None = None) -> bool:
        return self.cache.try_consume_quota(
            self.provider, window or self.window_date(), self.rpd
        )

    def release(self, window: str | None = None) -> None:
        self.cache.release_quota(self.provider, window or self.window_date())

    def used(self) -> int:
        return self.cache.get_quota_used(self.provider, self.window_date())

    def remaining(self) -> int | None:
        if self.rpd is None:
            return None
        return max(0, self.rpd - self.used())


class ProviderLimiter:
    """Combines a token bucket with a daily quota for one provider."""

    def __init__(
        self,
        provider: str,
        cache: SqliteCache,
        *,
        rpm: int | None = None,
        rpd: int | None = None,
        timezone: str = "UTC",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self.bucket = TokenBucket(rpm, clock=clock, sleep=sleep) if rpm is not None else None
        self.quota = DailyQuota(provider, cache, rpd=rpd, timezone=timezone, now=now)

    def acquire(self) -> AcquireOutcome:
        # Freeze the window now so a release after midnight refunds the day that
        # was actually charged, not the new one.
        window = self.quota.window_date()
        if not self.quota.try_consume(window):
            return AcquireOutcome(allowed=False, reason="daily_quota_exhausted")
        try:
            waited = self.bucket.acquire() if self.bucket is not None else 0.0
        except BaseException:
            # The quota slot was consumed but the bucket never used it (e.g.
            # interrupted during sleep); return it so the next caller isn't
            # short-changed.
            self.quota.release(window)
            raise
        return AcquireOutcome(allowed=True, waited_s=waited, window=window)

    def release(self, window: str | None = None) -> None:
        """Return a reservation when the request never reached the network."""
        self.quota.release(window)


def compute_backoff(
    attempt: int,
    *,
    base_s: float = 1.0,
    cap_s: float = 30.0,
    jitter: float = 0.5,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with equal-jitter, clamped to ``cap_s``.

    ``attempt`` is zero-based: 0 yields ~``base_s``, 1 yields ~``2 * base_s``.
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    if base_s <= 0:
        raise ValueError("base_s must be positive")
    if cap_s <= 0:
        raise ValueError("cap_s must be positive")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError("jitter must be within [0, 1]")
    # Guard against overflow for very large attempt values.
    raw = cap_s if attempt >= 60 else min(cap_s, base_s * (2.0**attempt))
    if jitter == 0.0:
        return raw
    low = raw * (1.0 - jitter)
    generator = rng or random.Random()
    return low + (raw - low) * generator.random()


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header in either seconds or HTTP-date form."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        seconds = float(cleaned)
    except ValueError:
        try:
            when = parsedate_to_datetime(cleaned)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)
