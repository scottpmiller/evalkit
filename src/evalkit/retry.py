"""Shared transient-failure retry policy and backoff.

Two call sites use it with different failure signals but one timing schedule:

* the **runner** retries an adapter ``Output`` flagged ``retryable`` (a
  value-signalled failure - the adapter contract never raises);
* the **LLM judge** retries a client call that *raised* a transient SDK error
  (a 429, a 5xx, a timeout).

The exponential-backoff-with-jitter schedule is common, so it lives here; each
site keeps its own loop and predicate. Defaults are a no-op
(``max_attempts: 1``), so opting in is per suite.
"""

import asyncio
import random

import pydantic


class RetryConfig(pydantic.BaseModel):
    """Transient-failure retry policy for a run.

    Defaults are a no-op (``max_attempts: 1``) so existing suites are
    unchanged; opt in per suite. Backoff before attempt *k+1* is
    ``backoff_base * 2**(k-1)`` seconds, capped at ``backoff_max``, with
    +/- ``jitter`` fractional randomization to avoid a thundering herd.
    """

    max_attempts: int = 1
    backoff_base: float = 0.5
    backoff_max: float = 30.0
    jitter: float = 0.1


def backoff_delay(attempt: int, cfg: RetryConfig) -> float:
    """Backoff (seconds) before retrying after ``attempt`` (1-based) failed."""
    delay = min(cfg.backoff_max, cfg.backoff_base * 2 ** (attempt - 1))
    if cfg.jitter:
        # +/- jitter fraction; non-crypto, only spreads retry timing.
        delay += delay * cfg.jitter * (random.random() * 2 - 1)  # noqa: S311
    return max(0.0, delay)


def is_transient_exc(exc: BaseException) -> bool:
    """Whether an exception looks transient (worth retrying).

    SDK-agnostic: an Anthropic/OpenAI ``RateLimitError`` / ``APITimeoutError``
    / ``APIConnectionError`` / ``InternalServerError`` (and 429/5xx status
    carriers) all match, without importing either SDK. A non-transient error
    (bad request, auth, a programming bug) does not, so it re-raises at once.
    """
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(exc, 'status', None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    name = type(exc).__name__
    markers = (
        'RateLimit',
        'Timeout',
        'Connection',
        'InternalServer',
        'ServiceUnavailable',
        'Overloaded',
    )
    return any(marker in name for marker in markers)


async def call_with_retry(thunk, cfg: RetryConfig):
    """Await ``thunk()``, retrying transient exceptions with backoff.

    Re-raises immediately for a non-transient exception, and re-raises the
    last one once the attempt budget is spent - so a caller's existing
    error handling still sees a genuine, sustained failure.
    """
    attempt = 1
    while True:
        try:
            return await thunk()
        except Exception as exc:
            if not is_transient_exc(exc) or attempt >= cfg.max_attempts:
                raise
            await asyncio.sleep(backoff_delay(attempt, cfg))
            attempt += 1
