"""Tests for the shared retry policy: backoff, classification, retry loop."""

import asyncio
import unittest
from unittest import mock

from evalkit import retry


class BackoffTests(unittest.TestCase):
    def test_exponential_and_capped_no_jitter(self):
        cfg = retry.RetryConfig(backoff_base=1.0, backoff_max=4.0, jitter=0)
        delays = [retry.backoff_delay(a, cfg) for a in range(1, 6)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 4.0, 4.0])

    def test_jitter_stays_within_band(self):
        cfg = retry.RetryConfig(
            backoff_base=1.0, backoff_max=100.0, jitter=0.1
        )
        for _ in range(50):
            d = retry.backoff_delay(3, cfg)  # nominal 4.0
            self.assertGreaterEqual(d, 4.0 - 0.4)
            self.assertLessEqual(d, 4.0 + 0.4)

    def test_config_defaults_are_a_no_op(self):
        cfg = retry.RetryConfig()
        self.assertEqual(cfg.max_attempts, 1)


class _Status(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class TransientTests(unittest.TestCase):
    def test_status_codes(self):
        self.assertTrue(retry.is_transient_exc(_Status(429)))
        self.assertTrue(retry.is_transient_exc(_Status(503)))
        self.assertFalse(retry.is_transient_exc(_Status(400)))
        self.assertFalse(retry.is_transient_exc(_Status(404)))

    def test_by_exception_name(self):
        for name in (
            'RateLimitError',
            'APITimeoutError',
            'APIConnectionError',
            'InternalServerError',
            'OverloadedError',
        ):
            exc = type(name, (Exception,), {})()
            self.assertTrue(retry.is_transient_exc(exc), name)

    def test_non_transient_errors(self):
        self.assertFalse(retry.is_transient_exc(ValueError('bad')))
        self.assertFalse(retry.is_transient_exc(KeyError('x')))


class CallWithRetryTests(unittest.TestCase):
    def _run(self, thunk, cfg):
        with mock.patch.object(retry.asyncio, 'sleep') as slept:
            result = asyncio.run(retry.call_with_retry(thunk, cfg))
        return result, slept

    def test_retries_transient_then_succeeds(self):
        calls = {'n': 0}

        async def thunk():
            calls['n'] += 1
            if calls['n'] <= 2:
                raise _Status(429)
            return 'ok'

        result, slept = self._run(
            thunk, retry.RetryConfig(max_attempts=5, jitter=0)
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(calls['n'], 3)
        self.assertEqual(slept.call_count, 2)

    def test_non_transient_raises_immediately(self):
        calls = {'n': 0}

        async def thunk():
            calls['n'] += 1
            raise ValueError('bad request')

        with self.assertRaises(ValueError):
            self._run(thunk, retry.RetryConfig(max_attempts=5))
        self.assertEqual(calls['n'], 1)  # not retried

    def test_reraises_after_budget(self):
        calls = {'n': 0}

        async def thunk():
            calls['n'] += 1
            raise _Status(500)

        with self.assertRaises(_Status):
            self._run(thunk, retry.RetryConfig(max_attempts=3, jitter=0))
        self.assertEqual(calls['n'], 3)

    def test_default_config_does_not_retry(self):
        calls = {'n': 0}

        async def thunk():
            calls['n'] += 1
            raise _Status(429)

        with self.assertRaises(_Status):
            self._run(thunk, retry.RetryConfig())
        self.assertEqual(calls['n'], 1)


if __name__ == '__main__':
    unittest.main()
