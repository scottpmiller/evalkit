"""Adapter tests: HTTP (mocked httpx), replay edges, and the registry."""

import asyncio
import unittest
from unittest import mock

import httpx

from evalkit import models
from evalkit.adapters import base, replay
from evalkit.adapters import http as http_adapter


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError('no json')
        return self._payload


class _FakeClient:
    """Stand-in for httpx.AsyncClient as an async context manager."""

    def __init__(self, response=None, exc=None, **_):
        self._response = response
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def request(self, method, url, json=None, headers=None):
        _FakeClient.last = {
            'method': method,
            'url': url,
            'json': json,
            'headers': headers,
        }
        if self._exc:
            raise self._exc
        return self._response


def _invoke(adapter, case, variant):
    return asyncio.run(adapter.invoke(case, variant))


class HttpAdapterTests(unittest.TestCase):
    def _adapter(self, **kw):
        return http_adapter.HTTPAdapter(
            base_url='${BASE}/',
            path='/gen',
            body={'q': '$input.q', 'model': '$variant.model'},
            extract={'text': 'data.text', 'label': 'data.label'},
            headers={
                'Authorization': 'Bearer ${TOK}',
                'X-Empty': '${MISSING}',
            },
            **kw,
        )

    def test_success_extracts_and_expands_env(self):
        resp = _FakeResponse(payload={'data': {'text': 'hi', 'label': 'ok'}})
        with (
            mock.patch.dict('os.environ', {'BASE': 'http://svc', 'TOK': 't'}),
            mock.patch.object(
                httpx, 'AsyncClient', lambda **k: _FakeClient(response=resp)
            ),
        ):
            out = _invoke(
                self._adapter(),
                models.Case(id='c', input={'q': 'yo'}),
                models.Variant(name='v', knobs={'model': 'm'}),
            )
        self.assertEqual(out.fields, {'text': 'hi', 'label': 'ok'})
        self.assertEqual(_FakeClient.last['url'], 'http://svc/gen')
        self.assertEqual(_FakeClient.last['json'], {'q': 'yo', 'model': 'm'})
        # Empty-expanded header dropped; real one kept.
        self.assertIn('Authorization', _FakeClient.last['headers'])
        self.assertNotIn('X-Empty', _FakeClient.last['headers'])
        self.assertIsNotNone(out.latency_ms)

    def test_http_error_status(self):
        resp = _FakeResponse(status_code=500, text='boom')
        with mock.patch.object(
            httpx, 'AsyncClient', lambda **k: _FakeClient(response=resp)
        ):
            out = _invoke(
                self._adapter(), models.Case(id='c'), models.Variant(name='v')
            )
        self.assertEqual(out.error, 'HTTP 500')
        self.assertEqual(out.raw, 'boom')
        self.assertTrue(out.retryable)  # 5xx is transient

    def test_retryable_marking_by_status(self):
        # 429 + 5xx transient; other 4xx terminal.
        for code, expected in [
            (429, True),
            (503, True),
            (400, False),
            (404, False),
        ]:
            resp = _FakeResponse(status_code=code, text='x')
            with mock.patch.object(
                httpx,
                'AsyncClient',
                lambda resp=resp, **k: _FakeClient(response=resp),
            ):
                out = _invoke(
                    self._adapter(),
                    models.Case(id='c'),
                    models.Variant(name='v'),
                )
            self.assertEqual(out.retryable, expected, f'status {code}')

    def test_non_json_response(self):
        resp = _FakeResponse(payload=None, text='<html>')
        with mock.patch.object(
            httpx, 'AsyncClient', lambda **k: _FakeClient(response=resp)
        ):
            out = _invoke(
                self._adapter(), models.Case(id='c'), models.Variant(name='v')
            )
        self.assertEqual(out.error, 'non-JSON response')

    def test_transport_error(self):
        exc = httpx.ConnectError('refused')
        with mock.patch.object(
            httpx, 'AsyncClient', lambda **k: _FakeClient(exc=exc)
        ):
            out = _invoke(
                self._adapter(), models.Case(id='c'), models.Variant(name='v')
            )
        self.assertIn('ConnectError', out.error)
        self.assertTrue(out.retryable)  # network error is transient


class ReplayAdapterTests(unittest.TestCase):
    def _adapter(self):
        return replay.ReplayAdapter(
            {
                'c1': {
                    'baseline': {'text': 'x'},
                    'candidate': {'error': 'gen failed'},
                }
            }
        )

    def test_returns_recorded_fields(self):
        out = _invoke(
            self._adapter(),
            models.Case(id='c1'),
            models.Variant(name='baseline'),
        )
        self.assertEqual(out.fields, {'text': 'x'})

    def test_error_fixture(self):
        out = _invoke(
            self._adapter(),
            models.Case(id='c1'),
            models.Variant(name='candidate'),
        )
        self.assertEqual(out.error, 'gen failed')

    def test_missing_case_and_variant(self):
        out = _invoke(
            self._adapter(),
            models.Case(id='nope'),
            models.Variant(name='baseline'),
        )
        self.assertIn('no fixture for case', out.error)
        out2 = _invoke(
            self._adapter(), models.Case(id='c1'), models.Variant(name='other')
        )
        self.assertIn("'c1'/'other'", out2.error)


class RegistryTests(unittest.TestCase):
    def test_unknown_adapter_type_raises(self):
        with self.assertRaises(ValueError):
            base.build_adapter({'type': 'does-not-exist'})

    def test_duplicate_registration_raises(self):
        with self.assertRaises(ValueError):

            @base.register('replay')  # already taken
            class _Dupe:
                pass


if __name__ == '__main__':
    unittest.main()
