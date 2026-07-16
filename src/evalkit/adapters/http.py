"""HTTP target adapter - POST a templated body, extract response fields.

The request ``body`` is a template whose ``$ref`` strings are resolved against
the case and variant (e.g. ``$input.content``, ``$variant.model``). ``extract``
maps output field names to dotted paths into the JSON response. ``base_url``
and header values support ``${ENV_VAR}`` expansion so secrets/URLs stay out of
the suite file.

``httpx`` is imported lazily so offline/replay runs need no network stack.
"""

import time

from evalkit import models, refs
from evalkit.adapters import base
from evalkit.adapters._env import _expand_env


@base.register('http')
class HTTPAdapter:
    """POST ``body`` to ``base_url + path`` and extract fields from JSON."""

    def __init__(
        self,
        base_url: str,
        path: str,
        body: dict,
        extract: dict[str, str],
        method: str = 'POST',
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.path = path
        self.body_template = body
        self.extract = extract
        self.method = method
        self.headers = headers or {}
        self.timeout = timeout

    async def invoke(
        self, case: models.Case, variant: models.Variant
    ) -> models.Output:
        import httpx

        context = {
            'input': case.input,
            'expected': case.expected or {},
            'variant': variant.knobs,
            'case': case.model_dump(),
        }
        payload = refs.build_value(self.body_template, context)
        url = _expand_env(self.base_url).rstrip('/') + self.path
        # Drop headers whose ${ENV} expanded to empty (e.g. unset auth) so we
        # never send a blank Authorization that a proxy might 400 on.
        headers = {k: v for k, v in _expand_env(self.headers).items() if v}

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    self.method, url, json=payload, headers=headers
                )
        except httpx.HTTPError as exc:
            # Network-level failures (timeouts, connection resets) are
            # transient - worth a retry.
            return models.Output(
                error=f'{type(exc).__name__}: {exc}',
                retryable=True,
                latency_ms=(time.monotonic() - start) * 1000,
            )
        latency_ms = (time.monotonic() - start) * 1000

        if response.status_code >= 400:
            # 429 (rate limit) and 5xx (server) are transient; other 4xx are
            # the request's fault and won't fix on retry.
            code = response.status_code
            return models.Output(
                error=f'HTTP {code}',
                retryable=code == 429 or code >= 500,
                raw=response.text[:2000],
                latency_ms=latency_ms,
            )
        try:
            body = response.json()
        except ValueError:
            return models.Output(
                error='non-JSON response',
                raw=response.text[:2000],
                latency_ms=latency_ms,
            )
        fields = {
            name: refs.resolve_path(body, path)
            for name, path in self.extract.items()
        }
        return models.Output(fields=fields, raw=body, latency_ms=latency_ms)
