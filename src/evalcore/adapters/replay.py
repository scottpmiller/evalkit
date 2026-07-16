"""Replay adapter - return recorded outputs instead of calling a service.

Lets the whole pipeline (runner -> graders -> compare -> report) run offline
in unit tests with no network, no API keys, and no deployed service. The
fixtures file maps ``case_id`` to per-variant recorded outputs::

    obfuscated_eval:
      baseline:  {verdict: ok}          # a recorded miss
      candidate: {verdict: malicious}

Each recorded value becomes ``Output.fields`` for that (case, variant).
"""

from evalcore import loader, models
from evalcore.adapters import base


@base.register('replay')
class ReplayAdapter:
    """Serve recorded ``Output.fields`` keyed by ``case.id`` and variant."""

    def __init__(self, fixtures: str | dict):
        if isinstance(fixtures, str):
            self._fixtures = loader.load_data_file(fixtures)
        else:
            self._fixtures = fixtures

    async def invoke(
        self, case: models.Case, variant: models.Variant
    ) -> models.Output:
        per_case = self._fixtures.get(case.id)
        if per_case is None:
            return models.Output(error=f'no fixture for case {case.id!r}')
        recorded = per_case.get(variant.name)
        if recorded is None:
            return models.Output(
                error=f'no fixture for {case.id!r}/{variant.name!r}'
            )
        if 'error' in recorded:
            return models.Output(error=recorded['error'])
        return models.Output(fields=dict(recorded))
