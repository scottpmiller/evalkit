"""Comparison / regression engine - candidate vs baseline -> gate verdict.

Generic across all consumers. Two ideas:

* **Guardrails** - metrics that must hold regardless of the headline result
  (e.g. ``false_negative_rate`` must stay under a ceiling and must not increase
  vs. baseline). A guardrail breach is a hard ``fail``.
* **Win metric** - the headline signal (``f1`` for ``/validation``; a pairwise
  win-rate for judged suites). A regression beyond the band is a ``warn``.

Verdict: any guardrail breach -> ``fail``; else win regressed -> ``warn``; else
``pass``. The on-regression policy is configurable per suite.
"""

from evalcore import models

_EPS = 1e-9


def _metric(card: models.Scorecard, name: str) -> float | None:
    found = card.metrics.get(name)
    return found.value if found else None


def _check_guardrail(
    rule: dict, baseline: models.Scorecard, candidate: models.Scorecard
) -> models.GuardrailResult:
    metric = rule['metric']
    cand = _metric(candidate, metric)
    base = _metric(baseline, metric)
    if cand is None:
        return models.GuardrailResult(
            metric=metric, passed=False, detail='metric absent on candidate'
        )

    problems: list[str] = []
    if 'max' in rule and cand > rule['max'] + _EPS:
        problems.append(f'{cand:.4f} > max {rule["max"]}')
    if 'min' in rule and cand < rule['min'] - _EPS:
        problems.append(f'{cand:.4f} < min {rule["min"]}')
    if (
        rule.get('must_not_increase')
        and base is not None
        and (cand > base + _EPS)
    ):
        problems.append(f'increased {base:.4f} -> {cand:.4f}')
    if (
        rule.get('must_not_decrease')
        and base is not None
        and (cand < base - _EPS)
    ):
        problems.append(f'decreased {base:.4f} -> {cand:.4f}')

    if problems:
        return models.GuardrailResult(
            metric=metric, passed=False, detail='; '.join(problems)
        )
    return models.GuardrailResult(
        metric=metric, passed=True, detail=f'{cand:.4f} ok'
    )


def _evaluate_win(
    thresholds: dict, baseline: models.Scorecard, candidate: models.Scorecard
) -> tuple[str | None, str]:
    metric = thresholds.get('win_metric')
    if not metric:
        return None, 'neutral'
    base = _metric(baseline, metric)
    cand = _metric(candidate, metric)
    if base is None or cand is None:
        return metric, 'neutral'
    higher_better = thresholds.get('win_higher_is_better', True)
    min_delta = thresholds.get('win_min_delta', 0.0)
    delta = cand - base if higher_better else base - cand
    if delta > min_delta:
        return metric, 'improved'
    if delta < -min_delta:
        return metric, 'regressed'
    return metric, 'neutral'


def compare(
    baseline: models.Scorecard,
    candidate: models.Scorecard,
    thresholds: dict | None = None,
) -> models.Comparison:
    """Compare two scorecards and produce a gate verdict."""
    thresholds = thresholds or {}

    deltas: list[models.MetricDelta] = []
    for metric in sorted(set(baseline.metrics) | set(candidate.metrics)):
        base = _metric(baseline, metric)
        cand = _metric(candidate, metric)
        delta = cand - base if base is not None and cand is not None else None
        deltas.append(
            models.MetricDelta(
                metric=metric, baseline=base, candidate=cand, delta=delta
            )
        )

    guardrails = [
        _check_guardrail(rule, baseline, candidate)
        for rule in thresholds.get('guardrails', [])
    ]
    win_metric, win = _evaluate_win(thresholds, baseline, candidate)

    breached = [g for g in guardrails if not g.passed]
    on_regression = thresholds.get('on_regression', 'warn')
    if breached:
        verdict = 'fail'
    elif win == 'regressed':
        verdict = 'fail' if on_regression == 'fail' else 'warn'
    else:
        verdict = 'pass'

    if breached:
        summary = 'guardrail breach: ' + '; '.join(
            f'{g.metric} ({g.detail})' for g in breached
        )
    elif win_metric:
        summary = f'{win_metric} {win}'
    else:
        summary = 'no win metric configured'

    return models.Comparison(
        project=candidate.project,
        suite=candidate.suite,
        baseline_variant=baseline.variant.name,
        candidate_variant=candidate.variant.name,
        win_metric=win_metric,
        win=win,
        verdict=verdict,
        deltas=deltas,
        guardrails=guardrails,
        summary=summary,
    )
