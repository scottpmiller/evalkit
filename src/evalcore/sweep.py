"""N-way sweeps: run many variants of a suite and rank them.

Where :mod:`compare` answers "candidate vs baseline", a sweep answers
"which of these N variants is best" - a model x prompt-version matrix.
It reuses the per-variant runner unchanged: run every named variant, then
tabulate one leaderboard (metric x variant) and a ranking by the suite's
win metric. Purely a run-orchestration + summary layer; no new seams.
"""

from evalcore import loader, models, runner


async def run_sweep(
    suite: loader.SuiteConfig,
    variant_names: list[str] | None = None,
    *,
    mode: str | None = None,
    grader_mode: str | None = None,
    revision: str | None = None,
    created_at: str | None = None,
) -> list[models.RunResult]:
    """Run each named variant (default: all) and return their runs."""
    names = variant_names or list(suite.variants)
    runs: list[models.RunResult] = []
    for name in names:
        runs.append(
            await runner.run_suite(
                suite,
                name,
                mode=mode,
                grader_mode=grader_mode,
                revision=revision,
                created_at=created_at,
            )
        )
    return runs


def summarize_sweep(
    runs: list[models.RunResult], thresholds: dict | None = None
) -> models.SweepResult:
    """Fold sweep runs into a ranked leaderboard by the win metric."""
    thresholds = thresholds or {}
    win_metric = thresholds.get('win_metric')
    higher = thresholds.get('win_higher_is_better', True)
    scorecards = [run.scorecard for run in runs]

    all_metrics = sorted(
        {metric for card in scorecards for metric in card.metrics}
    )
    matrix: dict[str, dict[str, float | None]] = {}
    for metric in all_metrics:
        matrix[metric] = {}
        for card in scorecards:
            found = card.metrics.get(metric)
            matrix[metric][card.variant.name] = found.value if found else None

    entries: list[models.SweepEntry] = []
    if win_metric:
        pairs = [
            (
                card.variant.name,
                (
                    card.metrics[win_metric].value
                    if win_metric in card.metrics
                    else None
                ),
            )
            for card in scorecards
        ]

        def sort_key(pair: tuple[str, float | None]):
            value = pair[1]
            if value is None:
                return (1, 0.0)  # missing metric sorts last
            return (0, -value if higher else value)

        for rank, (name, value) in enumerate(
            sorted(pairs, key=sort_key), start=1
        ):
            entries.append(
                models.SweepEntry(variant=name, win_value=value, rank=rank)
            )

    first = scorecards[0]
    return models.SweepResult(
        project=first.project,
        suite=first.suite,
        win_metric=win_metric,
        win_higher_is_better=higher,
        entries=entries,
        matrix=matrix,
    )
