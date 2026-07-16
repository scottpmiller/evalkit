"""Markdown reporter - the default. Delegates to :mod:`evalcore.report`.

The Markdown renderers live in ``evalcore.report`` (and are re-used directly
by the CLI's sweep/pairwise/agreement output); this class exposes them
through the pluggable reporter seam so ``--report markdown`` and a custom
reporter share one interface.
"""

from evalcore import models, report
from evalcore.reporters import base


@base.register('markdown')
class MarkdownReporter:
    """Render scorecards and comparisons as Markdown sections."""

    def scorecard(self, scorecard: models.Scorecard) -> str:
        return report.render_scorecard(scorecard)

    def comparison(self, comparison: models.Comparison) -> str:
        return report.render_comparison(comparison)

    def agreement(self, result: models.AgreementResult) -> str:
        return report.render_agreement(result)

    def preferences(self, result: models.PreferenceResult) -> str:
        return report.render_preferences(result)

    def pairwise_agreement(self, result: models.PairwiseAgreement) -> str:
        return report.render_pairwise_agreement(result)

    def run(self, run: models.RunResult) -> str:
        """Aggregate scorecard + a per-case matrix and failure notes."""
        lines = [report.render_scorecard(run.scorecard)]
        metrics, rows = base.per_case_matrix(run)
        if rows:
            lines += [
                '',
                '#### Per-case',
                '',
                '| case | ' + ' | '.join(metrics) + ' |',
                '| --- |' + ' ---: |' * len(metrics),
            ]
            notes: list[str] = []
            for row in rows:
                if row['error']:
                    cells = ' | '.join(['err'] * len(metrics))
                    notes.append(f'- `{row["label"]}` errored: {row["error"]}')
                else:
                    cells = ' | '.join(
                        _cell(row['cells'].get(m)) for m in metrics
                    )
                    for metric in metrics:
                        score = row['cells'].get(metric)
                        if score and score.passed is False and score.detail:
                            notes.append(
                                f'- `{row["label"]}` / {metric}: '
                                f'{score.detail}'
                            )
                lines.append(f'| {row["label"]} | {cells} |')
            if notes:
                lines += ['', '**Notes**', '', *notes]
        return '\n'.join(lines)

    def document(self, body: str) -> str:
        # Markdown needs no wrapper; a report is just its section(s).
        return body


def _cell(score: models.Score | None) -> str:
    if score is None or score.value is None:
        return 'n/a'
    text = f'{score.value:.3f}'
    return f'**{text}**' if score.passed is False else text
