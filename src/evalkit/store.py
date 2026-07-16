"""Scorecard/comparison persistence and the results-store export seam.

Scorecards and comparisons serialize to JSON (CI artifacts, local files). For
trend tracking you can store them in a column store such as ClickHouse, keyed
by ``project``/``suite``; rather than couple the engine to any particular
driver, this module flattens a scorecard into a stable row shape and writes it
to a JSONL **outbox** a separate shipper drains. Swap ``JsonlOutboxExporter``
for a real database client without touching the runner or any consumer.

The emitted rows follow a no-``Nullable`` convention that maps cleanly onto a
column store: a missing metric value is the sentinel pair
``(value=0, has_value=false)`` and ``passed`` is the tri-state string
``'true'|'false'|'null'``, so a JSONEachRow-style feed maps straight onto a
flat ``eval_runs``/``eval_scores`` schema.
"""

import json
import pathlib

from evalkit import models


def write_scorecard(
    path: str | pathlib.Path, scorecard: models.Scorecard
) -> None:
    """Write a scorecard as pretty JSON."""
    pathlib.Path(path).write_text(
        scorecard.model_dump_json(indent=2), encoding='utf-8'
    )


def read_scorecard(path: str | pathlib.Path) -> models.Scorecard:
    """Read a scorecard previously written by :func:`write_scorecard`."""
    return models.Scorecard.model_validate_json(
        pathlib.Path(path).read_text(encoding='utf-8')
    )


def load_scorecard(path: str | pathlib.Path) -> models.Scorecard:
    """Read a scorecard from either a scorecard file or a full-run file.

    Accepts both ``run --out`` (a bare Scorecard) and ``run --run-out`` (a
    RunResult wrapping one), so a comparison can consume whichever artifact
    a prior run left behind.
    """
    data = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'scorecard' in data and 'results' in data:
        return models.Scorecard.model_validate(data['scorecard'])
    return models.Scorecard.model_validate(data)


def write_run(path: str | pathlib.Path, run: models.RunResult) -> None:
    """Write a full run (scorecard + every per-sample result) as JSON.

    This is the persisted ground truth behind a scorecard: transcript
    review, human rating, and judge-agreement analysis all read it back
    rather than re-running the suite.
    """
    pathlib.Path(path).write_text(
        run.model_dump_json(indent=2), encoding='utf-8'
    )


def read_run(path: str | pathlib.Path) -> models.RunResult:
    """Read a run previously written by :func:`write_run`."""
    return models.RunResult.model_validate_json(
        pathlib.Path(path).read_text(encoding='utf-8')
    )


# -- run checkpointing (idempotent resume) ----------------------------------
#
# A checkpoint is a JSONL file: line 1 is a meta header (run_id + the content
# hashes that identify *which* eval it belongs to), and each subsequent line is
# one completed ``CaseResult``. The runner appends a line as each (case,
# sample) finishes, so an interrupted run leaves a valid partial file that a
# ``--resume`` re-run reads back to skip work already done.


def init_checkpoint(path: str | pathlib.Path, meta: dict) -> None:
    """Start a fresh checkpoint file with its meta header line (truncating)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(meta) + '\n')


def checkpoint_meta(path: str | pathlib.Path) -> dict | None:
    """The checkpoint meta header, or ``None`` if the file is absent/empty."""
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    with path.open(encoding='utf-8') as handle:
        first = handle.readline()
    return json.loads(first) if first.strip() else None


def append_checkpoint_result(
    path: str | pathlib.Path, result: models.CaseResult
) -> None:
    """Append one completed per-sample result to the checkpoint file."""
    with pathlib.Path(path).open('a', encoding='utf-8') as handle:
        handle.write(result.model_dump_json() + '\n')


def read_checkpoint_results(
    path: str | pathlib.Path,
) -> list[models.CaseResult]:
    """The completed results recorded in a checkpoint (after the meta line)."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    return [
        models.CaseResult.model_validate_json(line)
        for line in lines[1:]
        if line.strip()
    ]


def write_comparison(
    path: str | pathlib.Path, comparison: models.Comparison
) -> None:
    """Write a comparison as pretty JSON."""
    pathlib.Path(path).write_text(
        comparison.model_dump_json(indent=2), encoding='utf-8'
    )


def append_rating(path: str | pathlib.Path, rating: models.Rating) -> None:
    """Append one human rating to a JSONL ratings file (creating it)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(rating.model_dump_json() + '\n')


def read_ratings(path: str | pathlib.Path) -> list[models.Rating]:
    """Read a JSONL ratings file (the open human-rating interchange format).

    Rows may come from the ``rate`` web app or any external tool that emits
    the same shape - one JSON object per line.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    return [
        models.Rating.model_validate_json(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def append_preference(
    path: str | pathlib.Path, preference: models.Preference
) -> None:
    """Append one side-by-side preference to a JSONL file (creating it)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(preference.model_dump_json() + '\n')


def read_preferences(path: str | pathlib.Path) -> list[models.Preference]:
    """Read a JSONL preferences file (the open A-vs-B interchange format).

    Rows may come from the ``rank`` web app or any external tool that emits
    the same shape - one JSON object per line.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    return [
        models.Preference.model_validate_json(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def scorecard_rows(scorecard: models.Scorecard) -> list[dict]:
    """Flatten a scorecard into one row per metric (the store schema).

    Matches the ``eval_runs``/``eval_scores`` columns in the platform design:
    every row carries the full reproducibility key so rows are self-describing
    in a multi-tenant table.
    """
    key = _run_key(scorecard)
    return [
        {
            **key,
            'metric': metric.metric,
            # No-Nullable store convention: a missing value is the sentinel
            # pair (0, has_value=false), not null (docs/clickhouse-schema.sql).
            'value': metric.value if metric.value is not None else 0.0,
            'has_value': metric.value is not None,
            'stdev': metric.stdev if metric.stdev is not None else 0.0,
            'has_stdev': metric.stdev is not None,
            'metric_kind': metric.kind,
            'n': metric.n,
        }
        for metric in scorecard.metrics.values()
    ]


def _run_key(scorecard: models.Scorecard) -> dict:
    """The reproducibility key carried on every store row."""
    return {
        'run_id': scorecard.run_id,
        'project': scorecard.project,
        'suite': scorecard.suite,
        'variant': scorecard.variant.name,
        'dataset_version': scorecard.dataset_version,
        'model_id': scorecard.model_id,
        'prompt_version': scorecard.prompt_version,
        'judge_version': scorecard.judge_version,
        'revision': scorecard.revision,
        'suite_hash': scorecard.suite_hash,
        'dataset_hash': scorecard.dataset_hash,
        'mode': scorecard.mode,
        'created_at': scorecard.created_at,
    }


def score_rows(run: models.RunResult) -> list[dict]:
    """Flatten a run into one row per per-case score (``eval_scores``).

    The per-sample companion to :func:`scorecard_rows`: every individual
    grader score, keyed by ``run_id``/``case_id``/``sample_idx``, so the
    store keeps the raw observations the scorecard means were computed
    from (needed for variance, drill-down, and judge-vs-human analysis).
    """
    key = _run_key(run.scorecard)
    rows: list[dict] = []
    for result in run.results:
        for score in result.scores:
            rows.append(
                {
                    **key,
                    'case_id': result.case.id,
                    'sample_idx': result.sample_idx,
                    'grader': score.grader,
                    'metric': score.metric,
                    'value': score.value if score.value is not None else 0.0,
                    'has_value': score.value is not None,
                    # Tri-state (Enum8 'true'/'false'/'null') per the store
                    # convention: deterministic graders set passed, judges
                    # leave it null.
                    'passed': (
                        'true'
                        if score.passed is True
                        else 'false'
                        if score.passed is False
                        else 'null'
                    ),
                    'detail': score.detail or '',
                    'errored': result.output.error is not None,
                }
            )
    return rows


class JsonlOutboxExporter:
    """Append scorecard rows to a JSONL outbox for a ClickHouse shipper.

    A no-network stand-in for direct ClickHouse ingestion: real deployments
    point a shipper at this file (or replace this class with a ClickHouse
    client implementing the same ``export`` method).
    """

    def __init__(self, outbox_path: str | pathlib.Path):
        self.outbox_path = pathlib.Path(outbox_path)

    def _append(self, rows: list[dict]) -> int:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open('a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        return len(rows)

    def export(self, scorecard: models.Scorecard) -> int:
        """Append one row per metric; return the number of rows written."""
        return self._append(scorecard_rows(scorecard))

    def export_scores(self, run: models.RunResult) -> int:
        """Append one row per per-case score (the ``eval_scores`` feed)."""
        return self._append(score_rows(run))
