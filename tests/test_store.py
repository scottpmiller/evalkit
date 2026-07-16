"""Store tests: row flattening, the JSONL outbox exporter, and round-trips."""

import json
import pathlib
import tempfile
import unittest

from evalkit import models, store


def _run():
    card = models.Scorecard(
        run_id='R',
        project='p',
        suite='s',
        variant=models.Variant(name='cand'),
        dataset_version='v1',
        revision='sha1',
        mode='replay',
        metrics={
            'f1': models.MetricValue(
                metric='f1', value=0.9, kind='aggregate', n=2
            ),
            'gap': models.MetricValue(
                metric='gap', value=None, kind='mean', n=0
            ),
        },
    )
    result = models.CaseResult(
        case=models.Case(id='c1'),
        variant_name='cand',
        sample_idx=0,
        output=models.Output(fields={'v': 1}, error=None),
        scores=[
            models.Score(
                grader='det',
                metric='passed_check',
                value=1.0,
                passed=True,
                detail='ok',
                case_id='c1',
            ),
            models.Score(
                grader='j',
                metric='quality.overall',
                value=None,
                passed=None,
                case_id='c1',
            ),
        ],
    )
    return models.RunResult(run_id='R', scorecard=card, results=[result])


class RowTests(unittest.TestCase):
    def test_scorecard_rows_sentinel_for_missing_value(self):
        rows = {r['metric']: r for r in store.scorecard_rows(_run().scorecard)}
        self.assertTrue(rows['f1']['has_value'])
        self.assertEqual(rows['f1']['value'], 0.9)
        # Missing value -> (0.0, has_value=false), never null.
        self.assertFalse(rows['gap']['has_value'])
        self.assertEqual(rows['gap']['value'], 0.0)
        self.assertEqual(rows['f1']['revision'], 'sha1')

    def test_score_rows_tristate_passed(self):
        rows = {r['metric']: r for r in store.score_rows(_run())}
        self.assertEqual(rows['passed_check']['passed'], 'true')
        self.assertEqual(rows['quality.overall']['passed'], 'null')
        self.assertEqual(rows['passed_check']['case_id'], 'c1')


class ExporterTests(unittest.TestCase):
    def test_export_scorecard_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'nested' / 'outbox.jsonl'
            exporter = store.JsonlOutboxExporter(path)
            n1 = exporter.export(_run().scorecard)
            n2 = exporter.export_scores(_run())
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), n1 + n2)
            self.assertTrue(all(json.loads(line) for line in lines))


class RoundTripTests(unittest.TestCase):
    def test_scorecard_and_comparison_and_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _run()
            store.write_scorecard(root / 'sc.json', run.scorecard)
            self.assertEqual(
                store.read_scorecard(root / 'sc.json').revision, 'sha1'
            )
            # load_scorecard accepts a full-run file too.
            store.write_run(root / 'run.json', run)
            self.assertEqual(
                store.load_scorecard(root / 'run.json').run_id, 'R'
            )

            comparison = models.Comparison(
                project='p',
                suite='s',
                baseline_variant='b',
                candidate_variant='c',
                summary='ok',
            )
            store.write_comparison(root / 'cmp.json', comparison)
            self.assertIn('"verdict"', (root / 'cmp.json').read_text())

    def test_read_ratings_missing_file_is_empty(self):
        self.assertEqual(store.read_ratings('/no/such/file.jsonl'), [])

    def test_preferences_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp) / 'sub' / 'prefs.jsonl'
            )  # dir auto-created
            store.append_preference(
                path,
                models.Preference(
                    case_id='c1',
                    variant_a='a',
                    variant_b='b',
                    rater='r1',
                    winner='a',
                    dims={'visual': 'tie'},
                ),
            )
            store.append_preference(
                path,
                models.Preference(
                    case_id='c2',
                    variant_a='a',
                    variant_b='b',
                    rater='r1',
                    winner='b',
                ),
            )
            saved = store.read_preferences(path)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0].winner, 'a')
            self.assertEqual(saved[0].dims, {'visual': 'tie'})
            self.assertEqual(saved[1].winner, 'b')

    def test_read_preferences_missing_file_is_empty(self):
        self.assertEqual(store.read_preferences('/no/such/file.jsonl'), [])

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'sub' / 'ck.jsonl'  # dir auto-created
            store.init_checkpoint(path, {'run_id': 'R', 'suite_hash': 'H'})
            self.assertEqual(store.checkpoint_meta(path)['run_id'], 'R')
            self.assertEqual(store.read_checkpoint_results(path), [])
            for cid in ('c1', 'c2'):
                store.append_checkpoint_result(
                    path,
                    models.CaseResult(
                        case=models.Case(id=cid),
                        variant_name='v',
                        sample_idx=0,
                        output=models.Output(fields={'x': cid}),
                    ),
                )
            results = store.read_checkpoint_results(path)
            self.assertEqual([r.case.id for r in results], ['c1', 'c2'])
            # meta survives the appends (still line 1)
            self.assertEqual(store.checkpoint_meta(path)['suite_hash'], 'H')

    def test_checkpoint_meta_absent_file_is_none(self):
        self.assertIsNone(store.checkpoint_meta('/no/such/ck.jsonl'))


if __name__ == '__main__':
    unittest.main()
