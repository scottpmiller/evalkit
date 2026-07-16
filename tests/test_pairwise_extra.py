"""Extra pairwise coverage: client selection and content mapping."""

import asyncio
import unittest

from evalcore import models, pairwise


class BuildClientTests(unittest.TestCase):
    def test_replay_requires_path(self):
        with self.assertRaises(ValueError):
            pairwise.build_pairwise_client('replay', {})

    def test_live_requires_model(self):
        with self.assertRaises(ValueError):
            pairwise.build_pairwise_client('http', {})

    def test_selects_anthropic_and_openai(self):
        a = pairwise.build_pairwise_client('http', {'model': 'claude-x'})
        self.assertIsInstance(a, pairwise.AnthropicPairwiseClient)
        o = pairwise.build_pairwise_client(
            'http', {'model': 'openai:gpt-4o', 'provider': 'openai'}
        )
        self.assertIsInstance(o, pairwise.OpenAIPairwiseClient)
        self.assertEqual(o.model, 'gpt-4o')

    def test_replay_client_from_pairs_mapping(self):
        # Construction from a {'pairs': [...]} mapping, then a lookup.
        rc = pairwise.ReplayPairwiseClient(
            {'pairs': [{'a': 'x', 'b': 'y', 'winner': 'x'}]}
        )
        pick = asyncio.run(
            rc.compare(system='', user='', first='x', second='y')
        )
        self.assertEqual(pick, 'first')


class ContentMapTests(unittest.TestCase):
    def test_skips_missing_content(self):
        run = models.RunResult(
            run_id='R',
            scorecard=models.Scorecard(
                project='p',
                suite='s',
                variant=models.Variant(name='v'),
                dataset_version='v1',
            ),
            results=[
                models.CaseResult(
                    case=models.Case(id='c1'),
                    variant_name='v',
                    sample_idx=0,
                    output=models.Output(fields={'text': 'hi'}),
                ),
                models.CaseResult(
                    case=models.Case(id='c2'),
                    variant_name='v',
                    sample_idx=0,
                    output=models.Output(fields={}),
                ),  # no content -> skipped
            ],
        )
        mapped = pairwise._content_map(run, 'output.text')
        self.assertEqual(set(mapped), {('c1', 0)})


if __name__ == '__main__':
    unittest.main()
