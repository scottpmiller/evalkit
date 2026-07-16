"""Extra LLM-judge coverage: env expansion, panel gating, images, errors."""

import asyncio
import base64
import pathlib
import tempfile
import unittest
from unittest import mock

from evalkit import models
from evalkit.graders import judge

DIMS = [{'key': 'clarity', 'description': 'clear?'}]


def _grade(grader, output, case=None):
    return asyncio.run(grader.grade(case or models.Case(id='c1'), output))


class EnvExpandTests(unittest.TestCase):
    def test_expands_and_passes_through(self):
        with mock.patch.dict('os.environ', {'M': 'claude-x'}):
            self.assertEqual(judge._env_expand('${M}'), 'claude-x')
        self.assertEqual(judge._env_expand(None), None)
        self.assertEqual(judge._env_expand('${UNSET}'), '')


class PanelGatingTests(unittest.TestCase):
    def test_env_gated_judge_dropped_in_replay(self):
        # Two judges; the GPT one has no replay_path -> dropped in replay.
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMS,
            judges=[
                {
                    'key': 'claude',
                    'provider': 'anthropic',
                    'replay_path': None,
                },
                {'key': 'gpt', 'provider': 'openai', 'replay_path': None},
            ],
        )
        # Provide replay fixtures to the claude judge only.
        grader.judges[0]['replay_path'] = 'x'
        grader.set_mode('replay')
        active = grader._active_judges()
        self.assertEqual([j['key'] for j in active], ['claude'])
        self.assertFalse(grader.is_panel)

    def test_error_scores_on_client_failure(self):
        class _Boom:
            async def judge(self, **_):
                raise RuntimeError('api down')

        grader = judge.RubricJudge(
            content_ref='output.text', dimensions=DIMS, client=_Boom()
        )
        scores = _grade(grader, models.Output(fields={'text': 'hi'}))
        self.assertTrue(all(s.value is None for s in scores))
        self.assertTrue(any('judge error' in (s.detail or '') for s in scores))

    def test_no_content_yields_error_scores(self):
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMS,
            client=judge.ReplayJudgeClient({}),
        )
        scores = _grade(grader, models.Output(fields={}))
        self.assertTrue(all(s.value is None for s in scores))


class ImageLoadingTests(unittest.TestCase):
    def test_loads_from_path_and_skips_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / 'a.png'
            png.write_bytes(b'\x89PNGdata')
            loaded = judge._load_images([str(png), '/nope/x.png', None])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]['media_type'], 'image/png')
            self.assertEqual(
                base64.b64decode(loaded[0]['data']), b'\x89PNGdata'
            )

    def test_inline_dict_image(self):
        loaded = judge._load_images(
            [{'data': 'AAAA', 'media_type': 'image/gif'}]
        )
        self.assertEqual(loaded[0]['media_type'], 'image/gif')


class ClientConstructionTests(unittest.TestCase):
    def test_replay_without_path_raises(self):
        grader = judge.RubricJudge(
            content_ref='output.text', dimensions=DIMS, replay_path=None
        )
        grader.set_mode('replay')
        with self.assertRaises(ValueError):
            _grade(grader, models.Output(fields={'text': 'hi'}))

    def test_openai_model_id_stripped(self):
        client = judge.OpenAIJudgeClient('openai:gpt-4o')
        self.assertEqual(client.model, 'gpt-4o')


if __name__ == '__main__':
    unittest.main()
