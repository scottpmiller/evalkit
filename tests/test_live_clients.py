"""Cover the live judge/pairwise SDK paths with injected fake SDK modules.

The Anthropic/OpenAI SDKs are imported lazily inside the client methods, so
fake modules placed in sys.modules exercise the real request-building and
response-parsing code without the SDKs (or network) being present.
"""

import asyncio
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from evalkit import models, pairwise
from evalkit.graders import judge


def _fake_anthropic():
    """AsyncAnthropic whose forced tool call echoes a preset input dict."""
    holder = {}

    class _Msgs:
        async def create(self, **kw):
            name = kw['tool_choice']['name']
            block = types.SimpleNamespace(
                type='tool_use', name=name, input=holder['input']
            )
            holder['seen'] = kw
            return types.SimpleNamespace(content=[block])

    class _Client:
        def __init__(self, **_):
            self.messages = _Msgs()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    module = types.ModuleType('anthropic')
    module.AsyncAnthropic = _Client
    return module, holder


def _fake_openai(content):
    class _Comp:
        async def create(self, **kw):
            msg = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)]
            )

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=_Comp())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    module = types.ModuleType('openai')
    module.AsyncOpenAI = _Client
    return module


DIMS = [{'key': 'clarity', 'description': 'clear?'}]


class RubricLiveTests(unittest.TestCase):
    def test_panel_http_mode_with_image(self):
        anthropic_mod, holder = _fake_anthropic()
        holder['input'] = {'scores': {'clarity': 4}, 'rationale': 'ok'}
        openai_mod = _fake_openai(json.dumps({'scores': {'clarity': 5}}))
        with tempfile.TemporaryDirectory() as tmp:
            png = pathlib.Path(tmp) / 's.png'
            png.write_bytes(b'\x89PNGx')
            grader = judge.RubricJudge(
                content_ref='output.text',
                dimensions=DIMS,
                image_refs={'shot': 'artifacts.shot'},
                judges=[
                    {
                        'key': 'claude',
                        'provider': 'anthropic',
                        'model': 'claude-x',
                    },
                    {
                        'key': 'gpt',
                        'provider': 'openai',
                        'model': 'openai:gpt-4o',
                    },
                ],
            )
            grader.set_mode('http')
            output = models.Output(
                fields={'text': 'hi'}, artifacts={'shot': str(png)}
            )
            with (
                mock.patch.dict(
                    sys.modules,
                    {'anthropic': anthropic_mod, 'openai': openai_mod},
                ),
                mock.patch.dict(
                    'os.environ',
                    {'ANTHROPIC_API_KEY': 'a', 'OPENAI_API_KEY': 'o'},
                ),
            ):
                scores = asyncio.run(
                    grader.grade(models.Case(id='c1'), output)
                )
        by = {s.metric: s.value for s in scores}
        # panel mean of 4/5 and 5/5 on a 5-scale = (0.8 + 1.0)/2 = 0.9
        self.assertAlmostEqual(by['llm_judge.clarity'], 0.9)
        # the anthropic request carried the forced tool + an image block
        self.assertEqual(holder['seen']['tool_choice']['name'], 'score_rubric')
        self.assertTrue(
            any(
                b.get('type') == 'image'
                for b in holder['seen']['messages'][0]['content']
            )
        )

    def test_openai_non_json_raises(self):
        client = judge.OpenAIJudgeClient('gpt-4o')
        with (
            mock.patch.dict(sys.modules, {'openai': _fake_openai('not json')}),
            mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'o'}),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    client.judge(
                        key='k', system='s', user='u', dimensions=DIMS, scale=5
                    )
                )


class PairwiseLiveTests(unittest.TestCase):
    def test_anthropic_pick(self):
        mod, holder = _fake_anthropic()
        holder['input'] = {'winner': 'first'}
        client = pairwise.AnthropicPairwiseClient('claude-x')
        with (
            mock.patch.dict(sys.modules, {'anthropic': mod}),
            mock.patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'a'}),
        ):
            pick = asyncio.run(
                client.compare(system='s', user='u', first='A', second='B')
            )
        self.assertEqual(pick, 'first')

    def test_openai_pick(self):
        client = pairwise.OpenAIPairwiseClient('openai:gpt-4o')
        with (
            mock.patch.dict(
                sys.modules,
                {'openai': _fake_openai(json.dumps({'winner': 'second'}))},
            ),
            mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'o'}),
        ):
            pick = asyncio.run(
                client.compare(system='s', user='u', first='A', second='B')
            )
        self.assertEqual(pick, 'second')


if __name__ == '__main__':
    unittest.main()
