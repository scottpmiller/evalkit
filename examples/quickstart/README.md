# quickstart — a worked evalkit consumer

A complete, runnable eval for a **support-reply assistant**. It shows every
moving part of a consumer in one place and runs **fully offline** (no network,
no API keys) against recorded fixtures.

The story: a `baseline` prompt (v1) collapses onto one generic reply for every
ticket and misroutes a "billed twice" refund; a `candidate` prompt (v2) writes
a distinct, empathetic reply per ticket and routes every intent correctly. The
gate confirms the candidate is a clean improvement.

## What's here

```
quickstart/
  suite.yaml            adapter + graders + variants + thresholds (the gate policy)
  adapter.py            custom target adapter (an offline stub for a real API)
  graders.py            custom graders: a per-case check + a whole-run check
  datasets/support_reply/v1/cases/*.yaml    4 support tickets with ground-truth intent
  fixtures/support_reply_replay.yaml        recorded adapter outputs (offline runs)
  fixtures/support_reply_judge.yaml         recorded judge scores (offline judge)
  fixtures/support_reply_pairwise.yaml      recorded A-vs-B judgments
  run_eval.py           the same eval through the Python API
  tests/                asserts the whole thing end-to-end
```

## Run it

From the repo root:

```bash
# CLI gate (baseline vs candidate) — offline replay
evalkit --plugins examples.quickstart.graders gate \
    --suite examples/quickstart/suite.yaml --mode replay
# or: just example

# The same eval through the Python API
uv run python -m examples.quickstart.run_eval
# or: just example-api

# Head-to-head A-vs-B win-rate (offline)
evalkit --plugins examples.quickstart.graders pairwise \
    --suite examples/quickstart/suite.yaml --a baseline --b candidate --mode replay

# The example's assertions
uv run python -m unittest examples.quickstart.tests.test_quickstart
# or: just test-example
```

## The two extension seams

- **`adapter.py`** — how to call the system under test. Here it's a
  deterministic stub so the example runs offline; swap it for a class that
  calls your real API/library/CLI and nothing else in the suite changes.
- **`graders.py`** — how outputs are scored. `acknowledges_customer` is a
  per-case check (averaged into a pass-rate); `distinct_reply_rate` is a
  whole-run aggregate (catches mode collapse). Both register on import.

Everything else — the runner, comparison, gate, judge harness, reporters — is
the engine, reused unchanged. See the top-level `README.md` for the full
reference and `docs/design.md` for the design.
