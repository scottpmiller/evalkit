"""evalcore - a generic, consumer-agnostic evaluation engine.

The engine knows nothing about any particular system under test. A consumer
supplies four things as data/plug-ins:

1. an **adapter** config (how to call its system + the knobs a variant sets),
2. **datasets** (cases with opaque ``input``/``expected`` blobs),
3. **graders** (generic ones here + any custom ones it registers), and
4. a **suite + thresholds** config.

Everything else - runner, comparison/regression engine, results store, and
reporting - lives here and is reused unchanged across consumers.

See ``docs/evals/platform-design.md`` for the full design.
"""

from evalcore import (
    adapters,
    compare,
    graders,
    loader,
    models,
    refs,
    report,
    runner,
    store,
)

__all__ = [
    'adapters',
    'compare',
    'graders',
    'loader',
    'models',
    'refs',
    'report',
    'runner',
    'store',
]
