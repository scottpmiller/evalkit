"""``${ENV_VAR}`` expansion shared by adapters.

Adapters keep secrets and environment-specific URLs out of the suite file by
writing ``${VAR}`` placeholders that expand from the process environment at
invoke time. Kept here (not in any one adapter) so every adapter can reuse it
without importing a sibling adapter's internals.
"""

import os
import re
import typing

_ENV_RE = re.compile(r'\$\{([A-Z0-9_]+)\}')


def _expand_env(value: typing.Any) -> typing.Any:
    """Expand ``${VAR}`` in strings (recursively through dict/list)."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ''), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value
