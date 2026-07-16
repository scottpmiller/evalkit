"""Report rendering, pluggable by ``type`` like adapters and graders.

Importing this package registers the built-in ``markdown`` and ``html``
reporters. Select one with :func:`build_reporter` (the CLI's ``--report``
flag), and register more with ``@evalcore.reporters.base.register(...)``.
"""

from evalcore.reporters import (  # noqa: F401 - register built-ins
    html,
    markdown,
)
from evalcore.reporters.base import (
    Reporter,
    build_reporter,
    per_case_matrix,
    register,
    render_agreement,
    render_pairwise_agreement,
    render_preferences,
    render_run,
    run_body,
    wrap_document,
)

__all__ = [
    'Reporter',
    'build_reporter',
    'per_case_matrix',
    'register',
    'render_agreement',
    'render_pairwise_agreement',
    'render_preferences',
    'render_run',
    'run_body',
    'wrap_document',
]
