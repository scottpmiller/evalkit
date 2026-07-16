"""Report rendering, pluggable by ``type`` like adapters and graders.

Importing this package registers the built-in ``markdown`` and ``html``
reporters. Select one with :func:`build_reporter` (the CLI's ``--report``
flag), and register more with ``@evalkit.reporters.base.register(...)``.
"""

from evalkit.reporters import html, markdown  # noqa: F401 - register built-ins
from evalkit.reporters.base import (
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
