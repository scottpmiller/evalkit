"""Target adapters - the one seam to the system under test.

An adapter turns a (case, variant) into a normalized ``Output``. The built-in
``http`` adapter (POST a templated body to an endpoint, extract fields from the
JSON response) covers most request/response APIs; ``replay`` returns recorded
outputs so the whole pipeline runs offline and in tests. Consumers register
their own adapters for anything else - e.g. a browser-driven adapter or one
that reads results a deployed system already logged to an observability store.

Importing this package registers the built-in adapter types.
"""

from evalkit.adapters import base, http, replay

__all__ = ['base', 'http', 'replay']
