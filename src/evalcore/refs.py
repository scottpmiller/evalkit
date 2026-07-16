"""Reference resolution shared by adapters and graders.

Cases, variants, and outputs are *opaque* to the engine - only adapters and
graders crack them open, and they do so through string references like
``$input.content`` (adapter request bodies) or ``output.verdict`` (grader
field selectors). Keeping this resolution in one place is what lets the core
stay ignorant of any consumer's data shape.
"""

import typing

_MISSING = object()


def resolve_path(obj: typing.Any, path: str) -> typing.Any:
    """Traverse ``obj`` along a dotted ``path``.

    Supports dict keys, list indices (numeric segments), and object
    attributes, in that order of preference. Returns ``None`` when any
    segment is missing rather than raising, so a grader/adapter selector
    that points at an absent field degrades to ``None``.
    """
    current = obj
    for segment in path.split('.'):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment, _MISSING)
        elif (
            isinstance(current, (list, tuple))
            and segment.lstrip('-').isdigit()
        ):
            index = int(segment)
            if -len(current) <= index < len(current):
                current = current[index]
            else:
                current = _MISSING
        else:
            current = getattr(current, segment, _MISSING)
        if current is _MISSING:
            return None
    return current


def resolve_ref(context: dict, ref: str) -> typing.Any:
    """Resolve a ``root.path`` reference against a context dict.

    ``context`` maps root names (``input``, ``expected``, ``variant``,
    ``output``, ``case``) to the corresponding objects.
    """
    root, _, rest = ref.partition('.')
    if root not in context:
        return None
    base = context[root]
    return resolve_path(base, rest) if rest else base


def build_value(template: typing.Any, context: dict) -> typing.Any:
    """Render a (possibly nested) template, substituting ``$ref`` strings.

    A string beginning with ``$`` is treated as a reference (``$input.foo``)
    and replaced with the resolved value. Any other value - including dicts
    and lists, which are walked recursively - passes through literally.
    """
    if isinstance(template, str) and template.startswith('$'):
        return resolve_ref(context, template[1:])
    if isinstance(template, dict):
        return {k: build_value(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [build_value(v, context) for v in template]
    return template
