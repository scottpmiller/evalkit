"""Adapter protocol and registry."""

import typing

from evalkit import models


@typing.runtime_checkable
class TargetAdapter(typing.Protocol):
    """Invoke the system under test for one case under one variant."""

    async def invoke(
        self, case: models.Case, variant: models.Variant
    ) -> models.Output: ...


_REGISTRY: dict[str, type] = {}


def register(type_name: str) -> typing.Callable[[type], type]:
    """Class decorator registering an adapter under a config ``type``."""

    def _decorate(cls: type) -> type:
        if type_name in _REGISTRY:
            raise ValueError(f'adapter type {type_name!r} already registered')
        _REGISTRY[type_name] = cls
        return cls

    return _decorate


def build_adapter(spec: dict) -> TargetAdapter:
    """Instantiate an adapter from a ``{type, ...}`` config spec."""
    spec = dict(spec)
    type_name = spec.pop('type')
    if type_name not in _REGISTRY:
        raise ValueError(
            f'unknown adapter type {type_name!r}; known: {sorted(_REGISTRY)}'
        )
    return _REGISTRY[type_name](**spec)
