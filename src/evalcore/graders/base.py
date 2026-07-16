"""Grader protocols and the type registry.

A grader spec is a plain dict from the suite config: ``{type, name, ...}``.
``build_graders`` turns a list of specs into grader instances, split into the
per-case and aggregate buckets the runner needs.
"""

import typing

from evalcore import models


@typing.runtime_checkable
class Grader(typing.Protocol):
    """Per-case grader. Scores are averaged across cases by the runner."""

    name: str

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]: ...


@typing.runtime_checkable
class AggregateGrader(typing.Protocol):
    """Whole-run grader for set-level metrics (P/R/F1, win-rate, ...)."""

    name: str

    def aggregate(
        self, results: list[models.CaseResult]
    ) -> list[models.Score]: ...


_REGISTRY: dict[str, type] = {}


def register(type_name: str) -> typing.Callable[[type], type]:
    """Class decorator registering a grader under a suite-config ``type``."""

    def _decorate(cls: type) -> type:
        if type_name in _REGISTRY:
            raise ValueError(f'grader type {type_name!r} already registered')
        _REGISTRY[type_name] = cls
        return cls

    return _decorate


def build_graders(
    specs: list[dict],
) -> tuple[list[Grader], list[AggregateGrader]]:
    """Instantiate grader specs, partitioned into per-case and aggregate.

    Each spec's ``type`` selects a registered class; remaining keys (minus
    ``type``) are passed as keyword arguments to its constructor.
    """
    per_case: list[Grader] = []
    aggregate: list[AggregateGrader] = []
    for spec in specs:
        spec = dict(spec)
        type_name = spec.pop('type')
        if type_name not in _REGISTRY:
            raise ValueError(
                f'unknown grader type {type_name!r}; '
                f'known: {sorted(_REGISTRY)}'
            )
        grader = _REGISTRY[type_name](**spec)
        if isinstance(grader, AggregateGrader):
            aggregate.append(grader)
        elif isinstance(grader, Grader):
            per_case.append(grader)
        else:  # pragma: no cover - defensive
            raise TypeError(
                f'{type_name!r} is neither Grader nor AggregateGrader'
            )
    return per_case, aggregate
