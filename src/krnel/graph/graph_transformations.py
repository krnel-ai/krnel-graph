# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

TBaseModel = TypeVar("T", bound=BaseModel)
T = TypeVar("T")
U = TypeVar("U")

"""
A graph is a list of Pydantic model instances, where each model can reference other models (its "dependencies") via its fields.

Models are nodes in a directed acyclic graph (DAG), and the fields of these models specify edge relationships between nodes.

This module provides utilities to traverse and manipulate these graphs, such as finding all parent nodes of a given node based on type filtering.

```
    class A(BaseModel):           Graph structure:
        ...                           ╭──► B ──╮
    class B(BaseModel):           D ──┤        ├──► A
        a: A                          ╰──► C ──╯
    class C(BaseModel):
        a: list[A]                Here, D is the root node
    class D(BaseModel):           with dependencies on B and C,
        a: dict[str, A]           which in turn depend on A.

    a = A()
    b = B(a=a)
    c = C(a=[a])
    d = D(a={"key": a})
```

"""


def get_dependencies(
    *roots: TBaseModel,
    filter_type: type[TBaseModel],
    recursive: bool,
    path: list | None = None,
) -> list[tuple[TBaseModel, list]]:
    """Get the dependencies of a given Pydantic model."""
    results = []
    seen = set()

    def _visit(op: TBaseModel, depth: int = 0, path=None) -> TBaseModel:
        if not recursive and depth > 1:
            return op
        if isinstance(op, filter_type):
            for field in op.__class__.model_fields:
                v = getattr(op, field)
                map_fields(
                    v,
                    filter_type,
                    match_fun=lambda x, path: _visit(
                        x,
                        depth + 1,
                        (path or []) + [field],  # noqa: B023
                    ),
                    path=path,
                )
            if depth > 0:  # Only add dependencies, not the roots themselves
                if op not in seen:
                    seen.add(op)
                    results.append((op, path))
        return op

    for item in roots:
        _visit(item, depth=0, path=path)
    return results


def map_fields(
    val: Any,
    filter_type: type[T],
    match_fun: Callable[[T, list], U],
    unmatch_fun: Callable[[T, list], U] | None = None,
    path: list | None = None,
) -> Any:
    """
    Apply `fun` to all fields of type `filter_type` in the given value.

    Also supports some nested types:
      - list
      - dict

    Note: Not recursive.
    """
    path = path or []
    if isinstance(val, filter_type):
        return match_fun(val, path or [])
    elif isinstance(val, list):
        return [
            map_fields(item, filter_type, match_fun, unmatch_fun, path + [i])
            for i, item in enumerate(val)
        ]
    elif isinstance(val, dict):
        return {
            k: map_fields(v, filter_type, match_fun, unmatch_fun, path + [k])
            for k, v in val.items()
        }

    # other types
    if unmatch_fun is not None:
        return unmatch_fun(val, path)
    return val


def graph_substitute(
    roots: list[T],
    filter_type: type[T],
    substitutions: list[tuple[T, T]],
) -> list[T]:
    """Substitute nodes in the graph with new nodes.

    Args:
        roots: Root nodes of the graph to substitute.
        filter_type: Type of nodes to consider for substitution.
        substitutions: List of ``(old, new)`` node pairs for substitution.

    """
    all_deps = [
        op
        for (op, path) in get_dependencies(
            *roots, filter_type=filter_type, recursive=True
        )
    ]
    for old, _new in substitutions:
        if old not in all_deps:
            raise ValueError(
                f"Supposed to substitute {old!r}, but it is not in the graph dependencies: {all_deps!r}"
            )

    substitutions_dict = dict(substitutions)
    made_substitutions = set()

    def _visit(op: T, path: list) -> T:
        if isinstance(op, filter_type):
            if op in substitutions_dict:
                made_substitutions.add(op)
                return substitutions_dict[op]
            else:
                # reconstruct the model with the same type
                model = dict(op).copy()
                model = {
                    k: map_fields(v, filter_type, _visit) for k, v in model.items()
                }
                return op.__class__(**model)
        else:
            return op

    new_roots = [map_fields(root, filter_type, _visit) for root in roots]
    if made_substitutions != set(substitutions_dict.keys()):
        raise ValueError(
            f"Not all substitutions were made: {made_substitutions} != {set(substitutions_dict.keys())}"
        )
    return new_roots
