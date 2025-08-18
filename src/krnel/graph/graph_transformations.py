# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from pydantic import BaseModel
from typing import Any, Callable, TypeVar

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U')

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

def get_dependencies(*roots: T, filter_type: type[T], recursive: bool) -> list[T]:
    """Get the dependencies of a given Pydantic model."""
    results = []
    seen = set()

    def _visit(op: T, depth: int = 0) -> T:
        if isinstance(op, filter_type):
            for field in op.__class__.model_fields:
                v = getattr(op, field)
                map_fields(v, filter_type, lambda x: _visit(x, depth + 1))
            if depth > 0:  # Only add dependencies, not the roots themselves
                if op not in seen:
                    seen.add(op)
                    results.append(op)
        return op

    for item in roots:
        _visit(item, depth=0)
    return results

def map_fields(val: Any, filter_type: type[T], fun: Callable[[T], U], other_fun: Callable[[T], U] | None = None) -> Any:
    if isinstance(val, filter_type):
        return fun(val)
    elif isinstance(val, list):
        return [map_fields(item, filter_type, fun) for item in val]
    elif isinstance(val, dict):
        return {k: map_fields(v, filter_type, fun) for k, v in val.items()}
    # other types
    if other_fun is not None:
        return other_fun(val)
    return val


def graph_substitute(
    roots: list[T],
    filter_type: type[T],
    substitutions: list[tuple[T, T]],
) -> list[T]:
    """Substitute nodes in the graph with new nodes."""
    all_deps = get_dependencies(*roots, filter_type=filter_type, recursive=True)
    for old, new in substitutions:
        if old not in all_deps:
            raise ValueError(f"Supposed to substitute {old!r}, but it is not in the graph dependencies: {all_deps!r}")

    substitutions_dict = {old: new for old, new in substitutions}
    made_substitutions = set()

    def _visit(op: T) -> T:
        if isinstance(op, filter_type):
            if op in substitutions_dict:
                made_substitutions.add(op)
                return substitutions_dict[op]
            else:
                # reconstruct the model with the same type
                model = dict(op).copy()
                model = {
                    k: map_fields(v, filter_type, _visit)
                    for k, v in model.items()
                }
                return op.__class__(**model)
        else:
            return op
    new_roots = [map_fields(root, filter_type, _visit) for root in roots]
    assert made_substitutions == set(substitutions_dict.keys()), \
        f"Not all substitutions were made: {made_substitutions} != {set(substitutions_dict.keys())}"
    return new_roots