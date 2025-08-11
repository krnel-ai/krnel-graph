# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from pydantic import BaseModel
from typing import Any, Callable

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

def get_dependencies(*roots: BaseModel, filter_type: type, recursive: bool) -> set[BaseModel]:
    """Get the dependencies of a given Pydantic model."""
    results = set()

    def _visit(op: BaseModel, depth: int = 0):
        if isinstance(op, filter_type):
            if depth > 0:  # Only add dependencies, not the roots themselves
                results.add(op)
            for field in op.__class__.model_fields:
                v = getattr(op, field)
                map_fields(v, filter_type, lambda x: _visit(x, depth + 1))

    for item in roots:
        _visit(item, depth=0)
    return results

def map_fields(val: BaseModel, filter_type: type, fun: Callable[[BaseModel], BaseModel]):
    if isinstance(val, filter_type):
        return fun(val)
    elif isinstance(val, list):
        return [map_fields(item, filter_type, fun) for item in val]
    elif isinstance(val, dict):
        return {k: map_fields(v, filter_type, fun) for k, v in val.items()}
    # other types
    return val


def graph_substitute(
    roots: list[BaseModel],
    filter_type: type,
    substitutions: list[tuple[BaseModel, BaseModel]],
):
    """Substitute nodes in the graph with new nodes."""
    all_deps = get_dependencies(*roots, filter_type=filter_type, recursive=True)
    for old, new in substitutions:
        if old not in all_deps:
            raise ValueError(f"Supposed to substitute {old}, but it is not in the graph dependencies: {all_deps}")

    substitutions_dict = {old: new for old, new in substitutions}
    made_substitutions = set()

    def _visit(op: BaseModel) -> BaseModel:
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