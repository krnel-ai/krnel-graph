# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import copy
import difflib
import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from types import NoneType, UnionType
from typing import Annotated, Any, ClassVar, TypeVar, Union, get_args, get_origin
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    field_serializer,
    model_serializer,
)

from krnel.graph.graph_transformations import (
    get_dependencies,
    graph_substitute,
    map_fields,
)
from krnel.graph.repr_html import FlowchartReprMixin
from krnel.logging import get_logger

logger = get_logger(__name__)

OpSpecT = TypeVar("OpSpecT", bound="OpSpec")


class UUIDMismatchError(ValueError):
    def __init__(self, old_node_data, old_uuid, new_op):
        DIFFERENCE = "".join("    " + line for line in difflib.unified_diff(
            json.dumps(old_node_data, indent=2).splitlines(keepends=True),
            new_op.model_dump_json(indent=2).splitlines(keepends=True),
            fromfile=f"Previous (saved) {old_uuid}",
            tofile=  f"New (reconstructed) {new_op.uuid}",
        ))
        ERROR_MSG = (
            "UUID mismatch on reserialized node:\n"
            f"{DIFFERENCE}\n"
            f"The definition of {new_op.__class__.__name__} has changed since the graph was serialized (fields added/removed, default values changed, etc). If you're in a notebook, try restarting your Python process to clear any stale class definitions."
        )
        super().__init__(ERROR_MSG)

@dataclass
class ExcludeFromUUID:
    """
    Marker metadata to exclude field from UUID computation while keeping it in other serialization contexts.

    Usage:
        field_name: Annotated[Type, ExcludeFromUUID()]
    """

    pass


class OpSpec(BaseModel, FlowchartReprMixin):
    """
    OpSpec represents a single, immutable node in a content-addressable computation graph.

    Every OpSpec is a declarative specification of an operation or data artifact in a dataflow pipeline.  These nodes are composable: their fields may reference other OpSpecs, forming a directed acyclic graph (DAG) that models the provenance and transformation lineage of datasets, models, and derived artifacts.

    Unlike conventional task DAGs (Airflow, Prefect), or expression DAGs (Polars, TensorFlow), OpSpec graphs explicitly track *artifact identity* and *data lineage* at a fine granularity.

    Each OpSpec, through its structure, defines:
        - its dependencies (inputs) — other OpSpecs it derives from
        - its parameters — configuration values that influence its behavior but are not graph edges

    The key properties of OpSpecs:
        - **Content-Addressable:** Every OpSpec has a unique, deterministic `uuid` derived from its content.
          Two OpSpecs with identical structure and parameters will always yield the same UUID.
        - **Immutable:** Once created, an OpSpec cannot be modified. Mutations produce new OpSpecs.
        - **Type-Resolved DAG Semantics:** Fields of type OpSpec (or subclasses thereof) are treated as DAG edges (inputs).
          Scalar fields (str, int, float, dict, etc.) are treated as parameters.
        - **Self-Serializing:** OpSpecs can serialize themselves into JSON structures suitable for storage, hashing,
          or API payloads. Serialization formats distinguish between full graph snapshots and hash-ref substitutions
          for upstream nodes.
        - **Hydration-Friendly:** Deserialization can hydrate full DAG subtrees or leave upstream nodes as unresolved hash refs.
        - **Field Role Annotation (TODO):** Future extensions will allow for explicit declaration of field roles (inputs vs params),
          but current conventions infer this from field types.

    OpSpec is not a runtime object. It is a **specification** of how an artifact could be computed.  Materialization state (computed/not yet computed/failed) is tracked externally. Execution engines (local or remote) traverse OpSpec graphs to schedule and resolve pending nodes.

    OpSpec is intended to bridge the gap between:
        - Workflow DAGs (Airflow, Dagster) — which are task-centric
        - Artifact Provenance Graphs (DVC, Pachyderm) — which are dataset-centric
        - Expression DAGs (Polars, Ibis) — which are algebraic but ephemeral

    Example:

        .. code-block:: python

            class LLMEmbedSpec(OpSpec):
                input_column: PromptColumnSpec
                model_name: str

            class PromptColumnSpec(OpSpec):
                dataset_root: DatasetRootSpec
                column_name: str

    UUID Exclusion:

        Fields can be excluded from UUID computation using annotations:

        .. code-block:: python

            class CachedOpSpec(OpSpec):
                # These fields affect the UUID
                data: SomeOpSpec
                important_param: str

                # These fields do NOT affect the UUID - useful for caching/debugging
                cache_ttl: Annotated[int, ExcludeFromUUID()] = 3600
                last_accessed: Annotated[str, ExcludeFromUUID()] = ""
    """

    model_config = ConfigDict(frozen=True)

    @field_serializer("*", mode="wrap")
    def serialize_op_fields(
        self, v: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Serialize OpSpec fields by their UUID for content-addressable hashing.

        This field serializer ensures that OpSpec references within the graph
        are represented by their UUIDs rather than their full content when
        computing hashes.

        Returns:
            The serialized field value, with OpSpecs replaced by their UUIDs.
        """
        result = map_fields(v, OpSpec, lambda op, path: op.uuid)
        if result == v:
            # if nothing changed, just call the next handler
            return nxt(v)
        return result

    @model_serializer(mode="wrap")
    def inject_type_on_serialization(
        self, handler: ValidatorFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """Add the 'type' field to the serialized output."""
        result: dict[str, Any] = {}
        result["type"] = self.__class__.__name__
        result.update(handler(self))
        return result

    def _model_dump_for_uuid(self) -> dict[str, Any]:
        """
        Model dump that excludes fields marked with ExcludeFromUUID.
        """
        exclude_fields = set()

        # Check model annotations for ExcludeFromUUID metadata
        for field_name, annotation in self.__annotations__.items():
            if get_origin(annotation) is Annotated:
                metadata = get_args(annotation)[1:]
                if any(isinstance(meta, ExcludeFromUUID) for meta in metadata):
                    exclude_fields.add(field_name)
        # (can't use pydantic's built-in exclude stuff here because
        # only one field serializer is available at a time)

        return self.model_dump(exclude=exclude_fields)

    @cached_property
    def uuid_hash(self) -> str:
        """
        Generates a UUID based on a content hash for the OpSpec instance.
        This hash is used to uniquely identify the OpSpec and its outputs.
        Fields marked with ExcludeFromUUID are excluded from the hash computation.
        """
        content = self._model_dump_for_uuid()
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode("utf-8"),
        ).hexdigest()

    @property
    def uuid(self) -> str:
        return f"{self.__class__.__name__}_{self.uuid_hash}"

    @classmethod
    def parse_uuid(cls, uuid: str) -> tuple[str, str]:
        class_name, _, uuid_hash = uuid.partition("_")
        return class_name, uuid_hash

    def __hash__(self):
        """Return hash of the OpSpec based on its UUID.

        Returns:
            int: Hash value derived from the UUID string.
        """
        return hash(self.uuid)

    def get_dependencies(
        self,
        recursive=False,
        include_names=False,
        path: list | None = None,
    ) -> list["OpSpec"]:
        """
        Returns this operation's dependencies, i.e. all fields that are OpSpecs.

        Args:
            recursive: If True, will show all dependencies recursively.
            include_names: if True, then will return a tuple of (field_name, dep).
        """
        if include_names:
            return [
                (".".join(map(str, path)), op)
                for (op, path) in get_dependencies(
                    self,
                    filter_type=OpSpec,
                    recursive=recursive,
                    path=path,
                )
            ]
        return [
            op
            for (op, path) in get_dependencies(
                self,
                filter_type=OpSpec,
                recursive=recursive,
                path=path,
            )
        ]

    @property
    def is_ephemeral(self) -> bool:
        """
        Returns True if this operation is ephemeral, i.e. it can be computed instantly and does not need to be stored.
        """
        return isinstance(self, EphemeralOpMixin)

    def subs(
        self: OpSpecT,
        substitute: Union[
            "OpSpec", tuple["OpSpec", "OpSpec"], list[tuple["OpSpec", "OpSpec"]], None
        ] = None,
        **changes,
    ) -> OpSpecT:
        """
        Reconstruct the graph while making substitutions.

        - If no substitute target is specified, the given field changes are applied to this OpSpec.
        - If some substitute target is specified, that node is updated with the given field changes, and the entire graph is reconstructed with that node replaced.

        This makes it handy to update specific parts of a complex operation without having to re-specify the entire graph.


        Returns:
            A new OpSpec instance with the specified modifications applied.

            The original graph is not modified.

        Raises:
            ValueError: If invalid field names are provided or conflicting arguments given.

        Examples:

            .. code-block:: python

                dataset = runner.from_dataset("foo.parquet")
                activations = dataset.col_text("text").llm_layer_activations(
                    model_name="hf:gpt2",
                    layer_num=5,
                )
                umap = activations.umap_vis()

                # Update parameters on one operation
                new_activations = activations.subs(model_name="hf:llama2")

                # Update other nodes in the graph
                new_visualization = umap.subs(activations,
                    model_name="hf:llama2",
                    layer_num=6,
                )
                visualization_of_different_dataset = umap.subs(dataset,
                    file_path="different_dataset.parquet",
                )
                different_everything = umap.subs(dataset,
                    file_path="different_dataset.parquet",
                ).subs(activations,
                    model_name="hf:llama2",
                    layer_num=6,
                )

                # Replace a node elsewhere in the graph
                different_dataset = umap.subs(substitute=(dataset, other_dataset))

                # Replace multiple nodes in the graph
                new_op = umap.subs(substitute=[
                    (dataset, other_dataset),
                    (activations, new_activations),
                    ...
                ])

        """
        if substitute is not None:
            if isinstance(substitute, OpSpec):
                # Just replace one node elsewhere in this graph, with keyword arguments
                new_target = substitute.subs(**changes)
                return self.subs(substitute=[(substitute, new_target)])
            elif (
                isinstance(substitute, tuple)
                and len(substitute) == 2
                and all(isinstance(s, OpSpec) for s in substitute)
            ):
                # Just replace one node elsewhere in this graph
                if changes:
                    raise ValueError(
                        "Cannot provide both substitutions and field changes"
                    )
                return self.subs(substitute=[substitute])
            elif isinstance(substitute, list):
                # Replace multiple nodes
                if changes:
                    raise ValueError(
                        "Cannot provide both substitutions and field changes"
                    )
                return graph_substitute(
                    [self],
                    filter_type=OpSpec,
                    substitutions=substitute,
                )[0]
            else:
                raise ValueError("Invalid substitute argument")
        else:
            # If no substitution is provided, return a copy of just this node with updated fields
            cls = self.__class__

            # Validate that all provided field names exist in the model
            valid_fields = set(cls.model_fields.keys())
            invalid_fields = set(changes.keys()) - valid_fields
            if invalid_fields:
                raise ValueError(
                    f"Invalid field names for {cls.__name__}: {sorted(invalid_fields)}. Valid fields: {sorted(valid_fields)}"
                )

            # can't just use self.model_copy(updates=) because @cached_property won't update
            attrs = dict(self).copy()
            attrs.update(changes)
            return cls(**attrs)

    def materialize(self, runner: Any) -> Any:
        # TODO: torn between op.materialize(runner) vs runner.materialize(op)
        # seems like they both have plusses and minuses
        return runner.materialize(self)

    def _code_repr_identifier(self, short=True) -> str:
        """A single identifier that could represent this op on the LHS of
        an equals statement, e.g. trainclassifier_1234"""
        if short:
            return self.__class__.__name__.lower() + "_" + self.uuid_hash[:5]
        else:
            return self.__class__.__name__.lower() + "_" + self.uuid_hash

    def _code_repr_expr(self) -> str:
        """A string representation of this op that can be used in an expression
        as the arguments to some downstream node."""
        return self._code_repr_identifier()

    def _code_repr_statement(self) -> str | None:
        """A string representation of an assignment statement to instantiate this op.
        If not set, then this op only appears as expressions, not as separate
        lines in the code making up the graph.
        """
        results = []
        fq_class_name = self.__class__.__module__ + "." + self.__class__.__name__
        results.append(f"{self._code_repr_identifier()} = {fq_class_name}(")
        for k, v in dict(self).items():
            if k != "uuid_hash":
                v = map_fields(
                    v,
                    OpSpec,
                    lambda op, path: op._code_repr_expr(),
                    lambda op, path: repr(op),
                )
                if isinstance(v, list):
                    v = "[" + ", ".join(v) + "]"
                elif isinstance(v, dict):
                    v = "{" + ", ".join(f"{kk!r}: {vv}" for kk, vv in v.items()) + "}"
                results.append(f"  {k}={v},")
        results.append(")")
        return "\n".join(results)

    def to_code(
        self,
        include_banner_comment=True,
        include_deps=True,
    ) -> str:
        results = []
        seen = set()
        if include_banner_comment:
            results.append(f"# Graph for {self.uuid}")

        def _visit(op: OpSpec):
            if op.uuid not in seen:
                seen.add(op.uuid)
                for child in op.get_dependencies():
                    _visit(child)
                if (stmt := op._code_repr_statement()) is not None:
                    results.append(stmt)

        if include_deps:
            _visit(self)
        else:
            results.append(self._code_repr_statement() or self._code_repr_expr())
        return "\n\n".join(results)

    def __repr__(self) -> str:
        return self.to_code(
            include_deps=False,
            include_banner_comment=False,
        )

    def __str__(self) -> str:
        return self.to_code(
            include_deps=True,
            include_banner_comment=True,
        )

    def diff(self, other) -> str:
        """Compares this op with another op and returns a string describing the differences."""
        if not isinstance(other, OpSpec):
            raise ValueError("Can only diff with another OpSpec instance.")

        from krnel.graph.graph_diff import GraphDiff

        return GraphDiff(self, other)

    def _repr_flowchart_node_(self):
        """Render this node as a Mermaid flowchart (for rich display in html notebook)"""
        return f'{self._code_repr_identifier()}["{self._code_repr_expr()}"]'

    def _repr_flowchart_edges_(self):
        """Render this node as a Mermaid flowchart edge: self -> child."""
        for name, dep in self.get_dependencies(include_names=True):
            yield f'{dep._code_repr_identifier()} -->|"{name}"| {self._code_repr_identifier()}'


def graph_serialize(*graph: OpSpec) -> dict[str, Any]:
    """
    Serializes a graph of OpSpec instances into a JSON-compatible format.

    The on-disk serialization format is:
    {
        "outputs": ["uuid-123", "uuid-456", ...],
        "nodes": {
            "uuid-123": {"type": "OpSpecType", ...this node's fields... },
            "uuid-456": {"type": "OpSpecType", ...this node's fields... },
            ...
        }
    }
    """
    nodes: dict[str, dict[str, Any]] = {}

    def _visit(op: OpSpec):
        if op.uuid not in nodes:
            nodes[op.uuid] = {"type": op.__class__.__name__}
            nodes[op.uuid].update(op.model_dump())
            for parent in op.get_dependencies():
                _visit(parent)

    for op in graph:
        _visit(op)
    return {
        "outputs": [op.uuid for op in graph],
        "nodes": nodes,
    }


def find_subclass_of(
    cls: type, name: str, return_all_matching=False
) -> type | list[type] | None:
    """
    Finds a subclass of `cls` with the given name.

    If there are multiple subclasses with the same name, raises a ValueError
    unless `return_all_matching` is True, in which case it returns a list of
    all matching subclasses.

    If no subclass is found, returns None.
    """

    matching_subclasses = []
    if cls.__name__ == name:
        return cls
    for subclass in cls.__subclasses__():
        if found := find_subclass_of(
            subclass, name, return_all_matching=return_all_matching
        ):
            matching_subclasses.append(found)
    if not return_all_matching and matching_subclasses:
        if len(matching_subclasses) > 1:
            if any(m is not matching_subclasses[0] for m in matching_subclasses):
                raise ValueError(
                    f"Multiple subclasses found for {name}: {matching_subclasses}. If you're in a notebook, try restarting your Python process to clear any stale class definitions."
                )
        return matching_subclasses[0]
    return matching_subclasses or None


def graph_deserialize(data: dict[str, Any]) -> list[OpSpec]:
    """
    Deserializes a graph of OpSpec instances from the JSON-compatible format.

    See the docstring of `graph_serialize` for the on-disk format.

    Returns:
        A list of OpSpec instances corresponding to the output UUIDs.
    """
    original_node_data = copy.deepcopy(data.get("nodes", {}))
    nodes_data = data.get("nodes", {})
    uuid_to_op: dict[str, OpSpec] = {}

    anti_cycle_set = set()

    def _construct_op(uuid: str) -> OpSpec:
        if uuid in uuid_to_op:
            return uuid_to_op[uuid]
        if uuid in anti_cycle_set:
            raise ValueError(f"Cycle detected in graph at node {uuid}")
        anti_cycle_set.add(uuid)
        node_data = nodes_data.get(uuid)
        if node_data is None:
            raise ValueError(f"Node with UUID {uuid} not found in graph data.")
        cls = find_subclass_of(OpSpec, node_data["type"])
        if cls is None:
            raise ValueError(
                f"Class with name {node_data['type']} not found in OpSpec hierarchy."
            )
        # Gotta recursively resolve any OpSpec refs to their fields.
        for name, field in cls.model_fields.items():
            if issubclass(field.annotation, OpSpec):
                # If the field is supposed to be an OpSpec, we need to resolve it by its UUID
                node_data[name] = _construct_op(node_data[name])
            elif (
                get_origin(field.annotation) is UnionType
                or get_origin(field.annotation) is Union
            ):
                # Union types: SpecA | SpecB | SpecC
                # (note: SpecA | None is also allowed)
                args = get_args(field.annotation)
                if any(
                    isinstance(arg, type) and issubclass(arg, OpSpec) for arg in args
                ):
                    if not all(
                        (
                            (isinstance(arg, type) and issubclass(arg, OpSpec))
                            or arg == NoneType
                        )
                        for arg in args
                    ):
                        raise TypeError(
                            f"{cls.__name__}.{name}: Union type must all be OpSpecs, got {args}"
                        )
                    # special exception: fields of type OpSpec | OpSpec | None is permissible
                    if any(arg == NoneType for arg in args) and node_data[name] is None:
                        continue
                    # If the field is a Union that includes an OpSpec, resolve it by its UUID
                    node_data[name] = _construct_op(node_data[name])
            elif get_origin(field.annotation) is list:
                if field.annotation.__args__ and issubclass(
                    field.annotation.__args__[0], OpSpec
                ):
                    # If the field is a list of OpSpecs, resolve each UUID in the list
                    node_data[name] = [_construct_op(uuid) for uuid in node_data[name]]
            elif get_origin(field.annotation) is dict:
                if field.annotation.__args__ and issubclass(
                    field.annotation.__args__[1], OpSpec
                ):
                    # If the field is a dict of OpSpecs, resolve each UUID in the values
                    node_data[name] = {
                        k: _construct_op(v) for k, v in node_data[name].items()
                    }
        uuid_to_op[uuid] = cls(**node_data)
        if uuid != uuid_to_op[uuid].uuid:
            logger.error(
                "UUID mismatch on reserialized node",
                node_data=node_data,
                expected_uuid=uuid,
                actual_uuid=uuid_to_op[uuid].uuid,
            )
            raise UUIDMismatchError(original_node_data[uuid], uuid, uuid_to_op[uuid])
        anti_cycle_set.remove(uuid)
        return uuid_to_op[uuid]

    result = [_construct_op(uuid) for uuid in data["outputs"]]
    if len(nodes_data) != len(uuid_to_op):
        raise ValueError(
            f"Unreachable nodes in graph: {set(nodes_data.keys()) - set(uuid_to_op.keys())}"
        )
    return result


class EphemeralOpMixin(BaseModel):
    """
    Mixin for operations that can be computed instantly and do not need to be stored.

    These operations are typically used for quick computations that do not require
    persistent storage, such as simple transformations or aggregations.
    """

    EPHEMERAL: ClassVar[bool] = True
