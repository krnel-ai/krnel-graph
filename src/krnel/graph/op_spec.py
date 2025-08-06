import base64
from functools import cached_property
import hashlib
import json
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar, get_origin
from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler, ValidatorFunctionWrapHandler, field_serializer, model_serializer, model_validator
from collections import namedtuple

class OpSpec(BaseModel):
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

    By explicitly modeling column-level operations, index lineage, and content-addressability, OpSpec enables reproducible, cache-friendly, and UI-inspectable ML pipelines where both data and computation steps are first-class citizens in the graph.

    Example Usage:
        class LLMEmbedSpec(OpSpec):
            input_column: PromptColumnSpec
            model_name: str

        class PromptColumnSpec(OpSpec):
            dataset_root: DatasetRootSpec
            column_name: str
    """

    model_config = ConfigDict(frozen = True)

    @field_serializer('*', mode='wrap')
    def serialize_op_fields(self, v: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo):
        "Serialize fields that are OpSpecs by their UUID when generating a hash."
        result = map_fields(v, OpSpec, lambda op: op.uuid)
        if result == v:
            # if nothing changed, just call the next handler
            return nxt(v)
        return result

    @model_serializer(mode='wrap')
    def inject_type_on_serialization(
        self, handler: ValidatorFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """Add the 'type' field to the serialized output."""
        result: dict[str, Any] = {}
        result['type'] = self.__class__.__name__
        result.update(handler(self))
        return result

    @cached_property
    def uuid_hash(self) -> str:
        """
        Generates a UUID based on a content hash for the OpSpec instance.
        This hash is used to uniquely identify the OpSpec and its outputs.
        """
        content = self.model_dump()
        content_digest = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode("utf-8"),
        ).digest()
        short_content_digest = base64.b64encode(content_digest, altchars=b'-_').decode('utf-8')
        return short_content_digest.rstrip('=')

    @property
    def uuid(self) -> str:
        return f'{self.__class__.__name__}-{self.uuid_hash}'

    def __hash__(self):
        return hash(self.uuid)


    def get_parents(
        self,
        recursive=False,
    ) -> set["OpSpec"]:
        """
        Returns a set of parent OpSpec instances.  This is used to determine dependencies for the op step.

        Args:
            recursive: If True, will show all dependencies recursively.

        """
        results = set()

        def _visit(op: OpSpec):
            if isinstance(op, OpSpec):
                results.add(op)
                for field in op.__class__.model_fields:
                    v = getattr(op, field)
                    map_fields(v, OpSpec, lambda x: _visit(x))

        _visit(self)
        results.discard(self)
        return results

    def __rich_repr__(self):
        yield "uuid", self.uuid
        for field in self.__class__.model_fields:
            v = getattr(self, field)
            if isinstance(v, OpSpec):
                yield field, namedtuple(v.__class__.__name__, ["uuid", "extra"])(uuid=v.uuid, extra="...")
            else:
                yield field, v

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
            nodes[op.uuid] = {'type': op.__class__.__name__}
            nodes[op.uuid].update(op.model_dump())
            for parent in op.get_parents():
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
        if found := find_subclass_of(subclass, name, return_all_matching=return_all_matching):
            matching_subclasses.append(found)
    if not return_all_matching and matching_subclasses:
        if len(matching_subclasses) > 1:
            raise ValueError(f"Multiple subclasses found for {name}: {matching_subclasses}")
        return matching_subclasses[0]
    return matching_subclasses or None


def graph_deserialize(data: dict[str, Any]) -> list[OpSpec]:
    """
    Deserializes a graph of OpSpec instances from the JSON-compatible format.

    See the docstring of `graph_serialize` for the on-disk format.

    Returns:
        A list of OpSpec instances corresponding to the output UUIDs.
    """
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
        cls = find_subclass_of(OpSpec, node_data['type'])
        if cls is None:
            raise ValueError(f"Class with name {node_data['type']} not found in OpSpec hierarchy.")
        # Gotta recursively resolve any OpSpec refs to their fields.
        for name, field in cls.model_fields.items():
            if issubclass(field.annotation, OpSpec):
                # If the field is supposed to be an OpSpec, we need to resolve it by its UUID
                node_data[name] = _construct_op(node_data[name])
            elif get_origin(field.annotation) is list:
                if field.annotation.__args__ and issubclass(field.annotation.__args__[0], OpSpec):
                    # If the field is a list of OpSpecs, resolve each UUID in the list
                    node_data[name] = [_construct_op(uuid) for uuid in node_data[name]]
            elif get_origin(field.annotation) is dict:
                if field.annotation.__args__ and issubclass(field.annotation.__args__[1], OpSpec):
                    # If the field is a dict of OpSpecs, resolve each UUID in the values
                    node_data[name] = {k: _construct_op(v) for k, v in node_data[name].items()}
        uuid_to_op[uuid] = cls(**node_data)
        if uuid != uuid_to_op[uuid].uuid:
            raise ValueError(
                f"UUID mismatch on reserialized node: when deserializing {uuid}, the resulting value actually has {uuid_to_op[uuid].uuid}"
            )
        anti_cycle_set.remove(uuid)
        return uuid_to_op[uuid]

    outputs = data.get("outputs", [])
    return [_construct_op(uuid) for uuid in outputs]


def map_fields(val: Any, filter_type: type, fun: Callable[[Any], Any]):
    if isinstance(val, filter_type):
        return fun(val)
    elif isinstance(val, list):
        return [map_fields(item, filter_type, fun) for item in val]
    elif isinstance(val, dict):
        return {k: map_fields(v, filter_type, fun) for k, v in val.items()}
    # other types
    return val
