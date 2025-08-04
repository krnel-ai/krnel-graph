import base64
from functools import cached_property
import hashlib
import json
from typing import Any, Generic, TypeVar
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

    def model_dump_for_uuid(self) -> dict[str, Any]:
        """
        When computing the UUID, we reference parent OpSpecs by their UUIDs.
        """
        content = self.model_dump()
        for field in self.__class__.model_fields:
            if isinstance(getattr(self, field), OpSpec):
                content[field] = getattr(self, field).uuid
        return content

    @cached_property
    def uuid(self) -> str:
        """
        Generates a UUID based on a content hash for the OpSpec instance.
        This hash is used to uniquely identify the OpSpec and its outputs.
        """
        content = self.model_dump()
        for field in self.__class__.model_fields:
            if isinstance(getattr(self, field), OpSpec):
                content[field] = getattr(self, field).uuid
        content_digest = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode("utf-8"),
        ).digest()
        short_content_digest = base64.b64encode(content_digest, altchars=b'-_').decode('utf-8')
        return "op-" + short_content_digest.rstrip('=')

    # The below adds a "type" field to the serialized output that
    # can be used to identify the specific subclass of OpSpec
    # during both serialization and deserialization.
    # https://github.com/pydantic/pydantic/issues/7366#issuecomment-1742596823

    @model_serializer(mode='wrap')
    def inject_type_on_serialization(
        self, handler: ValidatorFunctionWrapHandler
    ) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if 'type' in self.__class__.model_fields:
            raise ValueError('Cannot use field "type". It is reserved.')
        result['type'] = f'{self.__class__.__name__}'
        return result

    @model_validator(mode='wrap')  # noqa  # the decorator position is correct
    @classmethod
    def retrieve_type_on_deserialization(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> "OpSpec":
        def find_subclass(cls: type, name: str) -> type:
            if cls.__name__ == name:
                return cls
            for subclass in cls.__subclasses__():
                if found := find_subclass(subclass, name):
                    return found
        if isinstance(value, dict):
            # WARNING: we do not want to modify `value` which will come from the outer scope
            # WARNING2: `sub_cls(**modified_value)` will trigger a recursion, and thus we need to remove `type`
            modified_value = value.copy()
            sub_cls_name = modified_value.pop('type', None)
            if sub_cls_name is not None:
                sub_cls = find_subclass(cls=OpSpec, name=sub_cls_name)
                if sub_cls is not None:
                    return sub_cls(**modified_value)
                raise ValueError(f'Subclass with name {sub_cls_name} not found in {cls.__name__} hierarchy.')
            else:
                return handler(value)
        return handler(value)

    def get_parents(
        self,
        recursive=False,
        of_type: type | set[type] | None = None,
    ) -> set["OpSpec"]:
        """
        Returns a list of parent OpSpec instances.  This is used to determine dependencies for the op step.

        Args:
            recursive: If True, will show all dependencies recursively.
            of_type: If provided, will only return dependencies having the specified type(s).

        Returns:
            A set of OpSpec instances.
        """
        results = set()

        if of_type is None:
            of_type = {OpSpec}
        elif isinstance(of_type, type):
            of_type = {of_type}

        for field in self.__class__.model_fields:
            v = getattr(self, field)
            if any(isinstance(v, t) for t in of_type):
                results.add(v)
            if recursive and isinstance(v, OpSpec):
                results.update(
                    v.get_parents(recursive=True, of_type=of_type)
                )
        return results


    def __rich_repr__(self):
        yield "uuid", self.uuid
        for field in self.__class__.model_fields:
            v = getattr(self, field)
            if isinstance(v, OpSpec):
                yield field, namedtuple(v.__class__.__name__, ["uuid", "extra"])(uuid=v.uuid, extra="...")
            else:
                yield field, v