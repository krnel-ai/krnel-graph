import base64
from functools import cached_property
import hashlib
import json
from typing import Any
from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler, ValidatorFunctionWrapHandler, field_serializer, model_serializer, model_validator

class ComputationSpec(BaseModel):
    """
    Base class for all computation specifications.
    Each ComputationSpec is content-addressable, meaning it can be uniquely identified by its content.

    Class names of all subclasses of ComputationSpec must be globally unique.
    """

    model_config = ConfigDict(
        frozen = True,  # Makes instances immutable
    )

    @field_serializer('*', mode='wrap')
    def serialize_for_hash(
        v: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """
        When generating a hash of any computation spec, we want to reference
        the computation specs that it depends on by their content hash.
        """
        if context := info.context:
            if context.get('for_hash', False):
                if isinstance(v, ComputationSpec):
                    return v.content_hash
        return nxt(v)

    @cached_property
    def content_hash(self) -> str:
        """
        Generates a content hash for the ComputationSpec instance.
        This hash is used to uniquely identify the computation step.
        """
        content = self.model_dump(
            context={"for_hash": True},
        )
        content_digest = hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
            ).encode("utf-8"),
        ).digest()
        short_content_digest = base64.b64encode(content_digest, altchars=b'-_').decode('utf-8')
        return "step-" + short_content_digest.rstrip('=')

    # The below adds a "type" field to the serialized output that
    # can be used to identify the specific subclass of ComputationSpec
    # during both serialization and deserialization.
    # https://github.com/pydantic/pydantic/issues/7366#issuecomment-1742596823

    @model_serializer(mode='wrap')
    def inject_type_on_serialization(
        self, handler: ValidatorFunctionWrapHandler
    ) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        print('...', result)
        if 'type' in self.__class__.model_fields:
            raise ValueError('Cannot use field "type". It is reserved.')
        result['type'] = f'{self.__class__.__name__}'
        return result

    @model_validator(mode='wrap')  # noqa  # the decorator position is correct
    @classmethod
    def retrieve_type_on_deserialization(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> "ComputationSpec":
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
                sub_cls = find_subclass(cls=ComputationSpec, name=sub_cls_name)
                if sub_cls is not None:
                    return sub_cls(**modified_value)
                raise ValueError(f'Subclass with name {sub_cls_name} not found in {cls.__name__} hierarchy.')
            else:
                return handler(value)
        return handler(value)
