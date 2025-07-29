from functools import cached_property
import hashlib
from typing import Any
from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler, field_serializer

class ComputationSpec(BaseModel):
    """
    Base class for all computation specifications.
    Each ComputationSpec is content-addressable, meaning it can be uniquely identified by its content.
    """

    @field_serializer('*', mode='wrap')
    def serialize_for_hash(v: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo):
        if context := info.context:
            # If the context is for hashing, we return the content hash directly
            if context.get('for_hash', False):
                if isinstance(v, ComputationSpec):
                    # If the value is a ComputationSpec, return its content hash
                    return v.content_hash
        return nxt(v)

    @cached_property
    def content_hash(self) -> str:
        """
        Generates a content hash for the ComputationSpec instance.
        This hash is used to uniquely identify the computation step.
        """
        return "HASH HERE"
        #content = self.model_dump()
        #return hashlib.sha256(content.encode('utf-8')).hexdigest()

    model_config = ConfigDict(
        frozen = True,  # Makes instances immutable
    )
