import base64
from functools import cached_property
import hashlib
import json
from typing import Any
from pydantic import BaseModel, ConfigDict, SerializationInfo, SerializerFunctionWrapHandler, field_serializer

class ComputationSpec(BaseModel):
    """
    Base class for all computation specifications.
    Each ComputationSpec is content-addressable, meaning it can be uniquely identified by its content.
    """

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

    model_config = ConfigDict(
        frozen = True,  # Makes instances immutable
    )
