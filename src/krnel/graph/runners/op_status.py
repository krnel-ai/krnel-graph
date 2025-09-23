# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_serializer

from krnel.graph.op_spec import OpSpec, graph_serialize


class OpStatus(BaseModel):
    """
    Model representing the status of an operation.
    """

    op: OpSpec
    state: Literal["new", "pending", "running", "completed", "failed", "ephemeral"]
    # - new: Not yet submitted to any runner
    # - pending: Seen by runner, waiting for execution
    # - running: Currently in progress
    # - completed: Finished successfully, result is available or can be downloaded
    # - failed: Finished with an error, no result is available
    # - ephemeral: Result can be computed instantly and therefore does not need to be stored (TBD)

    # Can this operation be quickly materialized?
    # locally_available: bool = False

    time_started: datetime | None = None
    time_completed: datetime | None = None
    # TODO: how to handle multiple successive runs of the same op?
    # e.g. if one fails

    # events: list['LogEvent'] = Field(default_factory=list)

    @field_serializer("op")
    def serialize_op(self, op: OpSpec, info):
        return graph_serialize(op)


class LogEvent(BaseModel):
    time: datetime
    message: str

    # Incremental progress update
    progress_complete: float | None = None
    progress_total: float | None = None
