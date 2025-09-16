# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime
import enum
from pydantic import BaseModel, Field, SerializeAsAny, field_serializer
from typing import Any, Literal

from krnel.graph.op_spec import OpSpec, graph_serialize

class OpStatus(BaseModel):
    """
    Model representing the status of an operation.
    """
    op: OpSpec
    state: Literal['new', 'pending', 'running', 'completed', 'failed', 'ephemeral']
    # - new: Not yet submitted to any runner
    # - pending: Seen by runner, waiting for execution
    # - running: Currently in progress
    # - completed: Finished successfully, result is available or can be downloaded
    # - failed: Finished with an error, no result is available
    # - ephemeral: Result can be computed instantly and therefore does not need to be stored (TBD)

    # Can this operation be quickly materialized?
    #locally_available: bool = False

    time_started: datetime | None = None
    time_completed: datetime | None = None
    # TODO: how to handle multiple successive runs of the same op?
    # e.g. if one fails

    #events: list['LogEvent'] = Field(default_factory=list)

    @field_serializer('op')
    def serialize_op(self, op: OpSpec, info):
        return graph_serialize(op)

    """
    @property
    def time_last_updated(self) -> datetime | None:
        "Returns the last time the status was updated."
        if self.time_completed:
            if len(self.events) > 0:
                return max(self.time_completed, self.events[-1].time)
            else:
                return self.time_completed
        elif len(self.events) > 0:
            return self.events[-1].time
        else:
            return None
    """

class LogEvent(BaseModel):
    time: datetime
    message: str

    # Incremental progress update
    progress_complete: float | None = None
    progress_total: float | None = None
