from datetime import datetime
import enum
from pydantic import BaseModel, Field, SerializeAsAny
from typing import Literal

from krnel.graph.op_spec import OpSpec

class OpStatus(BaseModel):
    """
    Model representing the status of an operation.
    """
    op: SerializeAsAny[OpSpec]
    state: Literal['pending', 'running', 'completed', 'failed', 'ephemeral']
    # pending: Not yet started
    # running: Currently in progress
    # completed: Finished successfully, result is available or can be downloaded
    # failed: Finished with an error, no result is available
    # ephemeral: Result does not need to be stored

    # Can this operation be quickly materialized?
    #locally_available: bool = False

    time_started: datetime | None = None
    time_completed: datetime | None = None
    # TODO: how to handle multiple successive runs of the same op?
    # e.g. if one fails

    events: list['LogEvent'] = Field(default_factory=list)

    @property
    def time_last_updated(self) -> datetime | None:
        """
        Returns the last time the status was updated.
        """
        if self.time_completed:
            if len(self.events) > 0:
                return max(self.time_completed, self.events[-1].time)
            else:
                return self.time_completed
        elif len(self.events) > 0:
            return self.events[-1].time
        else:
            return None

class LogEvent(BaseModel):
    time: datetime
    message: str

    progress_complete: float | None = None
    progress_total: float | None = None
