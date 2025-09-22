# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from krnel.graph.op_spec import OpSpec


class GroupedOp(OpSpec):
    """
    An operation that groups multiple operations together.

    This isn't super useful on its own, but can be used to group operations
    for organizational purposes, or to apply transformations to all operations
    in the group at once.
    """

    ops: list[OpSpec]
