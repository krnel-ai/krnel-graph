# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime, timezone
from typing import Annotated
import warnings
from cyclopts import Parameter
from rich.tree import Tree
from rich.panel import Panel
from rich.pretty import Pretty
from rich.columns import Columns
from rich import print, box
import humanize

from krnel import graph
from krnel.graph.graph_transformations import map_fields
from krnel.graph.op_spec import OpSpec
from krnel.runners import LocalArrowRunner

try:
    from cyclopts import App
except ImportError:
    raise ImportError("You must install the 'cli' extra to use the CLI features of Krnel. Run: pip install krnel[cli]")
try:
    import krnel_private.implementations
except ImportError:
    warnings.warn("No private implementations for krnel functions found. Some features may be limited.")

app = App( name="krnel")


@app.command
def status(
    store_uri: str,
    *op_uuid: str,
    verbose: Annotated[bool, Parameter(alias="-v")] = False,
):
    runner = LocalArrowRunner(store_uri=store_uri)
    ops = []
    for uuid in op_uuid:
        op = runner.uuid_to_op(uuid)
        if op is None:
            print(f"Operation with UUID {uuid} not found.")
            return
        ops.append(op)

    def format_time(status):
        match [status.time_started, status.time_completed]:
            case [None, None]:
                return "not started"
            case [time_started, None]:
                return f"for {humanize.naturaldelta(datetime.now(timezone.utc) - time_started)}"
            case [time_started, time_completed]:
                return f"{humanize.naturaldelta(datetime.now(timezone.utc) - time_completed)} ago, took {humanize.naturaldelta(time_completed - time_started)}"

    seen = set()
    def show_one(op):
        if op.uuid in seen:
            return
        seen.add(op.uuid)
        for dep in op.get_dependencies():
            show_one(dep)
        status = runner.get_status(op)
        time = format_time(status)
        show_stats = True
        match status.state:
            case "completed":
                print(f"[green]complete[/green]: {op.uuid}")
            case 'ephemeral':
                show_stats = False
            case 'running':
                print(f"[blue]running[/blue]: {op.uuid}")
            case _:
                print(f"[yellow]{status.state}[/yellow]: {op.uuid}")
        if show_stats and verbose:
            print(Columns(
                ["   ", Pretty(op.model_dump())],
            ))

    for op in ops:
        show_one(op)


@app.command
def materialize(store_uri: str, *op_uuid: str):
    runner = LocalArrowRunner(store_uri=store_uri)
    for uuid in op_uuid:
        op = runner.uuid_to_op(uuid)
        if op is None:
            print(f"Operation with UUID {uuid} not found.")
            return
        if runner.has_result(op):
            print(f"Operation {op.uuid} is already materialized.")
        else:
            result = runner._materialize_if_needed(op)
            print(f"Materialized operation {op.uuid}: {result}")
