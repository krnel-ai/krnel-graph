# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime, timezone
import warnings
from rich.tree import Tree
from rich import print
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
def status(store_uri: str, *op_uuid: str):
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
    def show_one(op, tree, name=""):
        if op.uuid in seen:
            return tree
        seen.add(op.uuid)
        status = runner.get_status(op)
        time = format_time(status)
        match status.state:
            case 'completed':
                branch = tree.add(f"{name}: [green]{status.state}[/green] {time}")
            case 'ephemeral':
                branch = tree.add(f"{name}: [green]{status.state}[/green] {time}")
            case 'running':
                branch = tree.add(f"{name}: [blue]{status.state}[/blue] {time}")
            case _:
                branch = tree.add(f"{name}: [yellow]{status.state}[/yellow] {time}")
        for fieldname in op.__class__.model_fields:
            child = getattr(op, fieldname)
            map_fields(
                child,
                OpSpec,
                lambda x: show_one(x, branch, name=f"{fieldname}-{child.uuid if hasattr(child, 'uuid') else ''}"),
            )
        return tree

    for op in ops:
        print(show_one(op, Tree(op.uuid)))

@app.command
def materialize(store_uri: str, *op_uuid: str):
    runner = LocalArrowRunner(store_uri=store_uri)
    for uuid in op_uuid:
        op = runner.uuid_to_op(uuid)
        if op is None:
            print(f"Operation with UUID {uuid} not found.")
            return
        result = runner.materialize(op)
        print(f"Materialized operation {op.uuid}: {result}")
