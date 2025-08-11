from krnel import graph
import warnings
from rich import print

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
def status(store_uri: str, op_uuid: list[str]):
    runner = LocalArrowRunner(store_uri=store_uri)
    ops = []
    for uuid in op_uuid:
        op = runner.uuid_to_op(uuid)
        if op is None:
            print(f"Operation with UUID {op_uuid} not found.")
            return
        ops.append(op)

    seen = set()
    def _visit(op):
        if op in seen:
            return
        status = runner.get_status(op)
        if status != 'completed':
            for dep in op.get_dependencies():
                _visit(dep)
        if status is None:
            print(f"No status found for operation {op.uuid}.")
        print(status)
        print()
        seen.add(op)
    for op in ops:
        _visit(op)