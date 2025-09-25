# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import importlib.util
import json as json_lib
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

try:
    from cyclopts import App
except ImportError as exc:
    raise ImportError(
        "You must install the 'cli' extra to use the CLI features of Krnel-graph. Run: pip install krnel-graph[cli]"
    ) from exc

from cyclopts import Group, Parameter, validators
from rich import box, print
from rich.markup import escape
from rich.text import Text
from tqdm.auto import tqdm

from krnel import graph, logging
from krnel.graph import config
from krnel.graph.graph_transformations import map_fields
from krnel.graph.grouped_ops import GroupedOp
from krnel.graph.op_spec import OpSpec
from krnel.graph.runners.base_runner import BaseRunner

logger = logging.get_logger(__name__)


app = App(
    name="krnel-graph",
    default_parameter=Parameter(negative=()),
    help="""
        Run, debug, and inspect Krnel computation graphs.

        **Quick example:** Suppose main.py contains the following Python code::

            from krnel.graph import Runner
            runner = Runner()

            # Load data
            ds_train = runner.from_parquet('data.parquet')
            ds_test = runner.from_parquet('data_test.parquet')

            # Get train activations
            X_train = ds_train.col_text("Prompt").llm_layer_activations(
                model="hf:gpt2",
                layer=-1,
            )

            # Get test activations by substituting training set with testing set
            X_test = X_train.subs((ds_train, ds_test))

            # Train and evaluate a probe
            probe = X_train.train_classifier(positives=..., negatives=...)
            test_scores = probe.predict(X_test)
            eval_result = test_scores.evaluate(gt_positives=..., gt_negatives=...)

        This source file defines the following graph of operations:

        .. code-block:: text

            ┌───(ds_train)┐    ┌────────────┐    ┌────────────(X_train)┐
            │ LoadParquet ├──► │ SelectText │ ──►│ LLMLayerActivations │
            └─────────────┘    └────────────┘    └─────────────────────┘
                                                    │
                  Note: (parens) denote             │   ┌──────────(probe)┐     ┌──────(test_scores)┐
                  Python variable names             └───│ TrainClassifier ├──┬─►│ ClassifierPredict │
                                                        └─────────────────┘  │  └───────────┬───────┘
                                                             ┌───────────────┘              ▼
            ┌────(ds_test)┐    ┌────────────┐    ┌───────────┴─(X_test)┐           ┌───(eval_result)┐
            │ LoadParquet ├──► │ SelectText │ ──►│ LLMLayerActivations │           │ ClassifierEval |
            └─────────────┘    └────────────┘    └─────────────────────┘           └────────────────┘

        To inspect the results, you can materialize the required operations
        on-demand in a notebook::

            from main import runner, eval_result, X_train

            # Run everything and print result
            print(runner.to_json(eval_result))

            # Display activations of training set (GPU-intense operation)
            print(runner.to_numpy(X_train))

        Alternatively, you can use this krnel-graph CLI to materialize individual
        operations ahead of time. This can be useful for debugging or for running
        long operations in a controlled manner, eg. on GPU clusters.

        Common krnel-graph operations include:

        Summarize status of all graph ops in a file:
            $ krnel-graph summary -f main.py
        Materialize operations by type (run this on a GPU machine):
            $ krnel-graph run -f main.py -t LLMLayerActivations
        Materialize operations by variable name:
            $ krnel-graph run -f main.py -s probe
        Show status of a single op by UUID:
            $ krnel-graph status -u TrainClassifierOp_123412341234
        Print the full pseudocode definition of operations by variable name:
            $ krnel-graph print -f main.py -s probe
        Specify where to save results:
            $ krnel-graph config --store-uri /path/to/local/dir

""",
)


@Parameter(name="*", group="Common parameters")
@dataclass
class CommonParameters:
    verbose: Annotated[bool, Parameter(alias="-v")] = False
    "Enable debug output"

    # runner_config: Annotated[
    #     config.KrnelGraphConfig | None,
    #     Parameter(name="*"),
    # ] = None


op_source_group = Group(
    "Graph sources",
    help="Specify where to load the graph from. At least one of -f or -u is required.",
    sort_key=1,
)
op_filter_group = Group(
    "Graph filtering",
    help="""Filter which ops to operate on. If multiple filters are specified, an op must match all of them to be included.


    """,
    sort_key=2,
)


@Parameter(name="*", group=op_filter_group)
@dataclass
class OpFilterParameters:
    input_file: Annotated[str | None, Parameter(group=op_source_group, alias="-f")] = (
        None
    )
    "Import the graph from this Python source file. Ops come from all named variables."

    uuid: Annotated[
        list[str] | None,
        Parameter(group=op_source_group, alias="-u", consume_multiple=True),
    ] = None
    "Import the graph by specifying one or more op UUIDs. Can also act as a filter if -f is also specified (substring OK)."

    include_deps: Annotated[bool, Parameter(negative=["--no-deps"])] = True
    "By default, -f and -u also include dependencies of each op. Use --no-deps to only load either top-level variables (-f) or the specified UUIDs (-u)."

    variable_name: Annotated[
        list[str] | None, Parameter(alias="-s", consume_multiple=True)
    ] = None
    "Filter ops by variable name. (Used with -f / --input-file)"

    type: Annotated[list[str] | None, Parameter(alias="-t", consume_multiple=True)] = (
        None
    )
    "Filter ops by operation type."

    parameters: Annotated[
        list[str] | None, Parameter(alias="-p", consume_multiple=True)
    ] = None
    "Filter ops by this parameter value."

    code: Annotated[list[str] | None, Parameter(alias="-S", consume_multiple=True)] = (
        None
    )
    "Pickaxe search through all graph pseudocode for each op."

    state: (
        list[Literal["new", "pending", "running", "completed", "failed", "ephemeral"]]
        | None
    ) = None
    "Filter ops by runtime state. (Can pass multiple --state arguments.)"

    num_ops: Annotated[int | None, Parameter(alias="-n")] = None
    "If non-zero, only include this many ops after all other filtering. For materialize, this is the number of ops to run."

    # ready: bool = False
    # "Only include incomplete ops that are ready to run (all dependencies completed)."


def parse_common_parameters(
    common: CommonParameters | None,
) -> tuple[BaseRunner | None, CommonParameters]:
    if common is None:
        common = CommonParameters()

    if common.verbose:
        logging.configure_logging(log_level="DEBUG", force_reconfigure=True)

    runner = None
    # if common.runner_config is not None:
    #     runner = graph.Runner(**(common.runner_config or {}))

    return runner, common


def filter_ops(
    runner: BaseRunner | None, filter_params: OpFilterParameters | None = None
) -> tuple[BaseRunner, set[OpSpec]]:
    """
    Given a graph of (variable_name, OpSpec) pairs, return only those that match any of the given patterns.

    The patterns can be:
    - OpSpec UUID
    - variable name
    - any substring of the above

    """
    graph_ops: dict[str, OpSpec] = {}

    if filter_params is None:
        filter_params = OpFilterParameters()

    if filter_params.input_file is not None:
        if not Path(filter_params.input_file).exists():
            raise ValueError(f"Input file {filter_params.input_file} does not exist.")
        module_path = Path(filter_params.input_file).resolve()
        sys.path.insert(0, str(module_path.parent))

        spec = importlib.util.spec_from_file_location(
            "__krnel_main__", str(module_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find runner
        if runner is None:
            runners = []
            map_fields(mod.__dict__, BaseRunner, lambda r, path: runners.append(r))
            if len(runners) == 1:
                runner = runners[0]
            else:
                raise ValueError(
                    f"Expected exactly one runner in the input file, found {len(runners)}"
                )

        # Add ops
        def _visit(op, path):
            if isinstance(op, OpSpec):
                graph_ops[".".join(map(str, path))] = op
                if filter_params.include_deps:
                    for new_path, dep in op.get_dependencies(
                        include_names=True, path=path
                    ):
                        graph_ops[new_path] = dep

        map_fields(mod.__dict__, OpSpec, _visit)

    if runner is None:
        # no input file specified and no config, so use defaults
        runner = graph.Runner()

    results: set[OpSpec] = set()

    # UUIDs can be looked up directly, even if they don't appear inside the source file's
    # exported variables.
    for uuid in filter_params.uuid or []:
        if (op := runner.uuid_to_op(uuid)) is not None:
            graph_ops[op.uuid] = op
            if filter_params.include_deps:
                for dep in op.get_dependencies(recursive=True):
                    graph_ops[dep.uuid] = dep

    def _matches(pattern: str, test) -> bool:
        return (
            pattern == test
            or pattern.lower() == test.lower()
            or pattern.lower() in test.lower()
        )

    ## Add dependencies if requested
    # if filter_params.include_deps:
    #    for op in set(graph_ops.values()):
    #        for dep, path in get_dependencies(op, filter_type=OpSpec, recursive=True):
    #            if not dep.is_ephemeral:
    #                graph_ops[dep.uuid] = dep

    # Perform filtering
    # TODO(kwilber): most ops have multiple variable names/paths in the source file, so should optimize this
    for var_name, op in graph_ops.items():
        to_add = True
        # By UUID
        if filter_params.uuid is not None:
            if not any(_matches(pattern, op.uuid) for pattern in filter_params.uuid):
                to_add = False
            else:
                logger.debug("UUID match", uuid=op.uuid, filter=filter_params.uuid)
        # By variable name
        if filter_params.variable_name is not None:
            if not any(
                _matches(pattern, var_name) for pattern in filter_params.variable_name
            ):
                to_add = False
        # By type
        if filter_params.type is not None:
            if not any(
                _matches(pattern, op.__class__.__name__)
                for pattern in filter_params.type
            ):
                to_add = False
        # By parameter value
        if filter_params.parameters is not None:
            any_param_matches = False
            for param_name, param_val in op.model_dump().items():
                if any(
                    _matches(pattern, str(param_val))
                    for pattern in filter_params.parameters
                ):
                    logger.debug(
                        "Parameter match",
                        param_name=param_name,
                        param_val=param_val,
                        filter=filter_params.parameters,
                        op=op.uuid,
                    )
                    any_param_matches = True
            to_add = to_add and any_param_matches
        # By source code
        if filter_params.code is not None:
            if not any(
                _matches(
                    pattern,
                    op.to_code(include_deps=False, include_banner_comment=False),
                )
                for pattern in filter_params.code
            ):
                to_add = False
        # By runtime state
        if to_add and filter_params.state is not None:
            status = runner.get_status(op)
            if status.state not in filter_params.state:
                to_add = False
        # Add to results if all filters match
        if to_add:
            results.add(op)

        if filter_params.num_ops is not None and len(results) >= filter_params.num_ops:
            break

    return runner, results


def exit_on_empty_ops(ops):
    if len(ops) == 0:
        print("[red bold]No ops found.\n")
        print("Specify ops using:")
        print(
            "[yellow bold]    -f [underline]main.py[/underline][/yellow bold] to read operations from a Python file"
        )
        print(
            "[yellow bold]    -f [underline]main.py[/underline] -s [underline]eval[/underline][/yellow bold] to use all operations from main.py that have 'eval' in the Python variable name"
        )
        print(
            "[yellow bold]    -u [underline]UUID[/underline][/yellow bold] if you know the operation's ID"
        )
        print("")
        print("See [yellow]--help[/yellow] for more filtering options.")
        sys.exit(1)


@app.command
def status(
    *,
    json: Annotated[
        bool, Parameter(alias="-j", help="JSON machine-readable output")
    ] = False,
    filter: OpFilterParameters | None = None,
    common: CommonParameters | None = None,
):
    """Show current status of ops."""
    runner, common = parse_common_parameters(common)
    runner, ops = filter_ops(runner, filter)
    exit_on_empty_ops(ops)
    for op in tqdm(
        sorted(ops, key=lambda o: o.uuid),
        desc="Checking status of ops",
        disable=json,
        leave=False,
    ):
        status = runner.get_status(op)
        if json:
            sys.stdout.write(status.model_dump_json())
            sys.stdout.write("\n")
        else:
            style = ""
            timestamp = ""
            if status.state == "new" or status.state == "pending":
                style = ""
            elif status.state == "running":
                style = "green bold"
            elif status.state == "completed":
                style = "blue"
            elif status.state == "failed":
                style = "red"
            print(
                Text.assemble(
                    (op.uuid, style),
                    ": ",
                    (status.state),
                    timestamp,
                )
            )


@app.command
def summary(
    *,
    json: Annotated[
        bool, Parameter(alias="-j", help="JSON machine-readable output")
    ] = False,
    filter: OpFilterParameters | None = None,
    common: CommonParameters | None = None,
):
    """Summarize ops by type."""
    runner, common = parse_common_parameters(common)
    runner, ops = filter_ops(runner, filter)
    exit_on_empty_ops(ops)
    op_to_status = {
        op.uuid: runner.get_status(op).state
        for op in tqdm(
            ops,
            desc="Checking status of ops",
            disable=json,
            leave=False,
        )
    }

    groups = defaultdict(set)
    for op in ops:
        groups[op.__class__.__name__].add(op)

    counter_by_group = {
        group: Counter(op_to_status[op.uuid] for op in ops)
        for group, ops in groups.items()
    }
    if json:
        sys.stdout.write(json_lib.dumps(counter_by_group, indent=2))
        sys.stdout.write("\n")
    else:
        rows = []
        for group, counter in sorted(
            counter_by_group.items(), key=lambda g: (-sum(g[1].values()), g[0])
        ):
            if "ephemeral" in counter:
                del counter["ephemeral"]
            total = sum(counter.values())
            if total == 0:
                continue
            row = [Text(group, style="bold"), str(total)]
            for state in ["new", "pending", "running", "completed", "failed"]:
                if state in counter:
                    style = ""
                    if state == "new" or state == "pending":
                        style = ""
                    elif state == "running":
                        style = "green bold"
                    elif state == "completed":
                        style = "blue"
                    elif state == "failed":
                        style = "red"
                    row.append(Text(str(counter[state]), style=style))
                else:
                    row.append("0")
            rows.append(row)
        from rich.table import Table

        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Group", style="bold")
        table.add_column("Total", justify="right")
        table.add_column("New", justify="right")
        table.add_column("Pending", justify="right")
        table.add_column("Running", justify="right")
        table.add_column("Completed", justify="right")
        table.add_column("Failed", justify="right")
        for row in rows:
            table.add_row(*row)
        print(table)


@app.command(name="print")
def print_(
    *,
    json: Annotated[
        bool, Parameter(alias="-j", help="JSON machine-readable output")
    ] = False,
    filter: OpFilterParameters | None = None,
    common: CommonParameters | None = None,
):
    """Print graph ops in human-readable or JSONL format."""
    runner, common = parse_common_parameters(common)
    runner, ops = filter_ops(runner, filter)
    exit_on_empty_ops(ops)

    if json:
        for op in ops:
            result = {"uuid": op.uuid}
            result.update(op.model_dump())
            sys.stdout.write(json_lib.dumps(result))
            sys.stdout.write("\n")
    else:
        if len(ops) == 1:
            op = list(ops)[0]
        else:
            op = GroupedOp(ops=sorted(ops, key=lambda o: o.uuid))
        print(escape(op.to_code(include_deps=True, include_banner_comment=True)))


@app.command
def make_group(
    *,
    filter: OpFilterParameters | None = None,
    common: CommonParameters | None = None,
):
    """Create a GroupedOp from a set of ops matching a filter."""
    runner, common = parse_common_parameters(common)
    runner, ops = filter_ops(runner, filter)
    exit_on_empty_ops(ops)
    if len(ops) == 1:
        op = list(ops)[0]
    else:
        op = GroupedOp(ops=sorted(ops, key=lambda o: o.uuid))
    runner.prepare(op)
    print(op.uuid)


@app.command(alias=["run"])
def materialize(
    *,
    shard_count: Annotated[
        int,
        Parameter(
            group=Group(
                "Sharding",
                help="Process a subset of ops, for manual parallelization.",
                validator=validators.all_or_none,
            ),
            help="Number of shards to split into",
        ),
    ] = 1,
    shard_idx: Annotated[
        int, Parameter(group="Sharding", help="Which shard to materialize")
    ] = 0,
    shuffle: Annotated[
        bool,
        Parameter(
            negative=["--no-shuffle"],
            help="Shuffle op order",
        ),
    ] = True,
    filter: OpFilterParameters | None = None,
    common: CommonParameters | None = None,
):
    """Materialize the outputs of ops matching a filter."""
    runner, common = parse_common_parameters(common)
    # Hijack num_ops behavior
    if filter is None:
        filter = OpFilterParameters()
    num_ops = filter.num_ops
    filter.num_ops = None

    runner, ops = filter_ops(runner, filter)
    exit_on_empty_ops(ops)

    ops = sorted(ops, key=lambda op: op.uuid)
    ops = [op for op in ops if hash(op.uuid) % shard_count == shard_idx]
    if shuffle:
        random.shuffle(ops)
    n_completed_ops = 0
    for op in tqdm(ops, desc="Materializing ops"):
        if op.is_ephemeral:
            continue
        if runner.has_result(op):
            print(f"[blue]Op {op.uuid} already has a result, skipping.[/blue]")
        else:
            print(f"[green]Materializing op {op.uuid}...[/green]")
            try:
                runner._materialize_if_needed(op)
                n_completed_ops += 1
                if num_ops is not None and n_completed_ops >= num_ops:
                    print(
                        f"[green]Reached requested number of ops ({num_ops}), stopping.[/green]"
                    )
                    break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[red]Error materializing op {op.uuid}: {e}[/red]")
                import traceback

                traceback.print_exc()
                continue
            print(f"[green]Op {op.uuid} materialized.[/green]")


@app.command(name="config")
def config_(
    *,
    new_config: Annotated[
        config.KrnelGraphConfig | None,
        Parameter(name="*"),
    ] = None,
):
    """Get or set configuration options.\n\nKrnel-graph can also be configured using environment variables: **KRNEL_RUNNER_TYPE**, **KRNEL_STORE_URI**, etc."""

    def _print_config(config):
        for field in config.model_fields:
            val = getattr(config, field)
            if isinstance(val, Path):
                val = str(val)
            if config.model_fields[field].description:
                print(f"    [dim]# {config.model_fields[field].description}[/dim]")
            print(f"    {field}: {val!r}")
        print()

    old_config = config.KrnelGraphConfig()
    if new_config is None:
        app.help_print(["config"])
        print(
            f"\n[bold]Path to config file:[/bold] {str(config.KrnelGraphConfig.model_config['json_file'])!r}"
        )
        print("\n[bold]Current config:[/bold]\n")
        _print_config(old_config)
    else:
        old_config = old_config.model_dump()
        for field in new_config.model_fields_set:
            old_config[field] = new_config.model_dump()[field]
        new_config = config.KrnelGraphConfig(**old_config)
        new_config.save()
        print("\n[bold]New config:[/bold]\n")
        _print_config(new_config)
        print(
            f"Configuration saved in config file: {str(config.KrnelGraphConfig.model_config['json_file'])!r}"
        )
