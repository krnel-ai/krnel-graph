# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

import functools
import inspect
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from krnel.graph import OpSpec
from krnel.graph.runners.op_status import OpStatus
from krnel.logging import get_logger

logger = get_logger(__name__)

RunnerT = TypeVar("RunnerT", bound="BaseRunner")
OpSpecT = TypeVar("OpSpecT", bound=OpSpec)

# Concrete implementations of Ops are stored in this dictionary.
# Mapping tuple[type[OpSpec], type[BaseRunner]] to function
_IMPLEMENTATIONS: dict[
    type["BaseRunner"], dict[type["OpSpec"], Callable[[Any, OpSpec], Any]]
] = defaultdict(dict)


class BaseRunner:
    """Abstract base class for executing OpSpec operations in various environments.

    BaseRunners provide a unified interface for executing operations (OpSpecs) across
    different environments like local machines, remote servers, or cloud platforms.
    They handle operation execution, caching, status tracking, and result materialization.

    Key Features:
        - Operation execution via registered implementations
        - Result caching and status persistence
        - Graph dependency resolution
        - Validation and error handling

    The core workflow is:
        1. Register implementations for specific OpSpec types using @implementation
        2. Call materialize() to execute operations and their dependencies
        3. Results are cached and status is tracked automatically

    Subclasses must implement:
        - Storage methods (get_result, put_result, etc.) for their target environment
        - Operation implementations using the @implementation decorator

    Example:
        class MyRunner(BaseRunner):
            ...

        @MyRunner.implementation
        def my_op_impl(runner, op: TrainClassifierOp) -> Any:
            # Dispatched by type annotation
            return process_my_op(op)

        runner = MyRunner()
        result = runner.materialize(my_op_spec)
    """

    def prepare(self, op: OpSpec) -> None:
        """Prepare a graph for execution, e.g. register datasets, validate invariants, make sure this op exists in status store, etc.

        This method is called before executing an operation and can be overridden
        to perform setup steps, validate graph invariants, or prepare the execution
        environment.

        Args:
            spec: The OpSpec that is about to be materialized.
        """
        # TODO: graph invariants: ensure that everything depends on only one dataset
        self.get_status(op)  # Ensure the op exists in the store
        return

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        """Retrieve an OpSpec instance by its UUID. The UUID must exist in the store.

        Args:
            uuid: The unique identifier of the OpSpec to retrieve.

        Returns:
            The OpSpec instance with the given UUID, or None if not found.
        """
        raise NotImplementedError()

    def get_status(self, op: OpSpec) -> OpStatus:
        """Retrieve the current execution status of an operation.

        Args:
            spec: The OpSpec whose status to retrieve.

        Returns:
            OpStatus object containing the current state, timestamps, and metadata.
        """
        raise NotImplementedError()

    def put_status(self, status: OpStatus) -> bool:
        """Persist the execution status of an operation.

        Args:
            status: OpStatus object to save.

        Returns:
            True if successfully saved, False otherwise.
        """
        return False

    def has_result(self, op: OpSpec) -> bool:
        """Check if a cached result exists for the given operation.

        Args:
            spec: The OpSpec to check for cached results.

        Returns:
            True if a cached result exists, False otherwise.
        """
        return False

    def _materialize_if_needed(self, op: OpSpec) -> bool:
        """Execute an OpSpec operation if needed. Returns True if execution was performed.

        Execution lifecycle:
        1. Update op status to 'running'
        2. Find and execute the implementation function
        3. Validate and process the result
        4. Update status to 'completed' or 'failed'
        5. Cache results if appropriate

        Args:
            op: The OpSpec operation to execute.

        Returns:
            True if execution was performed, False if already available.
        """
        log = logger.bind(op=op.uuid)
        log.debug("materialize_if_needed()")

        # If already completed, nothing to do
        status = self.get_status(op)
        try:
            if status.state == "completed":
                if self.has_result(op):
                    log.debug("materialize_if_needed(): result already available")
                    return False
                else:
                    log.error(
                        f"materialize_if_needed(): operation {op.uuid} is marked as completed but no result found in store."
                    )
                    raise ValueError(
                        f"Operation {op.uuid} is marked as completed but no result found in store."
                    )
        except NotImplementedError:
            pass

        # Which implementation to call?
        op_type = type(op)
        # Fast path
        # if op_type in _IMPLEMENTATIONS[self.__class__]:
        #    return _IMPLEMENTATIONS[self.__class__][op_type](self, spec)

        # Slow path: Search through method resolution order
        # to find all implementations that can accept op_type.
        # If we find more than one, raise an error for now.
        log = log.bind(op_type=op_type.__name__, runner_type=type(self).__name__)
        for superclass in self.__class__.mro():
            matching_implementations = []
            for match_type, fun in _IMPLEMENTATIONS[superclass].items():
                if issubclass(op_type, match_type):
                    log.debug(
                        f"...matches implementation {superclass.__name__}'s {fun.__name__}() accepting {str(match_type)}..."
                    )
                    matching_implementations.append((match_type, superclass, fun))
            if len(matching_implementations) > 1:
                # TODO: could be possible to distinguish by most specific subtype, but i feel like i'm implementing C++'s method resolution order here
                log.error(
                    "Multiple implementations found, cannot disambiguate",
                    count=len(matching_implementations),
                    matching_implementations=matching_implementations,
                )
                raise ValueError(
                    f"Multiple implementations found for {op_type.__name__}:\n"
                    + "\n".join(
                        f"- {cls.__name__}.{fun.__name__}, matching {match_type}"
                        for (match_type, cls, fun) in matching_implementations
                    )
                )
            elif len(matching_implementations) == 1:
                [(match_type, superclass, fun)] = matching_implementations

                self._do_run(fun, op, status)
                return True

        raise NotImplementedError(
            f"No implementation for {op_type.__name__} in {self.__class__.__name__}"
        )

    def _do_run(
        self, fun: Callable[[RunnerT, OpSpecT], Any], op: OpSpec, status: OpStatus
    ) -> Any:
        log = logger.bind(
            op=op.uuid, op_type=type(op).__name__, runner_type=type(self).__name__
        )
        status.state = "running"
        status.time_started = datetime.now(timezone.utc)
        self.put_status(status)

        log = log.bind(fun=fun.__name__)
        log.debug(f"Calling implementation {fun.__name__}()")
        result = fun(self, op)
        if result is not None:
            log.warn("@implementation functions shouldn't return anything")
            pass

        # Mark operation as completed - implementations handle their own storage and validation
        status.state = "completed"
        status.time_completed = datetime.now(timezone.utc)
        self.put_status(status)
        return result

    @classmethod
    def implementation(
        cls, func: Callable[[RunnerT, OpSpecT], Any]
    ) -> Callable[[RunnerT, OpSpecT], Any]:
        """Decorator to register an implementation function for a specific OpSpec type.

        This decorator inspects the function's type annotations to determine which
        OpSpec type it handles, then registers it with the runner class.

        Args:
            func: Implementation function with signature:
                  func(runner: RunnerType, spec: OpSpecType) -> Any

        Returns:
            The original function, unchanged (decorator pattern).

        Example:
            @MyRunner.implementation
            def handle_my_op(runner: MyRunner, op: MyOpSpec) -> str:
                return f"Processed {op.param}"

        Note:
            The OpSpec type is inferred from the second parameter's type annotation.
            Functions should follow the signature: func(runner, spec) -> result
        """
        # Extract OpSpec type from second parameter's annotation
        params = list(inspect.signature(func).parameters.values())
        match params:
            case [_, param] if isinstance(param.annotation, type) and issubclass(
                param.annotation, OpSpec
            ):
                op_type = param.annotation
            case [_, param]:
                # sometimes happens with union types like `SelectCategoricalColumnOp | SelectTextColumnOp | ...`
                op_type = param.annotation
            case _:
                raise ValueError(
                    "Function must have signature like: func(runner: BaseRunner, spec: SpecType) -> Any"
                )

        _IMPLEMENTATIONS[cls][op_type] = func
        # TODO: fix typing here ?
        return functools.wraps(func)(func)

    def show(self, op: OpSpec, **kwargs) -> str:
        # TODO(kwilber): Make this API better
        return op.__repr_html_runner__(self, **kwargs)

    def to_numpy(self, op: OpSpec) -> Any:
        """Materialize operation as a numpy array.

        Args:
            op: The OpSpec operation to get results for

        Returns:
            numpy array representation of the result
        """
        raise NotImplementedError()

    def to_arrow(self, op: OpSpec) -> Any:
        """Materialize operation as an Arrow table.

        Args:
            op: The OpSpec operation to get results for

        Returns:
            Arrow table representation of the result
        """
        raise NotImplementedError()

    def to_pandas(self, op: OpSpec) -> Any:
        """Materialize operation as a pandas DataFrame.

        Args:
            op: The OpSpec operation to get results for

        Returns:
            pandas DataFrame representation of the result
        """
        raise NotImplementedError()

    def to_json(self, op: OpSpec) -> Any:
        """Materialize operation as JSON and deserialize.

        Args:
            op: The OpSpec operation to get results for

        Returns:
            Python dictionary, list, or JSON data type
        """
        raise NotImplementedError()

    def to_sklearn_estimator(self, op: OpSpec) -> Any:
        """Materialize an SKLearn estimator from an operation result.

        Args:
            op: The OpSpec operation to get results for

        Returns:
            Deserialized Python object
        """
        raise NotImplementedError()

    def write_numpy(self, op: OpSpec, data: Any) -> bool:
        """Write numpy array operation result.

        Args:
            op: The OpSpec operation to write results for
            data: numpy array data to write

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError()

    def write_arrow(self, op: OpSpec, data: Any) -> bool:
        """Write Arrow table or array result for an operation.

        Args:
            op: The OpSpec operation to write results for
            data: Arrow table or array data to write

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError()

    def write_json(self, op: OpSpec, data: Any) -> bool:
        """Write JSON data for an operation.

        Args:
            op: The OpSpec operation to write results for
            data: JSON-serializable data to write

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError()

    def write_sklearn_estimator(self, op: OpSpec, data: Any) -> bool:
        """Write a sklearn estimator as a result of an operation (e.g. training)

        Args:
            op: The OpSpec operation to write results for
            data: Python object to pickle and write

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError()
