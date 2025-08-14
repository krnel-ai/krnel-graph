# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime, timezone
from typing import Any, Callable, TypeVar
from abc import ABC
from collections import defaultdict, namedtuple
import functools
import inspect
from krnel.graph import OpSpec
from krnel.logging import get_logger
from krnel.runners.op_status import LogEvent, OpStatus
from krnel.runners.materialized_result import MaterializedResult

logger = get_logger(__name__)

RunnerT = TypeVar('RunnerT', bound='BaseRunner')
OpSpecT = TypeVar('OpSpecT', bound=OpSpec)

# Concrete implementations of Ops are stored in this dictionary.
# Mapping tuple[type[OpSpec], type[BaseRunner]] to function
_IMPLEMENTATIONS: dict[
    type["BaseRunner"], dict[type["OpSpec"], Callable[[Any, OpSpec], Any]]
] = defaultdict(dict)

class BaseRunner(ABC):
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
        """Retrieve an OpSpec instance by its UUID.

        Args:
            uuid: The unique identifier of the OpSpec to retrieve.

        Returns:
            The OpSpec instance with the given UUID, or None if not found.
        """
        raise NotImplementedError()

    def get_status(self, spec: OpSpec) -> OpStatus:
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

    def has_result(self, spec: OpSpec) -> bool:
        """Check if a cached result exists for the given operation.

        Args:
            spec: The OpSpec to check for cached results.

        Returns:
            True if a cached result exists, False otherwise.
        """
        return False

    def get_result(self, spec: OpSpec) -> MaterializedResult:
        """Retrieve the cached result of an operation.

        Args:
            spec: The OpSpec whose result to retrieve.

        Returns:
            MaterializedResult containing the cached operation result.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def put_result(self, spec: OpSpec, result: MaterializedResult) -> bool:
        """Store the result of an operation for future use.

        Args:
            spec: The OpSpec whose result is being stored.
            result: The MaterializedResult to cache.

        Returns:
            True if successfully stored, False otherwise.
        """
        return False

    def _validate_result(self, spec: OpSpec, result: Any) -> Any | bool:
        """Validate and optionally transform operation results.

        Args:
            spec: The OpSpec that produced the result.
            result: The raw result from the operation implementation.

        Returns:
            - True: Result is valid, use as-is
            - False/None: Result is invalid, mark operation as failed
            - Any other value: Use this value as the transformed result
        """
        if result is None or result is False:
            return False
        return True

    def materialize(self, op: OpSpec) -> MaterializedResult:
        """Execute an OpSpec operation and return its materialized result.

        Execution lifecycle:
        1. Update op status to 'running'
        2. Find and execute the implementation function
        3. Validate and process the result
        4. Update status to 'completed' or 'failed'
        5. Cache results if appropriate

        Args:
            op: The OpSpec operation to execute.

        Returns:
            MaterializedResult containing the operation's output.

        Note:
            If the operation depends on other OpSpecs, runner implementations will usually materialize them first, so this method should be reentrant.
        """
        log = logger.bind(op=op.uuid)
        log.debug("materialize()")
        self.prepare(op)

        # If already completed, return cached result
        status = self.get_status(op)
        try:
            if status.state == 'completed':
                if self.has_result(op):
                    log.debug(f"materialize(): result served from store")
                    return self.get_result(op)
                else:
                    log.error(f"materialize(): operation {op.uuid} is marked as completed but no result found in store.")
                    raise ValueError(f"Operation {op.uuid} is marked as completed but no result found in store.")
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
                    log.debug(f"...matches implementation {superclass.__name__}'s {fun.__name__}() accepting {str(match_type)}...")
                    matching_implementations.append(
                        (match_type, superclass, fun)
                    )
            if len(matching_implementations) > 1:
                log.warn("Multiple implementations found, cannot disambiguate", count=len(matching_implementations), matching_implementations=matching_implementations)
                raise ValueError(
                    f"Multiple implementations found for {op_type.__name__}:\n"
                    + "\n".join(f"- {cls.__name__}.{fun.__name__}, matching {match_type}" for (match_type, cls, fun) in matching_implementations)
                )
            elif len(matching_implementations) == 1:
                [(match_type, superclass, fun)] = matching_implementations

                return self._do_run(fun, op, status)

        raise NotImplementedError(f"No implementation for {op_type.__name__} in {self.__class__.__name__}")

    def _do_run(
        self, fun: Callable[[RunnerT, OpSpecT], Any], op: OpSpec, status: OpStatus
    ) -> Any:
        log = logger.bind(op=op.uuid, op_type=type(op).__name__, runner_type=type(self).__name__)
        status.state = 'running'
        status.time_started = datetime.now(timezone.utc)
        self.put_status(status)

        log.debug(f"Calling implementation {fun.__name__}()")
        result = fun(self, op)

        # Validate the result
        is_valid = self._validate_result(op, result)
        if is_valid is False or is_valid is None:
            # validation rejected this result
            log.warn("Result invalid", result=result)
            status.state = 'failed'
            status.time_completed = datetime.now(timezone.utc)
            self.put_status(status)
            raise ValueError(f"Result of {op} is invalid: {result}")
        elif is_valid is not True:
            # validation transformed the result
            result = is_valid

        # Save the result and mark completed
        result = MaterializedResult.from_any(result, op)
        self.put_result(op, result)
        status.state = 'completed'
        status.time_completed = datetime.now(timezone.utc)
        self.put_status(status)
        return result

    @classmethod
    def implementation(cls, func: Callable[[RunnerT, OpSpecT], Any]) -> Callable[[RunnerT, OpSpecT], Any]:
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
        log = logger.bind(runner_type=cls.__name__, func=func.__name__)
        match params:
            case [_, param] if isinstance(param.annotation, type) and issubclass(
                param.annotation, OpSpec
            ):
                op_type = param.annotation
            case [_, param]:
                # sometimes happens with union types like `SelectCategoricalColumnOp | SelectTextColumnOp | ...`
                op_type = param.annotation
            case _:
                raise ValueError("Function must have signature like: func(runner: BaseRunner, spec: SpecType) -> Any")

        _IMPLEMENTATIONS[cls][op_type] = func
        # TODO: fix typing here ?
        return functools.wraps(func)(func)

    def show(self, op: OpSpec, **kwargs) -> str:
        # TODO(kwilber): Make this API better
        return op.__repr_html_runner__(self, **kwargs)
