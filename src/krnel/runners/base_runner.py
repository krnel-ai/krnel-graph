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
from krnel.runners.op_status import LogEvent, OpStatus
from krnel.runners.materialized_result import MaterializedResult

DontSave = namedtuple('DontSave', ['value'])

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

    def _pre_materialize(self, spec: OpSpec) -> None:
        """Hook for pre-materialization setup and validation.

        This method is called before executing an operation and can be overridden
        to perform setup steps, validate graph invariants, or prepare the execution
        environment.

        Args:
            spec: The OpSpec that is about to be materialized.
        """
        #print("TODO: graph invariants: ensure that everything depends on only one dataset")
        pass

    def _post_materialize(self, spec: OpSpec, result: Any, status: OpStatus) -> OpStatus:
        """Hook for post-execution processing and cleanup.

        This method is called after successful operation execution to perform
        post-processing, additional result caching, or cleanup tasks.

        Args:
            spec: The OpSpec that was executed.
            result: The materialized result of the operation.
            status: The current status object for the operation.

        Returns:
            Updated OpStatus object (can be the same instance or a new one).
        """
        return status

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

    def materialize(self, op: OpSpec, *, dry_run:bool = False) -> MaterializedResult:
        """Execute an OpSpec operation and return its materialized result.

        Execution lifecycle:
        1. Update op status to 'running'
        2. Find and execute the implementation function
        3. Validate and process the result
        4. Update status to 'completed' or 'failed'
        5. Cache results if appropriate

        Args:
            op: The OpSpec operation to execute.
            dry_run: If True, only validate and prepare without executing.

        Returns:
            MaterializedResult containing the operation's output.

        Note:
            If the operation depends on other OpSpecs, runner implementations will usually materialize them first, so this method should be reentrant.
        """
        self._pre_materialize(op)

        # If already completed, return cached result
        try:
            status = self.get_status(op)
            if status.state == 'completed':
                if self.has_result(op):
                    return self.get_result(op)
        except NotImplementedError:
            pass

        # Which implementation to call?
        op_type = type(op)
        # Fast path
        #if op_type in _IMPLEMENTATIONS[self.__class__]:
        #    return _IMPLEMENTATIONS[self.__class__][op_type](self, spec)

        # Slow path: Search through method resolution order
        # to find all implementations that can accept op_type.
        # If we find more than one, raise an error for now.
        for superclass in self.__class__.mro():
            matching_implementations = []
            for match_type, fun in _IMPLEMENTATIONS[superclass].items():
                #print(f"Checking {superclass.__name__}.{fun.__name__}, {match_type}...")
                if issubclass(op_type, match_type):
                    #print("    ... matches")
                    matching_implementations.append(
                        (match_type, superclass, fun)
                    )
            if len(matching_implementations) > 1:
                raise ValueError(
                    f"Multiple implementations found for {op_type.__name__}:\n"
                    + "\n".join(f"- {cls.__name__}.{fun.__name__}, matching {match_type}" for (match_type, cls, fun) in matching_implementations)
                )
            elif len(matching_implementations) == 1:
                [(match_type, superclass, fun)] = matching_implementations

                return self._do_run(fun, op, dry_run=dry_run)

        raise NotImplementedError(f"No implementation for {op_type.__name__} in {self.__class__.__name__}")

    def _do_run(self, fun: Callable[[RunnerT, OpSpecT], Any], op: OpSpec, dry_run: bool) -> Any:
        status = self.get_status(op) or OpStatus(
            op=op,
            state='pending',
        )
        if dry_run:
            self.put_status(status)
            return None
        else:
            status.state = 'running'
            status.time_started = datetime.now(timezone.utc)
            self.put_status(status)

        result = fun(self, op)

        if isinstance(result, DontSave):
            # fast path: DontSave means we don't need to save the result
            # or validate it
            result = result.value
            status.state = 'ephemeral'
            status.time_completed = datetime.now(timezone.utc)
            self.put_status(status)
            return MaterializedResult.from_any(result, op)

        # Validate the result
        is_valid = self._validate_result(op, result)
        if is_valid is False or is_valid is None:
            # validation rejected this result
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
        status = self._post_materialize(op, result, status)
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
        match params:
            case [_, param] if isinstance(param.annotation, type) and issubclass(
                param.annotation, OpSpec
            ):
                op_type = param.annotation
            case [_, param]:
                #raise ValueError(f"Expected OpSpec subclass, got {param.annotation}")
                #print(f"WARNING: Expected OpSpec subclass, got {param.annotation}, using it as op_type. {param}")
                op_type = param.annotation
            case _:
                raise ValueError("Function must have signature like: func(runner: BaseRunner, spec: SpecType) -> Any")

        _IMPLEMENTATIONS[cls][op_type] = func
        # TODO: fix typing here ?
        return functools.wraps(func)(func)


    def show(self, op: OpSpec, **kwargs) -> str:
        # TODO(kwilber): Make this API better
        return op.__repr_html_runner__(self, **kwargs)