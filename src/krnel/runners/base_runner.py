# Copyright (c) 2025 Krnel
# Points of Contact:
#   - kimmy@krnel.ai

from datetime import datetime
from typing import Any, Callable, TypeVar
from abc import ABC
from collections import defaultdict, namedtuple
import functools
import inspect
from krnel.graph import OpSpec
from krnel.runners.op_status import LogEvent, OpStatus
from krnel.runners.materialized_result import MaterializedResult

DontSave = namedtuple('DontSave', ['result'])

RunnerT = TypeVar('RunnerT', bound='BaseRunner')
OpSpecT = TypeVar('OpSpecT', bound=OpSpec)

# Concrete implementations of Ops are stored in this dictionary.
# Mapping tuple[type[OpSpec], type[BaseRunner]] to function
_IMPLEMENTATIONS: dict[
    type["BaseRunner"], dict[type["OpSpec"], Callable[[Any, OpSpec], Any]]
] = defaultdict(dict)

class BaseRunner(ABC):
    """
    BaseRunners know how to execute operations (OpSpecs) in a specific environment.

    They can be used to run operations locally, on a remote server, or in any other context according to their implementation.

    The two key methods are:
    - `materialize`: Executes an OpSpec and returns the result.
    - `implementation`: A decorator to register an implementation for a specific OpSpec type.

    """

    def _pre_materialize(self, spec: OpSpec) -> None:
        """
        This method can be overridden to perform any pre-materialization steps.
        For example, it can be used to check graph invariants or prepare the environment.
        """
        #print("TODO: graph invariants: ensure that everything depends on only one dataset")
        pass

    def _post_run(self, spec: OpSpec, result: Any, status: OpStatus) -> OpStatus:
        """
        Save the result of the operation to the cache or perform any post-materialization steps.

        Not run for steps whose result is DontSave.
        """
        return status

    def uuid_to_op(self, uuid: str) -> OpSpec | None:
        """
        Convert a UUID to an OpSpec.
        This is used to look up the OpSpec for a given UUID.
        """
        raise NotImplementedError()

    def get_status(self, spec: OpSpec) -> OpStatus:
        """Get the status of an operation."""
        raise NotImplementedError()
    def put_status(self, status: OpStatus) -> bool:
        """Save the status of an operation."""
        return False
    def has_result(self, spec: OpSpec) -> bool:
        return False
    def get_result(self, spec: OpSpec) -> MaterializedResult:
        """Load the result of an operation."""
        raise NotImplementedError()
    def put_result(self, spec: OpSpec, result: MaterializedResult) -> bool:
        """Write the result of an operation."""
        return False

    def _validate_result(self, spec: OpSpec, result: Any) -> Any | bool:
        if result is None or result is False:
            return False
        return True

    def materialize(self, op: OpSpec) -> MaterializedResult:
        """Execute an operation spec using registered implementations."""
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
                result = self._do_run(fun, op)
                return result

        raise NotImplementedError(f"No implementation for {op_type.__name__} in {self.__class__.__name__}")

    def _do_run(self, fun: Callable[[RunnerT, OpSpecT], Any], op: OpSpec) -> Any:
        status = self.get_status(op) or OpStatus(
            op=op,
            state='pending',
        )
        status.state = 'running'
        status.time_started = datetime.now()
        self.put_status(status)

        result = fun(self, op)

        if isinstance(result, DontSave):
            # fast path: DontSave means we don't need to save the result
            # or validate it
            result = result.result
            status.state = 'ephemeral'
            status.time_completed = datetime.now()
            self.put_status(status)
            return MaterializedResult.from_any(result, op)

        is_valid = self._validate_result(op, result)
        if is_valid is False or is_valid is None:
            # validation rejected this result
            status.state = 'failed'
            self.put_status(status)
            raise ValueError(f"Result of {op} is invalid: {result}")
        elif is_valid is not True:
            # validation transformed the result
            result = is_valid
        # save the result
        result = MaterializedResult.from_any(result, op)
        self.put_result(op, result)
        status.state = 'completed'
        status.time_completed = datetime.now()
        status = self._post_run(op, result, status)
        self.put_status(status)
        return result

    @classmethod
    def implementation(cls, func: Callable[[RunnerT, OpSpecT], Any]) -> Callable[[RunnerT, OpSpecT], Any]:
        """
        Register an implementation for a specific OpSpec type by inspecting the function's type annotations.
        This is intended to be used as a decorator on top-level functions.

        The function should have a signature like: func(runner: BaseRunner, spec: SpecType) -> Any
        The OpSpec type will be inferred from the second parameter's type annotation.
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
        """Return a string representation of the operation."""
        return op.__repr_html_runner__(self, **kwargs)