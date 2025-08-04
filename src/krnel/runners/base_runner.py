from typing import Any, Callable, TypeVar
from abc import ABC
import functools
from krnel.graph import OpSpec

T = TypeVar('T', bound='BaseRunner')

class BaseRunner(ABC):
    """
    BaseRunners know how to execute operations (OpSpecs) in a specific environment.

    They can be used to run operations locally, on a remote server, or in any other context according to their implementation.

    The two key methods are:
    - `materialize`: Executes an OpSpec and returns the result.
    - `implementation`: A decorator to register an implementation for a specific OpSpec type.

    """
    def materialize(self, spec: OpSpec) -> Any:
        """Execute an operation spec using registered implementations."""
        op_type = type(spec)
        if not hasattr(self.__class__, '_implementations'):
            raise NotImplementedError(f"No implementations registered for {self.__class__.__name__}")

        if op_type not in self.__class__._implementations:
            raise NotImplementedError(f"No implementation for {op_type.__name__} in {self.__class__.__name__}")

        return self.__class__._implementations[op_type](self, spec)

    @classmethod
    def implementation(cls, op_type: type[OpSpec]) -> Callable[[Callable], Callable]:
        """
        Register an implementation for a specific OpSpec type.
        This is intended to be used as a decorator on top-level functions.
        """
        if not hasattr(cls, '_implementations'):
            cls._implementations = {}

        def decorator(func: Callable[[T, OpSpec], Any]) -> Callable[[T, OpSpec], Any]:
            cls._implementations[op_type] = func
            return functools.wraps(func)(func)

        return decorator
