"""Poor man profiler"""

import contextlib
from collections import defaultdict
from functools import wraps
from time import time
from typing import Any, Callable, Iterator, Never

import torch
from torch.nn import Module

from task_solution.custom_logging import LOGGER


class Profiler:
    """Very simple profiler that can handle models, statements and functions."""

    def __init__(self) -> None:
        self.measurements = defaultdict[str, list[float]](list)
        self.operation_name_prefix = ""

    @contextlib.contextmanager
    def profile_operation(self, operation_name: str) -> Iterator[None]:
        """Use this context manager if you want to profile any statement in
        the code.

        Args:
            operation_name: Name of the operation that will be profiled.
        """
        operation_name = self.operation_name_prefix + operation_name
        # TODO: gpu handling, synchronize etc.
        torch.cuda.synchronize()
        start = time()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            duration_in_ms = (time() - start) * 1000
            LOGGER.info(f"{operation_name}: {duration_in_ms} ms")
            self.measurements[operation_name].append(duration_in_ms)

    def profiled_func(
        self,
        func: Callable[..., Any],
        name: str | None = None
    ) -> Callable[..., Any]:
        """Creates wrapper for a function that should be profiled

        Args:
            func: function that will be profiled.
            name: Name of the function.

        Returns:

        """
        name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.profile_operation(name):
                return func(*args, **kwargs)

        return wrapper

    def register_model(self, model: Module) -> None:
        """Registers all layers of the pytorch model to be profiled.

        Args:
            model: Pytorch model.

        """
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.register_model(module)
            else:
                original_forward = module.forward
                module.forward = self.profiled_func(
                    original_forward, name=name
                )


PROFILER = Profiler()
