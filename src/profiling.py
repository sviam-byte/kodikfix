"""Мини-профилирование: декоратор timeit."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def timeit(name: str | None = None) -> Callable[[F], F]:
    """Замер времени выполнения функции и лог в logging."""
    def deco(fn: F) -> F:
        label = name or fn.__name__
        logger = logging.getLogger("kodik.perf")

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                logger.info("%s: %.1f ms", label, dt)

        return cast(F, wrapper)

    return deco
