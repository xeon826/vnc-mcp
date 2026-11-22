"""vnc-mcp

Copyright (C) 2025  Parker Wahle

SPDX-License-Identifier: AGPL-3.0-or-later
"""  # noqa: E501, D415
# This code is from https://github.com/regulad/cookiecutter-neopy/blob/99065f33c840f6980c655e5d251c4225d613fd06/%7B%7Bcookiecutter.project_name%7D%7D/src/%7B%7Bcookiecutter.package_name%7D%7D/utils/asyncio.py, which was a work of mine that never saw the light of day
from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
from collections.abc import AsyncGenerator
from collections.abc import Callable
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import ParamSpec
from typing import TypeVar

import anyio
from anyio import TASK_STATUS_IGNORED
from anyio import CancelScope
from anyio import Event
from anyio import open_signal_receiver
from anyio.abc import TaskStatus


THREAD_POOL_EXECUTORS = min(32, (os.cpu_count() or 1) + 4)

logger = logging.getLogger(__name__)

A_RETVAL = TypeVar("A_RETVAL")


@asynccontextmanager
async def _async_lifespan_manager() -> AsyncGenerator[None, None]:
    """Async context manager for the lifespan of the application."""
    loop = asyncio.get_running_loop()

    # Setup a more managed ThreadPoolExecutor
    # asyncio actually manages the lifespan of the default executor, but it's not very good at it.
    with ThreadPoolExecutor(
        max_workers=THREAD_POOL_EXECUTORS, thread_name_prefix="asyncio_thread"
    ) as executor:
        loop.set_default_executor(executor)
        yield


def run_coroutine_managed(coro: Callable[[], Coroutine[None, None, A_RETVAL]]) -> A_RETVAL:
    """Abstract away lifecycle management for asyncio."""

    async def _async_bootstrap() -> A_RETVAL:
        async with _async_lifespan_manager():
            return await coro()

    return anyio.run(
        _async_bootstrap,
        backend="asyncio",
        backend_options={
            "debug": __debug__,
        },
    )


async def signal_handler(
    cancel_scope: CancelScope,
    event: Event,
    *,
    task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
) -> None:  # pragma: no cover
    """Handles KeyboardInterrupts by cancelling the cancel scope and setting the event.

    Args:
        cancel_scope (CancelScope): The cancel scope to cancel.
        event (Event): The event to set.
        task_status (TaskStatus[None], optional): The task status to set. Defaults to TASK_STATUS_IGNORED.

    """
    # Not sure how I would test this, so I'm just going to ignore it and hope it works
    async with open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
        task_status.started()
        async for signum in signals:
            # We have recieved a shutdown signal, so we need to cancel the cancel scope
            if signum == signal.SIGINT:
                logger.info("Received SIGINT, shutting down...")
            elif signum == signal.SIGTERM:
                logger.info("Received SIGTERM, shutting down...")
            event.set()
            return


P = ParamSpec("P")
R = TypeVar("R")


@functools.cache  # type: ignore
def make_sync(coro: Callable[P, Coroutine[None, None, R]]) -> Callable[P, R]:
    """A decorator that wraps an async function and abstracts away event loop & cancel scope creation."""

    @functools.wraps(coro)
    def sync_function(*args: P.args, **kwargs: P.kwargs) -> R:
        return run_coroutine_managed(functools.partial(coro, *args, **kwargs))

    # Special marker values to make testing easier
    sync_function.__is_wrapped__ = True  # type: ignore
    sync_function.__async_function__ = coro  # type: ignore

    return sync_function


@functools.cache  # type: ignore
def make_async(func: Callable[P, R]) -> Callable[P, Coroutine[None, None, R]]:
    """A decorator that wraps a sync function and abstracts away calling it in an executor."""
    return functools.wraps(func)(functools.partial(asyncio.to_thread, func))


__all__ = ("run_coroutine_managed", "signal_handler", "make_sync")
