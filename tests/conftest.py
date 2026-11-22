"""Shared fixtures: a fake VNCClient so no test ever touches the network."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from pyvnc import VNCConfig

import vnc_mcp.mcp as mcp_module
from vnc_mcp.mcp import create_mcp_server


FRAMEBUFFER_WIDTH = 200
FRAMEBUFFER_HEIGHT = 100


class FakeVNCClient:
    """Minimal stand-in for pyvnc's VNCClient with recorded calls."""

    def __init__(self, width: int = FRAMEBUFFER_WIDTH, height: int = FRAMEBUFFER_HEIGHT) -> None:
        self.rect = __import__("pyvnc").Rect(0, 0, width, height)
        self.desktop_name = "fake-desktop"
        self.capture_calls: list[Any] = []
        self.pressed_keys: list[tuple[str, ...]] = []
        self.written_strings: list[str] = []
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[int] = []
        self.scrolled: list[tuple[str, int]] = []
        self.reconnect_calls = 0
        self.reconnect_fails = False
        self.capture_errors: list[Exception] = []
        self.last_capture_stale = False
        self.last_cut_text: str | None = None
        self.last_cut_text_time: float = 0.0
        self.last_update_time: float = 0.0
        self.last_input_time: float = 0.0
        self.last_error: Exception | None = None
        self.capture_array = np.zeros((height, width, 4), dtype=np.uint8)

    @property
    def is_connected(self) -> bool:
        return True

    async def capture(self, rect: Any = None, **kwargs: Any) -> np.ndarray:
        if self.capture_errors:
            raise self.capture_errors.pop(0)
        self.capture_calls.append(rect)
        if rect is None:
            return self.capture_array.copy()
        return self.capture_array[
            rect.y : rect.y + rect.height, rect.x : rect.x + rect.width
        ].copy()

    async def press(self, *keys: str) -> FakeVNCClient:
        self.pressed_keys.append(keys)
        return self

    async def write(self, text: str) -> FakeVNCClient:
        self.written_strings.append(text)
        return self

    @asynccontextmanager
    async def hold_key(self, *keys: str):
        yield self

    @asynccontextmanager
    async def hold_mouse(self, button: int = 0):
        yield self

    async def move(self, point: Any) -> FakeVNCClient:
        self.moves.append((point.x, point.y))
        return self

    async def click(self, button: int = 0) -> FakeVNCClient:
        self.clicks.append(button)
        return self

    async def scroll_up(self, repeat: int = 1) -> FakeVNCClient:
        self.scrolled.append(("up", repeat))
        return self

    async def scroll_down(self, repeat: int = 1) -> FakeVNCClient:
        self.scrolled.append(("down", repeat))
        return self

    async def reconnect(self) -> bool:
        self.reconnect_calls += 1
        if getattr(self, "reconnect_fails", False):
            self.last_error = ConnectionError("reconnect failed")
            return False
        self.last_capture_stale = False
        return True


class FakeOCREngine:
    """Stand-in for rapidocr.RapidOCR returning configurable fake results."""

    def __init__(self, boxes: Any, txts: tuple[str, ...]) -> None:
        self.boxes = boxes
        self.txts = txts
        self.calls: list[Any] = []

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        self.calls.append(image)

        class _Result:
            pass

        result = _Result()
        result.boxes = self.boxes
        result.txts = self.txts
        return result


@pytest.fixture
def fake_client() -> FakeVNCClient:
    return FakeVNCClient()


async def _make_server() -> tuple[Any, FakeVNCClient]:
    client = FakeVNCClient()
    config = VNCConfig(host="invalid.invalid", port=1)
    server = await create_mcp_server(config)
    # Inject the fake client directly: lazy-connect is bypassed, no network.
    mcp_module._vnc_client = client
    mcp_module._vnc_config = config
    return server, client


@pytest.fixture
async def server_and_client(anyio_backend: str) -> Any:
    server, client = await _make_server()
    yield server, client
    mcp_module._vnc_client = None
    mcp_module._vnc_config = None


@pytest.fixture
async def call(server_and_client: Any) -> Any:
    """Call a registered tool by name, unwrapping UnexpectedToolError causes."""

    from mcp.server.mcpserver.exceptions import UnexpectedToolError

    server, _ = server_and_client

    async def _call(name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await server.call_tool(name, arguments)
        except UnexpectedToolError as exc:
            raise exc.__cause__ from exc  # surface the original validation error

    return _call


def make_caller(server: Any) -> Callable[..., Any]:
    """Call a registered tool by name, unwrapping UnexpectedToolError causes."""

    async def call(name: str, arguments: dict[str, Any]) -> Any:
        from mcp.server.mcpserver.exceptions import UnexpectedToolError

        try:
            return await server.call_tool(name, arguments)
        except UnexpectedToolError as exc:
            raise exc.__cause__ from exc  # surface the original validation error

    return call


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


__all__ = (
    "FakeOCREngine",
    "FakeVNCClient",
    "anyio_backend",
    "fake_client",
    "make_caller",
    "server_and_client",
)
