"""vnc-mcp

Copyright (C) 2025 Parker Wahle

SPDX-License-Identifier: AGPL-3.0-or-later
"""  # noqa: E501, D415

from __future__ import annotations

import asyncio
import functools
import logging
import os
import sys
import time
from functools import lru_cache
from io import BytesIO
from typing import Any
from typing import Literal

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver import Image as MCPImage
from PIL import Image as PILImage
from pyvnc import Point
from pyvnc import Rect
from pyvnc import VNCClient as AsyncVNCClient
from pyvnc import VNCConfig
from pyvnc import key_codes


logger = logging.getLogger(__package__)

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VAR = "VNCMCP_LOG_LEVEL"
MAX_LOGGED_ARG_LENGTH = 200

_logging_configured = False


def parse_log_level(level: str | None) -> tuple[int, str | None]:
    """Parse a level name case-insensitively; return (effective level, warning or None).

    ``None`` or an empty string yields the default INFO level. Invalid values
    also fall back to INFO, with a warning message the caller should log once
    logging has been configured.
    """
    if level is None or not level.strip():
        return logging.INFO, None
    resolved = logging.getLevelName(level.strip().upper())
    if isinstance(resolved, int):
        return resolved, None
    return (
        logging.INFO,
        f"Invalid {LOG_LEVEL_ENV_VAR} value {level!r}; falling back to {DEFAULT_LOG_LEVEL}.",
    )


def configure_logging(level: str | None = None) -> int:
    """Configure root logging to stderr, exactly once.

    The stream must be stderr, never stdout: stdout is the MCP stdio protocol
    channel and any stray output there corrupts the handshake. The ``mcp`` and
    ``pyvnc`` loggers are deliberately left inheriting the root config so their
    output (e.g. pyvnc listener death) is visible.

    Args:
        level: explicit level name (the CLI ``--log-level`` flag). It overrides
            the ``VNCMCP_LOG_LEVEL`` environment variable. ``None`` means use
            the env var, defaulting to INFO.

    Returns:
        The effective numeric level that was applied.

    """
    global _logging_configured

    effective_level, warning = parse_log_level(level or os.environ.get(LOG_LEVEL_ENV_VAR))

    root = logging.getLogger()
    if not _logging_configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(effective_level)
        _logging_configured = True
    else:
        root.setLevel(effective_level)

    if warning:
        logger.warning(warning)

    return effective_level


def _truncate_for_log(payload: str, limit: int = MAX_LOGGED_ARG_LENGTH) -> str:
    return payload if len(payload) <= limit else payload[: limit - 1] + "…"


def _logged_tool(fn: Any) -> Any:
    """Decorator adding entry/exit/exception DEBUG-ERROR logging to a tool.

    Applied below ``mcp_server.tool()`` so the MCP schema is still generated
    from the original signature (``functools.wraps`` preserves ``__wrapped__``).
    ``logger.exception`` here is the single choke point where every tool error
    is recorded with a full traceback before the model sees a clean message.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = fn.__name__
        payload = _truncate_for_log(repr(kwargs) if kwargs else repr(args))
        logger.debug("%s: entry, args=%s", name, payload)
        start = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            logger.exception("%s: failed after %.1fms", name, (time.perf_counter() - start) * 1000)
            raise
        logger.debug("%s: exit, %.1fms", name, (time.perf_counter() - start) * 1000)
        return result

    return wrapper

MAX_CLICK_REPEAT = 50
MAX_SCROLL_REPEAT = 50
MAX_MOUSE_BUTTON = 7  # RFB pointer-event button mask is a single byte

STALE_WARNING_LINE = (
    "[stale] server returned no framebuffer changes for a forced full-frame request; "
    "this view may be outdated — retry this tool, or call reconnect_vnc"
)

TOOL_INSTRUCTIONS = (
    "This VNC client can be used to view the contents of and control the user's computer. "
    "Please, use it responsibly. "
    "Prefer the OCR tools (get_text_from_screen and get_text_from_rectangle_of_screen) to inspect "
    "screen state: they return text with normalized bounding boxes so you can pick click targets "
    "without receiving images. Only use the image tools when you genuinely need to see visual "
    "layout, color, or icons."
)


class VNCNotConnectedError(RuntimeError):
    """Raised when a tool is used but the VNC client is not connected."""


_vnc_client: AsyncVNCClient | None = None
_vnc_config: VNCConfig | None = None
_vnc_connect_lock = asyncio.Lock()


async def _get_client() -> AsyncVNCClient:
    """Lazily connect to the VNC server on first use.

    Connecting at server startup would kill the whole MCP server (and the
    handshake with the MCP client) if the VNC server refuses the connection.
    Instead, the connection is established on the first tool call and
    connection failures surface as clean tool errors. The pyvnc fork
    reconnects automatically afterwards (``auto_reconnect`` defaults to True).
    """
    global _vnc_client

    assert _vnc_config is not None
    if _vnc_client is None:
        async with _vnc_connect_lock:
            if _vnc_client is None:
                config = _vnc_config
                try:
                    _vnc_client = await AsyncVNCClient.connect(config)
                except Exception:
                    logger.warning(
                        "Failed to connect to VNC server at %s:%s.",
                        config.host,
                        config.port,
                        exc_info=True,
                    )
                    raise
                logger.info(
                    "Connected to VNC server at %s:%s (framebuffer %sx%s).",
                    config.host,
                    config.port,
                    _vnc_client.rect.width,
                    _vnc_client.rect.height,
                )
    return _vnc_client


def _require_client() -> AsyncVNCClient:
    """Return the client if connected, else raise a clean tool error."""
    if _vnc_client is None or not _vnc_client.is_connected:
        raise VNCNotConnectedError(
            "Not connected to a VNC server. The connection may have failed or been closed; "
            "retrying the tool may re-establish it."
        )
    return _vnc_client


def _validate_coordinate(client: AsyncVNCClient, name: str, value: int, axis_size: int) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value >= axis_size:
        raise ValueError(
            f"{name}={value} is outside the framebuffer bounds (0..{axis_size - 1}); "
            f"the framebuffer is {client.rect.width}x{client.rect.height}. "
            "Use get_screen_resolution to check the workspace size."
        )


def _validate_point(client: AsyncVNCClient, x: int, y: int) -> None:
    _validate_coordinate(client, "x", x, client.rect.width)
    _validate_coordinate(client, "y", y, client.rect.height)


def _validate_rect(client: AsyncVNCClient, top_left_x: int, top_left_y: int, width: int, height: int) -> None:
    _validate_coordinate(client, "top_left_x", top_left_x, client.rect.width)
    _validate_coordinate(client, "top_left_y", top_left_y, client.rect.height)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got width={width}, height={height}")
    if top_left_x + width > client.rect.width or top_left_y + height > client.rect.height:
        raise ValueError(
            f"Crop rectangle ({top_left_x}, {top_left_y}, {width}x{height}) extends past the "
            f"framebuffer bounds {client.rect.width}x{client.rect.height}."
        )


def _validate_mouse_button(button: int) -> None:
    if not isinstance(button, int) or not 0 <= button <= MAX_MOUSE_BUTTON:
        raise ValueError(
            f"mouse_button must be an integer in 0..{MAX_MOUSE_BUTTON}, got {button!r}. "
            "0=left, 1=middle, 2=right, 3=scroll up, 4=scroll down."
        )


def _validate_click_repeat(n: int, *, maximum: int = MAX_CLICK_REPEAT) -> None:
    if not isinstance(n, int) or not 1 <= n <= maximum:
        raise ValueError(f"n must be an integer in 1..{maximum}, got {n!r}")


def _validate_keys(*keys: str) -> None:
    for key in keys:
        if key not in key_codes:
            raise ValueError(
                f"Unknown key {key!r}. Use keysymdef-style names "
                "(e.g. 'a', 'Delete', 'Escape', 'Control_L', 'Shift_L', 'BackSpace', 'space')."
            )


def _validate_string(string: str) -> None:
    untypeable = sorted({char for char in string if char not in key_codes})
    if untypeable:
        raise ValueError(
            "The string contains characters that cannot be typed over VNC "
            f"(no keysym mapping): {untypeable!r}. Nothing was sent to the server."
        )


INPUT_PENDING_WARNING_LINE = (
    "[input-pending] this frame was last updated before the most recent input "
    "sent through this server — the view may not reflect your last action; "
    "re-read to confirm"
)


def _apply_stale_flag(client: AsyncVNCClient, result: Any) -> Any:
    """Surface pyvnc's stale-capture flag in tool output.

    Text results get the warning appended as a final line; non-text results
    (images) are returned as ``[result, warning]`` so both blocks reach the
    model (mcp 2.x converts lists to content blocks).

    Also appends the input-pending advisory when the last framebuffer update
    was parsed before the most recent key/mouse event we sent: the server may
    be handing us queued pre-input frames.
    """
    advisories: list[str] = []
    if getattr(client, "last_capture_stale", False):
        advisories.append(STALE_WARNING_LINE)
    input_time = getattr(client, "last_input_time", 0.0)
    update_time = getattr(client, "last_update_time", 0.0)
    if input_time and update_time and update_time < input_time:
        advisories.append(INPUT_PENDING_WARNING_LINE)
    if not advisories:
        return result
    for line in advisories:
        if isinstance(result, str):
            result = f"{result}\n{line}"
        else:
            result = [result, line] if not isinstance(result, list) else [*result, line]
    return result


def _rgba_to_pil(array: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(array, "RGBA")


def _prepare_image(
    array: np.ndarray,
    *,
    max_dimension: int | None,
    format: Literal["png", "jpeg"],
) -> MCPImage:
    pilimage = _rgba_to_pil(array)
    if max_dimension is not None:
        if max_dimension <= 0:
            raise ValueError(f"max_dimension must be positive, got {max_dimension}")
        pilimage.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
    with BytesIO() as bio:
        if format == "jpeg":
            # JPEG cannot store an alpha channel.
            pilimage = pilimage.convert("RGB")
            pilimage.save(bio, "jpeg", quality=80)
        else:
            pilimage.save(bio, "png")
        return MCPImage(data=bio.getvalue(), format=format)


@lru_cache(maxsize=1)
def _get_ocr_engine() -> Any:
    """Create the RapidOCR engine once, on first use (loads models)."""
    from rapidocr import RapidOCR  # noqa: PLC0415 (heavy import; deferred)

    return RapidOCR()


def _run_ocr(array: np.ndarray) -> str:
    """Run RapidOCR on an RGBA framebuffer array and return lines with normalized boxes.

    Each detected region is formatted as ``[x0, y0, x1, y1] text`` where the
    coordinates are floats in 0..1 relative to the captured image, so the
    calling model can compute absolute click targets with any image size.
    """
    engine = _get_ocr_engine()
    rgb = np.asarray(_rgba_to_pil(array).convert("RGB"))
    start = time.perf_counter()
    result = engine(rgb)
    elapsed_ms = (time.perf_counter() - start) * 1000

    height, width = array.shape[0], array.shape[1]
    if result.boxes is None or result.txts is None:
        logger.info("OCR found no text regions in %.1fms.", elapsed_ms)
        return ""

    lines: list[str] = []
    for box, text in zip(result.boxes, result.txts, strict=False):
        xs = [float(point[0]) / width for point in box]
        ys = [float(point[1]) / height for point in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        lines.append(f"[{x0:.3f}, {y0:.3f}, {x1:.3f}, {y1:.3f}] {text}")
    logger.info(
        "OCR detected %d region(s) in %.1fms.", len(lines), elapsed_ms
    )
    return "\n".join(lines)


async def create_mcp_server(vnc_config: VNCConfig, *, log_level: str | None = None) -> MCPServer:
    """Creates an MCPServer configured for the given VNC configuration.

    The VNC connection is established lazily on the first tool call so that a
    refused connection does not kill the MCP server during startup.

    Args:
        vnc_config: connection settings (host, port, credentials).
        log_level: optional explicit level name (the CLI ``--log-level`` flag).
            It overrides the ``VNCMCP_LOG_LEVEL`` environment variable, which
            itself defaults to INFO. Logging is always on stderr.

    """
    global _vnc_config

    configure_logging(log_level)

    _vnc_config = vnc_config

    mcp_server = MCPServer(
        "VNC Client",
        instructions=TOOL_INSTRUCTIONS,
    )

    # When declaring a tool, the description will default to the docstring of the function.

    # Get screen resolution
    @mcp_server.tool()
    @_logged_tool
    async def get_screen_resolution() -> str:
        """Returns a string containing the width times height of the VNC session's workspace.

        Please note that this may not actually be the real resolution of the user's computer monitor.

        This resolution should be used for all calls to tools that take a resolution or position.
        """
        client = await _get_client()
        return f"{client.rect.width}x{client.rect.height}"

    @mcp_server.tool()
    @_logged_tool
    async def get_desktop_name() -> str:
        """Returns the name of the current environment reported by the server.
        """
        client = await _get_client()
        return client.desktop_name

    @mcp_server.tool()
    @_logged_tool
    async def get_connection_status() -> str:
        """Returns the current VNC connection status: whether the client is connected,
        the desktop name, and the framebuffer resolution.
        """
        if _vnc_client is None:
            return "Not connected yet (the connection is established on the first tool call)."
        if _vnc_client.is_connected:
            return (
                f"Connected. Desktop name: {_vnc_client.desktop_name!r}. "
                f"Framebuffer: {_vnc_client.rect.width}x{_vnc_client.rect.height}."
            )
        detail = f" Last error: {_vnc_client.last_error!r}." if _vnc_client.last_error else ""
        return f"Disconnected.{detail} The client will attempt to reconnect automatically."
    # Whole screen image
    @mcp_server.tool()
    @_logged_tool
    async def get_screen(
        max_dimension: int | None = None,
        format: Literal["png", "jpeg"] = "png",
    ) -> MCPImage:
        """Gets an image of the entire VNC session's workspace.

        Prefer the OCR tools for reading text: they are cheaper and return click targets.
        Use this tool when you genuinely need to see the visual layout, colors, or icons.

        max_dimension (optional): downscale the image so its longest side is at most this
        many pixels, preserving the aspect ratio. Useful to reduce payload size.
        format: "png" (lossless) or "jpeg" (smaller, quality 80).
        """
        client = await _get_client()
        raw_rgba_array = await client.capture()
        return _apply_stale_flag(
            client, _prepare_image(raw_rgba_array, max_dimension=max_dimension, format=format)
        )

    @mcp_server.tool()
    @_logged_tool
    async def get_text_from_screen() -> str:
        """Reads all visible text on the screen using OCR. This is the DEFAULT way to
        inspect screen state: prefer it over image tools.

        Returns one line per detected text region, formatted as
        "[x0, y0, x1, y1] text" where the coordinates are floats in 0..1 relative
        to the captured image. Multiply by the workspace resolution (see
        get_screen_resolution) to get pixel coordinates for clicking.

        If you are only interested in a certain subrectangle of the workspace,
        consider using the get_text_from_rectangle_of_screen tool.
        """
        client = await _get_client()
        raw_rgba_array = await client.capture()
        text = await asyncio.to_thread(_run_ocr, raw_rgba_array)
        return _apply_stale_flag(client, text)

    # Relative rectangle image
    @mcp_server.tool()
    @_logged_tool
    async def get_rectangle_of_screen(
        top_left_x: int,
        top_left_y: int,
        width: int,
        height: int,
        max_dimension: int | None = None,
        format: Literal["png", "jpeg"] = "png",
    ) -> MCPImage:
        """Captures a subrectangle of the VNC workspace and returns it as an image.

        Prefer the OCR tools for reading text; use this tool when you genuinely
        need to see the visual layout, colors, or icons of a region.

        Capturing a subrectangle is more performant than capturing the entire
        workspace, and should be preferred where possible.

        max_dimension (optional): downscale the image so its longest side is at most this
        many pixels, preserving the aspect ratio. Useful to reduce payload size.
        format: "png" (lossless) or "jpeg" (smaller, quality 80).
        """
        client = await _get_client()
        _validate_rect(client, top_left_x, top_left_y, width, height)
        rect = Rect(top_left_x, top_left_y, width, height)
        raw_rgba_array = await client.capture(rect)
        return _apply_stale_flag(
            client, _prepare_image(raw_rgba_array, max_dimension=max_dimension, format=format)
        )

    @mcp_server.tool()
    @_logged_tool
    async def get_text_from_rectangle_of_screen(
        top_left_x: int, top_left_y: int, width: int, height: int
    ) -> str:
        """Reads the text in a subrectangle of the VNC workspace using OCR. This is the
        DEFAULT way to inspect screen state: prefer it over image tools.

        Returns one line per detected text region, formatted as
        "[x0, y0, x1, y1] text" where the coordinates are floats in 0..1 relative
        to the captured rectangle. To get workspace pixel coordinates, add the
        rectangle's origin and scale by the rectangle's size.

        If you would like the text from the entire screen, you should use the
        get_text_from_screen tool.
        """
        client = await _get_client()
        _validate_rect(client, top_left_x, top_left_y, width, height)
        rect = Rect(top_left_x, top_left_y, width, height)
        raw_rgba_array = await client.capture(rect)
        text = await asyncio.to_thread(_run_ocr, raw_rgba_array)
        return _apply_stale_flag(client, text)

    # Strike key(s)
    @mcp_server.tool()
    @_logged_tool
    async def strike_keys(keys: list[str]) -> str:
        """Strikes the keys inputted without any delay between strikes nor any modifier keys held.

        The format for keys is similar to keysymdef.py in Python.
        Alphanumeric characters (0-9a-z) are supported, although only lowercase keys may be pressed.
        To input an uppercase letter, it must either be typed with a shift modifier ot inputted using a tool
        that writes a string into the session as opposed to typing a key.

        List of example modifiers in their correct format:
            Delete
            Escape
            Super_L
            Alt_L
            Control_L
            Shift_L
            BackSpace
            space

        The "right" versions of keys are also acceptable for versions of keys that are "sided," like most modifiers.

        Keys are released in the reverse order they are input. The final key in the array will always be typed with all
        other keys pressed.
        """
        client = await _get_client()
        _validate_keys(*keys)
        await client.press(*keys)
        return "Successfully struck " + ", ".join(keys)

    # Write string
    @mcp_server.tool()
    @_logged_tool
    async def write_string(
        string: str,
    ) -> str:
        """Directly writes a string into the VNC session. A textbox or similar text-accepting appliance must first
        be selected using the mouse tool(s) for a likely effect.

        Because the string is inputted directly, it may not trigger actions in softwares that expect some delay
        between keypresses. Be mindful of this when using this tool.

        However, it is likely to work in most textboxes. Because if its efficiency, it should be preferred over pressing
        keys and waiting. You may always try it first and then evaluate alternatives.

        Characters without a keysym mapping (e.g. emoji) are rejected before anything is sent.
        """
        client = await _get_client()
        _validate_string(string)
        await client.write(string)
        return "Successfully typed " + string

    # Strike key(s) with key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def strike_keys_with_keys_held(
        keys_to_strike: list[str],
        keys_to_hold: list[str],
    ) -> str:
        """Strikes the provided keys while holding the provided keys. Useful for entering keyboard shortcuts.
        The keyboard shortcut "CTRL+X" may be entered with "keys to strike" as "X," and "keys to hold" as "Control_L."

        The format for keys is similar to keysymdef.py in Python.
        Alphanumeric characters (0-9a-z) are supported, although only lowercase keys may be pressed.
        To input an uppercase letter, it must either be typed with a shift modifier ot inputted using a tool
        that writes a string into the session as opposed to typing a key.

        List of example modifiers in their correct format:
            Delete
            Escape
            Super_L
            Alt_L
            Control_L
            Shift_L
            BackSpace
            space

        The "right" versions of keys are also acceptable for versions of keys that are "sided," like most modifiers.

        Keys are released in the reverse order they are input. The final key in the array will always be typed with all
        other keys pressed.
        """
        client = await _get_client()
        _validate_keys(*keys_to_strike, *keys_to_hold)
        async with client.hold_key(*keys_to_hold):
            await client.press(*keys_to_strike)

        return f"Successfully struck {', '.join(keys_to_strike)} while holding {', '.join(keys_to_hold)}"

    # Write string with key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def write_string_with_keys_held(
        string_to_write: str,
        keys_to_hold: list[str],
    ) -> str:
        """Directly writes a string into the VNC session. A textbox or similar text-accepting appliance must first
        be selected using the mouse tool(s) for a likely effect.

        Because the string is inputted directly, it may not trigger actions in softwares that expect some delay
        between keypresses. Be mindful of this when using this tool.

        However, it is likely to work in most textboxes. Because if its efficiency, it should be preferred over pressing
        keys and waiting. You may always try it first and then evaluate alternatives.

        The format for keys is similar to keysymdef.py in Python.
        Alphanumeric characters (0-9a-z) are supported, although only lowercase keys may be pressed.
        To input an uppercase letter, it must either be typed with a shift modifier ot inputted using a tool
        that writes a string into the session as opposed to typing a key.

        List of example modifiers in their correct format:
            Delete
            Escape
            Super_L
            Alt_L
            Control_L
            Shift_L
            BackSpace
            space

        The "right" versions of keys are also acceptable for versions of keys that are "sided," like most modifiers.

        Keys are released in the reverse order they are input. The final key in the array will always be typed with all
        other keys pressed.

        Characters without a keysym mapping (e.g. emoji) are rejected before anything is sent.
        """
        client = await _get_client()
        _validate_keys(*keys_to_hold)
        _validate_string(string_to_write)
        async with client.hold_key(*keys_to_hold):
            await client.write(string_to_write)

        return f"Successfully wrote {string_to_write} while holding {', '.join(keys_to_hold)}"

    # Strike key(s) with mouse button held
    @mcp_server.tool()
    @_logged_tool
    async def strike_keys_with_mouse_button_held(
        keys_to_strike: list[str], mouse_button_to_hold: int
    ) -> str:
        """Strikes a number of keys while holding a certain mouse button.

        Keys are released in the reverse order they are input. The final key in the array will always be typed with all
        other keys pressed.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4
        """
        client = await _get_client()
        _validate_keys(*keys_to_strike)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            await client.press(*keys_to_strike)

        return f"Successfully struck {', '.join(keys_to_strike)} while holding mouse button {mouse_button_to_hold}"

    # Write string with mouse button held
    @mcp_server.tool()
    @_logged_tool
    async def write_string_with_mouse_button_held(
        string_to_write: str, mouse_button_to_hold: int
    ) -> str:
        """Writes a string while holding a certain mouse button.

        Same rules apply as other tools that write strings.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4
        """
        client = await _get_client()
        _validate_string(string_to_write)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            await client.write(string_to_write)

        return f"Successfully wrote {string_to_write} while holding mouse button {mouse_button_to_hold}"

    # Strike key(s) with mouse button held and key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def strike_keys_with_mouse_button_and_keys_held(
        keys_to_strike: list[str],
        mouse_button_to_hold: int,
        keys_to_hold: list[str],
    ) -> str:
        """Strikes a number of keys while holding a certain mouse button and a number of other keys.

        Keys are released in the reverse order they are input. The final key in the array will always be typed with all
        other keys pressed.

        Same rules apply to the modifier keys as the strike_keys_with_keys_held tool.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4
        """
        client = await _get_client()
        _validate_keys(*keys_to_strike, *keys_to_hold)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            async with client.hold_key(*keys_to_hold):
                await client.press(*keys_to_strike)

        return f"Successfully struck {', '.join(keys_to_strike)} while holding mouse button {mouse_button_to_hold} and keys {', '.join(keys_to_hold)}"

    # Write string with mouse button held and key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def write_string_with_mouse_button_and_keys_held(
        string_to_write: str,
        mouse_button_to_hold: int,
        keys_to_hold: list[str],
    ) -> str:
        """Writes a string while holding a certain mouse button and a number of other keys.

        Same rules apply to the modifier keys as the write_string_with_keys_held tool.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4

        Characters without a keysym mapping (e.g. emoji) are rejected before anything is sent.
        """
        client = await _get_client()
        _validate_string(string_to_write)
        _validate_keys(*keys_to_hold)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            async with client.hold_key(*keys_to_hold):
                await client.write(string_to_write)

        return f"Successfully wrote {string_to_write} while holding mouse button {mouse_button_to_hold} and keys {', '.join(keys_to_hold)}"

    # Move mouse from (x,y) to (w,z) with mouse button held
    @mcp_server.tool()
    @_logged_tool
    async def move_mouse_with_mouse_button_held(
        start_x: int, start_y: int, end_x: int, end_y: int, mouse_button_to_hold: int
    ) -> str:
        """Moves the mouse from (start_x, start_y) to (end_x, end_y) while holding a certain mouse button.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4
        """
        client = await _get_client()
        _validate_point(client, start_x, start_y)
        _validate_point(client, end_x, end_y)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            await client.move(Point(start_x, start_y))
            await client.move(Point(end_x, end_y))

        return f"Successfully moved mouse from ({start_x}, {start_y}) to ({end_x}, {end_y}) while holding mouse button {mouse_button_to_hold}"

    # Move mouse from (x,y) to (w,z) with key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def move_mouse_with_keys_held(
        start_x: int, start_y: int, end_x: int, end_y: int, keys_to_hold: list[str]
    ) -> str:
        """Moves the mouse from (start_x, start_y) to (end_x, end_y) while holding a number of keys.

        Same rules apply to the modifier keys as the strike_keys_with_keys_held tool.
        """
        client = await _get_client()
        _validate_point(client, start_x, start_y)
        _validate_point(client, end_x, end_y)
        _validate_keys(*keys_to_hold)
        async with client.hold_key(*keys_to_hold):
            await client.move(Point(start_x, start_y))
            await client.move(Point(end_x, end_y))

        return f"Successfully moved mouse from ({start_x}, {start_y}) to ({end_x}, {end_y}) while holding keys {', '.join(keys_to_hold)}"

    # Move mouse from (x,y) to (w,z) with key(s) and mouse button held
    @mcp_server.tool()
    @_logged_tool
    async def move_mouse_with_keys_and_mouse_button_held(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        keys_to_hold: list[str],
        mouse_button_to_hold: int,
    ) -> str:
        """Moves the mouse from (start_x, start_y) to (end_x, end_y) while holding a number of keys and a certain mouse button.

        Same rules apply to the modifier keys as the strike_keys_with_keys_held tool.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4
        """
        client = await _get_client()
        _validate_point(client, start_x, start_y)
        _validate_point(client, end_x, end_y)
        _validate_keys(*keys_to_hold)
        _validate_mouse_button(mouse_button_to_hold)
        async with client.hold_mouse(mouse_button_to_hold):
            async with client.hold_key(*keys_to_hold):
                await client.move(Point(start_x, start_y))
                await client.move(Point(end_x, end_y))

        return f"Successfully moved mouse from ({start_x}, {start_y}) to ({end_x}, {end_y}) while holding keys {', '.join(keys_to_hold)} and mouse button {mouse_button_to_hold}"

    # Move mouse to (x,y)
    @mcp_server.tool()
    @_logged_tool
    async def move_mouse_to(x: int, y: int) -> str:
        """Moves the mouse to the coordinates (x, y) in the VNC session's workspace.

        In order to click on a specific point on the screen, you should use this tool to move the mouse to the desired position and then use the click_at_current_position tool.
        """
        client = await _get_client()
        _validate_point(client, x, y)
        await client.move(Point(x, y))
        return f"Successfully moved mouse to ({x}, {y})"

    # Click (n) times at current position
    @mcp_server.tool()
    @_logged_tool
    async def click_at_current_position(mouse_button: int, n: int) -> str:
        """Clicks the mouse at the current position n times (n is capped at 50).

        This can be used to perform a double-click or triple-click, for example.

        This can also be used to scroll the mouse wheel by clicking the scroll buttons.
        In the case that this is used to scroll, n is the number of lines to scroll.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4

        The current position is the position of the mouse in the VNC session's workspace.
        """
        client = await _get_client()
        _validate_mouse_button(mouse_button)
        _validate_click_repeat(n)
        for _ in range(n):
            await client.click(mouse_button)

        return f"Successfully clicked {n} times at current position"

    # Click (n) times at current position with key(s) held
    @mcp_server.tool()
    @_logged_tool
    async def click_at_current_position_with_keys_held(
        mouse_button: int, n: int, keys_to_hold: list[str]
    ) -> str:
        """Clicks the mouse at the current position n times (n is capped at 50) while holding a number of keys.

        Same rules apply to the modifier keys as the strike_keys_with_keys_held tool.

        This can be used to perform a double-click or triple-click, for example.

        This can also be used to scroll the mouse wheel by clicking the scroll buttons.
        In the case that this is used to scroll, n is the number of lines to scroll.

        This tool may be used for actions such as a CTRL+click, SHIFT+click, or CTRL+Scroll.

        Mouse buttons are as follows:
        MOUSE_BUTTON_LEFT = 0
        MOUSE_BUTTON_MIDDLE = 1
        MOUSE_BUTTON_RIGHT = 2
        MOUSE_BUTTON_SCROLL_UP = 3
        MOUSE_BUTTON_SCROLL_DOWN = 4

        The current position is the position of the mouse in the VNC session's workspace.
        """
        client = await _get_client()
        _validate_mouse_button(mouse_button)
        _validate_click_repeat(n)
        _validate_keys(*keys_to_hold)
        async with client.hold_key(*keys_to_hold):
            for _ in range(n):
                await client.click(mouse_button)

        return f"Successfully clicked {n} times at current position while holding keys {', '.join(keys_to_hold)}"

    # Reconnect
    @mcp_server.tool()
    @_logged_tool
    async def reconnect_vnc() -> str:
        """Forces a fresh VNC reconnect: discards the current stream, re-handshakes,
        and resets all server-side client state.

        Use this when tools report a stale view (the "[stale]" marker in OCR or
        image output) or when the remote server's update pipeline appears wedged:
        a reconnect is the reliable cure for a stuck dirty-region pipeline.
        """
        client = await _get_client()
        success = await client.reconnect()
        if success:
            # The first update after a reconnect can be the server's queued
            # pre-reconnect damage (partial/blank), which would hand the caller
            # a stale-looking frame. Absorb it: one throwaway capture so the
            # caller's next read reflects the live screen.
            try:
                await client.capture()
            except (ConnectionError, RuntimeError):
                pass
            return (
                f"Reconnected to {client.desktop_name!r}; framebuffer "
                f"{client.rect.width}x{client.rect.height}."
            )
        return f"Reconnection failed (last error: {client.last_error!r})."

    # Clipboard
    @mcp_server.tool()
    @_logged_tool
    async def set_clipboard(text: str) -> str:
        """Sets the guest machine's clipboard to *text* (RFB ClientCutText).

        Paste it into a console afterwards with strike_keys(["Control_L", "v"])
        (or Shift+Insert, the most universal paste in Windows consoles). This
        is the preferred way to enter long or special-character-laden text:
        it is lossless and instant, unlike write_string which synthesizes one
        keystroke per character. Caveats: overwrites whatever is currently on
        the guest clipboard; multi-line text pastes into consoles execute
        line-by-line (each newline acts as Enter).
        """
        client = await _get_client()
        await client.set_clipboard(text)
        return f"Clipboard set to {len(text)} characters. Paste with strike_keys([\"Control_L\", \"v\"]) or Shift+Insert."

    @mcp_server.tool()
    @_logged_tool
    async def get_clipboard(wait_seconds: float = 0.0) -> str:
        """Returns the guest machine's clipboard text as pushed by the server
        (ServerCutText). Empty string result means nothing was received yet.

        Workflow for reading long console output exactly (no OCR): run the
        command as `command | clip` (Windows) in the guest console, then call
        this with wait_seconds 2-3 to wait for the clipboard push. Only
        receives text pushed while this MCP server is connected. Binary and
        non-text clipboard formats are not representable over RFB.
        """
        if not 0 <= wait_seconds <= 30:
            raise ValueError(f"wait_seconds must be 0..30, got {wait_seconds}")
        client = await _get_client()
        t0 = time.monotonic()
        while wait_seconds > 0 and client.last_cut_text_time <= t0:
            if time.monotonic() - t0 >= wait_seconds:
                break
            await asyncio.sleep(0.2)
        if client.last_cut_text is None:
            return (
                "(clipboard empty — no ServerCutText received this session; "
                "put text on the guest clipboard, e.g. run `command | clip`)"
            )
        return client.last_cut_text

    # Scroll the mouse wheel
    @mcp_server.tool()
    @_logged_tool
    async def scroll_up(repeat: int = 1) -> str:
        """Scrolls the mouse wheel upwards (content moves down) at the current mouse position,
        repeat times (capped at 50). In most applications, each repeat scrolls a few lines.
        """
        client = await _get_client()
        _validate_click_repeat(repeat, maximum=MAX_SCROLL_REPEAT)
        await client.scroll_up(repeat)
        return f"Successfully scrolled up {repeat} time(s)"

    @mcp_server.tool()
    @_logged_tool
    async def scroll_down(repeat: int = 1) -> str:
        """Scrolls the mouse wheel downwards (content moves up) at the current mouse position,
        repeat times (capped at 50). In most applications, each repeat scrolls a few lines.
        """
        client = await _get_client()
        _validate_click_repeat(repeat, maximum=MAX_SCROLL_REPEAT)
        await client.scroll_down(repeat)
        return f"Successfully scrolled down {repeat} time(s)"

    return mcp_server


__all__ = ("VNCNotConnectedError", "create_mcp_server")
