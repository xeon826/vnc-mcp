"""Tests for the MCP tools: registration, validation, OCR formatting."""

from __future__ import annotations

import asyncio
import unittest.mock
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

import vnc_mcp.mcp as mcp_module


pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = {
    "get_screen_resolution",
    "get_desktop_name",
    "get_connection_status",
    "get_screen",
    "get_text_from_screen",
    "get_rectangle_of_screen",
    "get_text_from_rectangle_of_screen",
    "strike_keys",
    "write_string",
    "strike_keys_with_keys_held",
    "write_string_with_keys_held",
    "strike_keys_with_mouse_button_held",
    "write_string_with_mouse_button_held",
    "strike_keys_with_mouse_button_and_keys_held",
    "write_string_with_mouse_button_and_keys_held",
    "move_mouse_with_mouse_button_held",
    "move_mouse_with_keys_held",
    "move_mouse_with_keys_and_mouse_button_held",
    "move_mouse_to",
    "click_at_current_position",
    "click_at_current_position_with_keys_held",
    "scroll_up",
    "scroll_down",
    "reconnect_vnc",
}


def get_text(result: Any) -> str:
    """Extract the text payload from a CallToolResult."""
    from mcp.types import TextContent

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def get_image(result: Any) -> tuple[bytes, str]:
    """Extract (raw bytes, mime type) from a CallToolResult image content."""
    import base64

    from mcp.types import ImageContent

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ImageContent)
    return base64.b64decode(content.data), content.mime_type


class TestToolRegistration:
    async def test_all_expected_tools_registered(self, server_and_client: Any) -> None:
        server, _ = server_and_client
        names = {tool.name for tool in await server.list_tools()}
        missing = EXPECTED_TOOLS - names
        assert not missing, f"missing tools: {missing}"

    async def test_removed_tools_not_registered(self, server_and_client: Any) -> None:
        server, _ = server_and_client
        names = {tool.name for tool in await server.list_tools()}
        # The old v1 tool names should be gone.
        assert "get_whole_screen_image" not in names
        assert "get_text_from_whole_screen_image" not in names


class TestScreenState:
    async def test_screen_resolution(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        assert "200x100" in get_text(await call("get_screen_resolution", {}))

    async def test_desktop_name(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        assert "fake-desktop" in get_text(await call("get_desktop_name", {}))

    async def test_connection_status_connected(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        assert "Connected" in get_text(await call("get_connection_status", {}))

    async def test_connection_status_not_connected(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        mcp_module._vnc_client = None
        assert "Not connected" in get_text(await call("get_connection_status", {}))


class TestValidation:
    async def test_negative_x_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="non-negative"):
            await call("move_mouse_to", {"x": -1, "y": 5})

    async def test_out_of_bounds_x_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="outside the framebuffer bounds"):
            await call("move_mouse_to", {"x": 500, "y": 5})

    async def test_out_of_bounds_y_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="outside the framebuffer bounds"):
            await call("move_mouse_to", {"x": 5, "y": 100})

    async def test_crop_out_of_bounds_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="extends past the framebuffer"):
            await call(
                "get_rectangle_of_screen",
                {"top_left_x": 150, "top_left_y": 0, "width": 100, "height": 10},
            )

    async def test_crop_zero_size_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="positive"):
            await call(
                "get_rectangle_of_screen",
                {"top_left_x": 0, "top_left_y": 0, "width": 0, "height": 10},
            )

    async def test_unknown_key_rejected_before_send(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="Unknown key"):
            await call("strike_keys", {"keys": ["NotARealKey"]})
        assert client.pressed_keys == []

    async def test_partial_valid_key_list_rejected(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="Unknown key"):
            await call("strike_keys", {"keys": ["a", "NotARealKey"]})
        assert client.pressed_keys == []

    async def test_untypeable_string_rejected_before_send(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="cannot be typed"):
            await call("write_string", {"string": "ok \U0001f600"})
        assert client.written_strings == []

    async def test_mouse_button_out_of_range_rejected(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="mouse_button"):
            await call(
                "click_at_current_position", {"mouse_button": 8, "n": 1}
            )
        assert client.clicks == []

    async def test_mouse_button_negative_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="mouse_button"):
            await call(
                "click_at_current_position", {"mouse_button": -1, "n": 1}
            )

    async def test_click_repeat_capped(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="1..50"):
            await call(
                "click_at_current_position", {"mouse_button": 0, "n": 51}
            )
        assert client.clicks == []

    async def test_click_repeat_non_positive_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="1..50"):
            await call(
                "click_at_current_position", {"mouse_button": 0, "n": 0}
            )

    async def test_scroll_repeat_capped(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        with pytest.raises(ValueError, match="1..50"):
            await call("scroll_down", {"repeat": 51})
        assert client.scrolled == []


class TestHappyPaths:
    async def test_write_string(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        await call("write_string", {"string": "hello"})
        assert client.written_strings == ["hello"]

    async def test_scroll_tools(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        await call("scroll_up", {"repeat": 3})
        await call("scroll_down", {"repeat": 2})
        assert client.scrolled == [("up", 3), ("down", 2)]

    async def test_keyboard_shortcut(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        await call(
            "strike_keys_with_keys_held",
            {"keys_to_strike": ["x"], "keys_to_hold": ["Control_L"]},
        )
        assert client.pressed_keys == [("x",)]

    async def test_click_at_position(self, server_and_client: Any, call: Any) -> None:
        server, client = server_and_client
        await call("move_mouse_to", {"x": 10, "y": 20})
        await call(
            "click_at_current_position", {"mouse_button": 0, "n": 2}
        )
        assert client.moves == [(10, 20)]
        assert client.clicks == [0, 0]


class TestOcr:
    async def test_bbox_output_normalized(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        server, client = server_and_client
        boxes = np.array([[[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]]])
        monkeypatch.setattr(mcp_module, "_get_ocr_engine", lambda: _FakeEngine(boxes, ("Hello",)))
        result = get_text(await call("get_text_from_screen", {}))
        # 200x100 image: box (10,10)-(50,30) normalizes to (0.050, 0.100, 0.250, 0.300)
        assert result == "[0.050, 0.100, 0.250, 0.300] Hello"

    async def test_bbox_output_multiple_regions(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        server, _ = server_and_client
        boxes = np.array(
            [
                [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]],
                [[0.0, 50.0], [200.0, 50.0], [200.0, 100.0], [0.0, 100.0]],
            ]
        )
        monkeypatch.setattr(
            mcp_module,
            "_get_ocr_engine",
            lambda: _FakeEngine(boxes, ("Top", "Bottom")),
        )
        result = get_text(await call("get_text_from_screen", {}))
        lines = result.splitlines()
        assert lines == [
            "[0.000, 0.000, 0.500, 0.200] Top",
            "[0.000, 0.500, 1.000, 1.000] Bottom",
        ]

    async def test_no_text_detected(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        server, _ = server_and_client
        monkeypatch.setattr(mcp_module, "_get_ocr_engine", lambda: _FakeEngine(None, None))
        result = get_text(await call("get_text_from_screen", {}))
        assert result == ""

    async def test_rectangle_ocr_uses_crop(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        server, client = server_and_client
        monkeypatch.setattr(mcp_module, "_get_ocr_engine", lambda: _FakeEngine(None, None))
        await call(
            "get_text_from_rectangle_of_screen",
            {"top_left_x": 10, "top_left_y": 10, "width": 50, "height": 40},
        )
        assert len(client.capture_calls) == 1
        rect = client.capture_calls[0]
        assert (rect.x, rect.y, rect.width, rect.height) == (10, 10, 50, 40)


class _FakeEngine:
    def __init__(self, boxes: Any, txts: Any) -> None:
        self.boxes = boxes
        self.txts = txts

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        result = type("Result", (), {})()
        result.boxes = self.boxes
        result.txts = self.txts
        return result


class TestImageTools:
    async def test_screen_png(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        data, mime = get_image(await call("get_screen", {}))
        assert mime == "image/png"
        image = PILImage.open(BytesIO(data))
        assert image.size == (200, 100)

    async def test_screen_jpeg(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        data, mime = get_image(await call("get_screen", {"format": "jpeg"}))
        assert mime == "image/jpeg"
        image = PILImage.open(BytesIO(data))
        assert image.format == "JPEG"

    async def test_screen_downscaled(self, server_and_client: Any, call: Any) -> None:
        _server, _ = server_and_client
        data, _mime = get_image(await call("get_screen", {"max_dimension": 100}))
        image = PILImage.open(BytesIO(data))
        assert max(image.size) == 100
        # aspect ratio preserved
        assert image.size == (100, 50)

    async def test_rectangle_image(self, server_and_client: Any, call: Any) -> None:
        server, _client = server_and_client
        data, _mime = get_image(
            await call(
                "get_rectangle_of_screen",
                {"top_left_x": 10, "top_left_y": 10, "width": 50, "height": 40},
            )
        )
        image = PILImage.open(BytesIO(data))
        assert image.size == (50, 40)

    async def test_invalid_max_dimension_rejected(self, server_and_client: Any, call: Any) -> None:
        server, _ = server_and_client
        with pytest.raises(ValueError, match="max_dimension"):
            await call("get_screen", {"max_dimension": 0})


class TestStaleFlag:
    async def test_stale_text_tool_appends_warning(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        server, client = server_and_client
        monkeypatch.setattr(
            mcp_module, "_get_ocr_engine", lambda: _FakeEngine(None, None)
        )
        client.last_capture_stale = True
        result = get_text(await call("get_text_from_screen", {}))
        assert result.endswith(mcp_module.STALE_WARNING_LINE)

    async def test_fresh_text_tool_no_warning(self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any) -> None:
        server, client = server_and_client
        monkeypatch.setattr(
            mcp_module, "_get_ocr_engine", lambda: _FakeEngine(None, None)
        )
        client.last_capture_stale = False
        result = get_text(await call("get_text_from_screen", {}))
        assert mcp_module.STALE_WARNING_LINE not in result

    async def test_stale_image_tool_appends_warning(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        client.last_capture_stale = True
        result = await call("get_screen", {})
        assert isinstance(result.content, list) and len(result.content) == 2
        assert result.content[1].text == mcp_module.STALE_WARNING_LINE

    async def test_fresh_image_tool_no_warning(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        client.last_capture_stale = False
        result = await call("get_screen", {})
        assert len(result.content) == 1

    async def test_stale_rectangle_text_appends_warning(self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any) -> None:
        _server, client = server_and_client
        monkeypatch.setattr(
            mcp_module, "_get_ocr_engine", lambda: _FakeEngine(None, None)
        )
        client.last_capture_stale = True
        result = get_text(
            await call(
                "get_text_from_rectangle_of_screen",
                {"top_left_x": 0, "top_left_y": 0, "width": 10, "height": 10},
            )
        )
        assert result.endswith(mcp_module.STALE_WARNING_LINE)


class TestReconnectVnc:
    async def test_reconnect_success(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        result = get_text(await call("reconnect_vnc", {}))
        assert client.reconnect_calls == 1
        # The absorbing throwaway capture: swallows the server's queued
        # pre-reconnect damage so the caller's next read is live.
        assert len(client.capture_calls) == 1
        assert "Reconnected" in result

    async def test_reconnect_absorbing_capture_failure_is_swallowed(
        self, server_and_client: Any, call: Any
    ) -> None:
        _server, client = server_and_client
        client.capture_errors = [ConnectionError("mid-resync")]
        result = get_text(await call("reconnect_vnc", {}))
        assert client.reconnect_calls == 1
        assert "Reconnected" in result

    async def test_reconnect_failure(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        client.reconnect_fails = True
        result = get_text(await call("reconnect_vnc", {}))
        assert client.reconnect_calls == 1
        assert "failed" in result

    async def test_reconnect_registered(self, server_and_client: Any) -> None:
        server, _ = server_and_client
        names = {tool.name for tool in await server.list_tools()}
        assert "reconnect_vnc" in names
        assert "get_clipboard" in names
        assert "set_clipboard" in names

    async def test_set_clipboard_sends_text(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        result = get_text(await call("set_clipboard", {"text": "long diagnostic command"}))
        assert client.clipboard_sets == ["long diagnostic command"]
        assert "23 characters" in result

    async def test_get_clipboard_empty(self, server_and_client: Any, call: Any) -> None:
        _server, client = server_and_client
        result = get_text(await call("get_clipboard", {}))
        assert "clipboard empty" in result

    async def test_get_clipboard_returns_text(
        self, server_and_client: Any, call: Any
    ) -> None:
        _server, client = server_and_client
        client.last_cut_text = "DISM output line 1\nDISM output line 2"
        client.last_cut_text_time = 123.0
        result = get_text(await call("get_clipboard", {}))
        assert result == "DISM output line 1\nDISM output line 2"

    async def test_get_clipboard_wait_returns_when_new_text_arrives(
        self, server_and_client: Any, call: Any
    ) -> None:
        """Text pushed after the call started is returned without waiting it out."""
        _server, client = server_and_client
        original_sleep = asyncio.sleep
        delays: list[float] = []

        async def fast_sleep(seconds: float) -> None:
            delays.append(seconds)
            client.last_cut_text = "pushed during wait"
            client.last_cut_text_time += 1.0
            await original_sleep(0)

        with unittest.mock.patch(
            "vnc_mcp.mcp.asyncio.sleep", side_effect=fast_sleep
        ):
            result = get_text(await call("get_clipboard", {"wait_seconds": 3.0}))
        assert result == "pushed during wait"
        assert delays  # the wait loop actually polled

    async def test_get_clipboard_rejects_out_of_range_wait(
        self, server_and_client: Any, call: Any
    ) -> None:
        _server, client = server_and_client
        with pytest.raises(ValueError, match="wait_seconds"):
            await call("get_clipboard", {"wait_seconds": 99.0})

    async def test_input_pending_advisory_appended_when_frame_predates_input(
        self, server_and_client: Any, call: Any
    ) -> None:
        _server, client = server_and_client
        client.last_input_time = 100.0
        client.last_update_time = 50.0  # frame parsed BEFORE the input
        result = get_text(await call("get_text_from_screen", {}))
        assert "[input-pending]" in result

    async def test_input_pending_advisory_absent_when_frame_is_newer(
        self, server_and_client: Any, call: Any
    ) -> None:
        _server, client = server_and_client
        client.last_input_time = 50.0
        client.last_update_time = 100.0
        result = get_text(await call("get_text_from_screen", {}))
        assert "[input-pending]" not in result
        assert "[stale]" not in result


class TestLazyConnection:
    async def test_connection_failure_is_clean_error(
        self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch, call: Any
    ) -> None:
        import vnc_mcp.mcp as module

        async def failing_connect(config: Any) -> None:
            raise ConnectionError("refused")

        monkeypatch.setattr(module.AsyncVNCClient, "connect", staticmethod(failing_connect))
        module._vnc_client = None
        try:
            with pytest.raises(ConnectionError, match="refused"):
                await module._get_client()
        finally:
            module._vnc_client = None

    async def test_lazy_client_reused(self, server_and_client: Any) -> None:
        import asyncio
        from types import SimpleNamespace

        import vnc_mcp.mcp as module

        marker = SimpleNamespace(
            rect=__import__("pyvnc").Rect(0, 0, 100, 100),
            desktop_name="fake",
            is_connected=True,
        )

        async def fake_connect(config: Any) -> Any:
            await asyncio.sleep(0)
            return marker

        module._vnc_client = None
        module._vnc_config = module.VNCConfig()
        original = module.AsyncVNCClient.connect
        module.AsyncVNCClient.connect = staticmethod(fake_connect)  # type: ignore[attr-defined]
        try:
            results = await asyncio.gather(module._get_client(), module._get_client())
            assert results[0] is marker
            assert results[0] is results[1]
            assert module._vnc_client is marker
        finally:
            module.AsyncVNCClient.connect = staticmethod(original)  # type: ignore[attr-defined]
            module._vnc_client = None
