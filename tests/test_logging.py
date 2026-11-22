"""Tests for logging configuration and the tool-call logging decorator."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest

import vnc_mcp.mcp as mcp_module
from vnc_mcp.mcp import LOG_LEVEL_ENV_VAR
from vnc_mcp.mcp import configure_logging
from vnc_mcp.mcp import parse_log_level


pytestmark = pytest.mark.anyio


def _server_stream_handlers(root: logging.Logger) -> list[logging.StreamHandler]:
    """StreamHandlers installed by vnc-mcp (pytest's LogCaptureHandler excluded)."""
    from _pytest.logging import LogCaptureHandler

    return [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, LogCaptureHandler)
    ]


class TestParseLogLevel:
    def test_none_defaults_to_info(self) -> None:
        level, warning = parse_log_level(None)
        assert level == logging.INFO
        assert warning is None

    def test_empty_string_defaults_to_info(self) -> None:
        level, warning = parse_log_level("   ")
        assert level == logging.INFO
        assert warning is None

    @pytest.mark.parametrize("name", ["debug", "DEBUG", "Info", "warning", "ERROR", "Critical"])
    def test_valid_names_case_insensitive(self, name: str) -> None:
        level, warning = parse_log_level(name)
        assert warning is None
        assert level == logging.getLevelName(name.upper())

    def test_invalid_falls_back_to_info_with_warning(self) -> None:
        level, warning = parse_log_level("not-a-level")
        assert level == logging.INFO
        assert warning is not None
        assert "not-a-level" in warning


class TestConfigureLogging:
    @pytest.fixture(autouse=True)
    def _reset_flag(self) -> None:
        """Isolate the module's configured-once flag for every test."""
        original_flag = mcp_module._logging_configured
        original_handlers = logging.getLogger().handlers[:]
        original_level = logging.getLogger().level
        mcp_module._logging_configured = False
        logging.getLogger().handlers = []
        yield
        mcp_module._logging_configured = original_flag
        logging.getLogger().handlers = original_handlers
        logging.getLogger().setLevel(original_level)

    def test_uses_stderr_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
        configure_logging(None)
        root = logging.getLogger()
        stream_handlers = _server_stream_handlers(root)
        assert len(stream_handlers) == 1
        assert stream_handlers[0].stream is sys.stderr
        # stdout is the MCP stdio channel: nothing may write there.
        assert all(handler.stream is not sys.stdout for handler in root.handlers)

    def test_env_var_sets_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "debug")
        configure_logging(None)
        assert logging.getLogger().level == logging.DEBUG

    def test_explicit_level_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")
        applied = configure_logging("ERROR")
        assert applied == logging.ERROR
        assert logging.getLogger().level == logging.ERROR

    def test_invalid_env_var_falls_back_to_info(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "bogus")
        applied = configure_logging(None)
        assert applied == logging.INFO
        assert any("bogus" in record.message for record in caplog.records)

    def test_configures_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
        configure_logging("INFO")
        configure_logging("ERROR")
        # The handler is installed exactly once; later calls only adjust levels.
        assert len(_server_stream_handlers(logging.getLogger())) == 1
        assert mcp_module._logging_configured is True

    def test_mcp_and_pyvnc_loggers_not_silenced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
        configure_logging("DEBUG")
        for name in ("mcp", "pyvnc"):
            module_logger = logging.getLogger(name)
            assert module_logger.disabled is False
            assert module_logger.level == logging.NOTSET  # inherits root


class TestLoggedTool:
    async def test_entry_exit_logged_with_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        @mcp_module._logged_tool
        async def sample_tool(x: int) -> int:  # pragma: no cover - trivial
            """Trivial tool."""
            return x + 1

        with caplog.at_level(logging.DEBUG, logger=mcp_module.logger.name):
            result = await sample_tool(x=1)

        assert result == 2
        messages = [record.message for record in caplog.records]
        assert any("sample_tool: entry" in message for message in messages)
        assert any("sample_tool: exit" in message and "ms" in message for message in messages)

    async def test_args_are_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        @mcp_module._logged_tool
        async def sample_tool(payload: str) -> str:  # pragma: no cover - trivial
            """Trivial tool."""
            return payload

        with caplog.at_level(logging.DEBUG, logger=mcp_module.logger.name):
            await sample_tool(payload="x" * 500)

        entry = next(record.message for record in caplog.records if "entry" in record.message)
        assert len(entry) < 300
        assert "…" in entry

    async def test_exception_logged_and_reraised(self, caplog: pytest.LogCaptureFixture) -> None:
        @mcp_module._logged_tool
        async def failing_tool() -> None:  # pragma: no cover - trivial
            """Trivial tool."""
            raise ValueError("boom")

        with caplog.at_level(logging.DEBUG, logger=mcp_module.logger.name):
            with pytest.raises(ValueError, match="boom"):
                await failing_tool()

        error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(error_records) == 1
        record = error_records[0]
        assert "failing_tool" in record.message
        assert record.exc_info is not None  # full traceback attached
        assert "failed after" in record.message

    def test_wraps_preserves_name(self) -> None:
        @mcp_module._logged_tool
        async def named_tool() -> None:  # pragma: no cover - trivial
            """Trivial tool."""
            return None

        assert named_tool.__name__ == "named_tool"


class TestCreateServerPlumbsLogLevel:
    async def test_create_mcp_server_applies_cli_level(self, server_and_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _client = server_and_client
        mcp_module._logging_configured = False
        logging.getLogger().handlers = []
        monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")
        from pyvnc import VNCConfig

        from vnc_mcp.mcp import create_mcp_server

        await create_mcp_server(VNCConfig(host="invalid.invalid", port=1), log_level="ERROR")
        assert logging.getLogger().level == logging.ERROR
        mcp_module._logging_configured = True  # keep suite state sane
