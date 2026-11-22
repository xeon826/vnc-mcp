"""Tests for the CLI: argument parsing only (no VNC connection is made)."""

from __future__ import annotations

from typer.testing import CliRunner

from vnc_mcp.__main__ import cli


runner = CliRunner()


class TestCLI:
    def test_help_succeeds(self) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_positional_args_documented(self) -> None:
        result = runner.invoke(cli, ["--help"])
        for arg in ("HOST", "PORT", "USERNAME", "PASSWORD", "TIMEOUT"):
            assert arg in result.output

    def test_env_vars_documented(self) -> None:
        result = runner.invoke(cli, ["--help"])
        for envvar in ("VNCMCP_HOST", "VNCMCP_PORT", "VNCMCP_TIMEOUT", "VNCMCP_USERNAME", "VNCMCP_PASSWORD"):
            assert envvar in result.output

    def test_invalid_port_rejected(self) -> None:
        result = runner.invoke(cli, ["localhost", "not-a-port"])
        assert result.exit_code != 0
