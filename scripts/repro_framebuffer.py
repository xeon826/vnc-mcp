"""Repro harness for the silent stale-framebuffer failure mode.

What this diagnoses
-------------------
The deployed vnc-mcp (pyvnc @ regulad/pyvnc 52703ba) has a silent failure mode:
after a while, ``capture()`` starts returning a stale framebuffer with NO error.
Two known culprits live inside pyvnc itself (this script changes none of it):

1. ``VNCClient.capture()`` waits up to 10s for a framebuffer-update event and
   then **swallows the timeout** (``except TimeoutError: pass``), returning the
   stale ``_pixels_rgba`` buffer instead of raising.
2. ``_framebuffer_listener`` can die via unknown-message / decode-error paths
   that only ``logger.*`` — invisible when the server never configures logging.

How to read the output
----------------------
- ``capture returned in ~10.0s`` -> you hit the swallowed-timeout path.
- Identical hashes across cycles -> stale buffer confirmed (nothing updated).
- Listener introspection at the end shows whether the listener task died
  (done/cancelled + its exception) — the root cause of (1) above.

Usage
-----
Reads connection settings from the same VNCMCP_* env convention as the CLI:

    export VNCMCP_HOST=192.168.1.7      # default: localhost
    export VNCMCP_PORT=5900             # default: 5900
    export VNCMCP_PASSWORD=...          # optional; never logged
    python scripts/repro_framebuffer.py [--cycles 3] [--delay 5] \
        [--type] [--rect X Y W H]

``--cycles``  capture cycles to run (default 3).
``--delay``   seconds to sleep between cycles (default 5).
``--type``    send a benign keystroke ("a" press/release) between cycles.
              Default OFF: the passive scenario reproduces the staleness
              without touching the remote session.
``--rect``    capture only a subrectangle instead of the full screen.

Requires DEBUG logging on stderr throughout (forced here, regardless of env).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from typing import Annotated

import numpy as np
import typer
from pyvnc import Rect
from pyvnc import VNCClient
from pyvnc import VNCConfig


app = typer.Typer()


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _describe_listener(client: VNCClient) -> None:
    """Introspect the (possibly dead) background listener task."""
    task = client._listener_task
    if task is None:
        print("listener task: None (never started)")
    else:
        state = "alive" if not task.done() else "done"
        if task.cancelled():
            state = "cancelled"
        print(f"listener task: {state} ({task.get_name()})")
        if task.done() and not task.cancelled():
            exception = task.exception()
            print(f"listener exception: {exception!r}")
    print(f"client._connected: {client._connected}")
    print(f"last_capture_stale: {getattr(client, 'last_capture_stale', 'n/a')}")
    print(f"_last_update_rect_count: {getattr(client, '_last_update_rect_count', 'n/a')}")
    pixels = client._pixels_rgba
    if pixels is None:
        print("framebuffer: None (no data ever received)")
    else:
        nonzero = float(np.count_nonzero(pixels)) / pixels.size
        print(f"framebuffer: shape={pixels.shape}, non-zero fraction={nonzero:.4f}")


@app.command()
def main(
    cycles: Annotated[int, typer.Option(help="Number of capture cycles.")] = 3,
    delay: Annotated[float, typer.Option(help="Seconds to sleep between cycles.")] = 5.0,
    type_key: Annotated[bool, typer.Option("--type", help="Send a benign keystroke between cycles.")] = False,
    rect: Annotated[str | None, typer.Option(help="Optional crop 'X Y W H'.")] = None,
) -> None:
    """Reproduce and diagnose the silent stale-framebuffer failure mode."""
    _run(cycles, delay, type_key, rect)


def _run(cycles: int, delay: float, type_key: bool, rect: str | None) -> None:
    """Blocking entry point; runs the async diagnosis loop."""

    async def _async_main() -> None:
        # Force DEBUG on stderr regardless of env: the pyvnc listener logs its
        # death via logging only, so without this it is invisible.
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        host = os.environ.get("VNCMCP_HOST", "localhost")
        port = int(os.environ.get("VNCMCP_PORT", "5900"))
        password = os.environ.get("VNCMCP_PASSWORD")  # never logged
        config = VNCConfig(host=host, port=port, password=password)

        crop: Rect | None = None
        if rect is not None:
            x, y, w, h = (int(part) for part in rect.split())
            crop = Rect(x, y, w, h)

        print(f"connecting to {host}:{port} (password {'set' if password else 'not set'})...")
        client = await VNCClient.connect(config)
        print(f"connected; framebuffer {client.rect.width}x{client.rect.height}, desktop {client.desktop_name!r}")

        hashes: list[str] = []
        try:
            for cycle in range(1, cycles + 1):
                if type_key and cycle > 1:
                    await client.write("a")
                start = time.perf_counter()
                array = await client.capture(crop)
                duration = time.perf_counter() - start
                digest = _hash_array(array)
                hashes.append(digest)
                flag = " <-- ~10s: swallowed capture timeout" if duration > 9.0 else ""
                stale = getattr(client, "last_capture_stale", None)
                print(
                    f"cycle {cycle}: capture returned in {duration:.2f}s, "
                    f"sha256={digest}, stale={stale}{flag}"
                )
                if cycle < cycles:
                    await asyncio.sleep(delay)
        finally:
            print("\n--- listener state introspection ---")
            _describe_listener(client)

        if len(set(hashes)) == 1 and len(hashes) > 1:
            print("\nSTALE BUFFER CONFIRMED: identical hashes across all cycles.")
        elif hashes:
            print("\nFramebuffers differ across cycles (no staleness observed).")

    asyncio.run(_async_main())


if __name__ == "__main__":
    app()
