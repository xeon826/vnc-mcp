"""Raw RFB wire probe: identify non-standard server->client messages.

Diagnoses the stale-framebuffer failure on servers that push vendor-specific
message types (observed: 24/26/27/61/63/255) that the pyvnc listener cannot
frame, desyncing the stream so framebuffer updates stop parsing.

Phases:
  1. Full RFB 3.8 handshake (mirrors pyvnc's _perform_handshake exactly).
  2. One non-incremental full-frame update request; structured parse of the
     update (rect coordinates + encodings actually sent).
  3. Quiet phase: NO further requests; record every unsolicited message the
     server pushes (type byte, size, hexdump, inter-arrival gaps) for --quiet
     seconds. Unsolicited messages here are what kills the listener.

Read-only: sends no pointer/keyboard events; shared flag = 1. Password is read
from VNCMCP_PASSWORD and never logged.

Usage:
    export VNCMCP_HOST=... VNCMCP_PORT=5900 VNCMCP_PASSWORD=...
    python scripts/probe_rfb.py [--quiet 60]
"""

import asyncio
import os
import sys
import time

from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms as _algs
from cryptography.hazmat.primitives.ciphers import modes as _modes
from pyvnc.pyvnc_async import AUTH_STATE_PERMITTED  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import AUTH_TYPE_NONE  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import AUTH_TYPE_VNC  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import DEFAULT_PIXEL_FORMAT_NAME  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import ENCODING_RAW  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import ENCODING_ZLIB  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import RFC_6143_CANON_STRING_ENCODING  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import SUPPORTED_ENCODINGS  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import VNC_PROTOCOL_HEADER  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import VNC_PROTOCOL_HEADER_SIZE  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import _parse_protocol_version  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import _read_bytes  # noqa: I001 - probe-only imports
from pyvnc.pyvnc_async import _read_int  # noqa: I001 - probe-only imports


def _hexdump(data: bytes, limit: int = 96) -> str:
    return data[:limit].hex(" ") + (f" …(+{len(data) - limit}B)" if len(data) > limit else "")


async def _handshake(host: str, port: str, password: str | None):
    """Mirror pyvnc._perform_handshake and return (reader, writer, dims, name)."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 10)

    intro = await _read_bytes(reader, VNC_PROTOCOL_HEADER_SIZE)
    version = _parse_protocol_version(intro)
    print(f"[handshake] server protocol: {intro!r} -> {version}")
    writer.write(VNC_PROTOCOL_HEADER)
    await writer.drain()

    auth_types = set(await _read_bytes(reader, await _read_int(reader, 1)))
    print(f"[handshake] server auth types: {sorted(auth_types)}")
    chosen: int | None = None
    if password is not None and AUTH_TYPE_VNC in auth_types:
        chosen = AUTH_TYPE_VNC
    elif AUTH_TYPE_NONE in auth_types:
        chosen = AUTH_TYPE_NONE
    if chosen is None:
        raise SystemExit(f"no usable auth type in {sorted(auth_types)}")

    writer.write(bytes([chosen]))
    await writer.drain()
    if chosen == AUTH_TYPE_VNC:
        des_key = password.encode(RFC_6143_CANON_STRING_ENCODING)[:8].ljust(8, b"\x00")
        des_key = bytes(int(bin(n)[:1:-1].ljust(8, "0"), 2) for n in des_key)
        encryptor = Cipher(_algs.TripleDES(des_key), _modes.ECB()).encryptor()
        challenge = await _read_bytes(reader, 16)
        writer.write(encryptor.update(challenge) + encryptor.finalize())
        await writer.drain()
    result = await _read_int(reader, 4)
    if result != AUTH_STATE_PERMITTED:
        raise SystemExit(f"auth failed (result={result})")
    print("[handshake] auth ok")

    writer.write(b"\x01")  # shared
    await writer.drain()
    width = await _read_int(reader, 2)
    height = await _read_int(reader, 2)
    pixfmt = await _read_bytes(reader, 16)
    name_len = await _read_int(reader, 4)
    name = (await _read_bytes(reader, name_len)).decode("latin-1")
    print(f"[handshake] framebuffer {width}x{height} name={name!r} pixfmt={pixfmt.hex()}")
    return reader, writer, (width, height), pixfmt, name


async def main() -> None:
    """Run the probe: handshake, one parsed full-frame update, then quiet listening."""
    host = os.environ.get("VNCMCP_HOST", "localhost")
    port = os.environ.get("VNCMCP_PORT", "5900")
    password = os.environ.get("VNCMCP_PASSWORD")  # never logged
    quiet_secs = float(sys.argv[sys.argv.index("--quiet") + 1]) if "--quiet" in sys.argv else 60.0

    reader, writer, (width, height), pixfmt, _name = await _handshake(host, int(port), password)

    # Match pyvnc's behavior: only force SetPixelFormat when the server's own
    # format isn't a 32bpp true-color format pyvnc can reorder (shifts 0/8/16).
    bpp, depth, _be, true_color = pixfmt[0], pixfmt[1], pixfmt[2], pixfmt[3]
    if (bpp, depth, true_color) != (32, 24, 1):
        from pyvnc.pyvnc_common import PIXEL_FORMATS as _PFS

        pf = _PFS[DEFAULT_PIXEL_FORMAT_NAME]
        print(f"[probe] pixfmt bpp={bpp} depth={depth} tc={true_color} unusable; sending SetPixelFormat {DEFAULT_PIXEL_FORMAT_NAME}")
        writer.write(b"\x00\x00\x00\x00" + pf.serialize())
        await writer.drain()
    else:
        print(f"[probe] pixfmt bpp={bpp} depth={depth} tc={true_color} usable as-is")
    writer.write(b"\x02\x00" + len(SUPPORTED_ENCODINGS).to_bytes(2, "big") + b"".join(
        e.to_bytes(4, "big", signed=True) for e in SUPPORTED_ENCODINGS
    ))
    await writer.drain()
    print(f"[probe] SetEncodings {SUPPORTED_ENCODINGS}")

    # Phase 2: one full-frame update request, parsed strictly.
    print("[probe] requesting full-frame update (non-incremental)…")
    writer.write(b"\x03\x00" + (0).to_bytes(2, "big") + (0).to_bytes(2, "big")
                 + width.to_bytes(2, "big") + height.to_bytes(2, "big"))
    await writer.drain()

    msg_type = await _read_int(reader, 1)
    if msg_type != 0:
        body = await _read_bytes(reader, 64)
        print(f"[probe] !! first message after request was type={msg_type}: {_hexdump(body)}")
    else:
        await _read_bytes(reader, 1)  # padding
        num_rects = await _read_int(reader, 2)
        print(f"[probe] update: {num_rects} rect(s)")
        total = 0
        for i in range(num_rects):
            x = await _read_int(reader, 2)
            y = await _read_int(reader, 2)
            w = await _read_int(reader, 2)
            h = await _read_int(reader, 2)
            enc = int.from_bytes(await _read_bytes(reader, 4), "big", signed=True)
            size = ""
            if enc == ENCODING_RAW:
                payload = await _read_bytes(reader, w * h * 4)
                total += len(payload)
                size = f"raw {len(payload)}B"
            elif enc == ENCODING_ZLIB:
                n = await _read_int(reader, 4)
                payload = await _read_bytes(reader, n)
                total += n
                size = f"zlib {n}B"
            else:
                print(f"[probe] rect {i}: {w}x{h}+{x}+{y} UNADVERTISED encoding {enc} — desync risk; stopping parse")
                break
            if i < 6 or i == num_rects - 1:
                print(f"[probe] rect {i}: {w}x{h}+{x}+{y} enc={enc} ({size})")
        print(f"[probe] update parsed cleanly; {total} payload bytes total")

    # Phase 3: quiet listening — what does the server push unsolicited?
    print(f"[probe] quiet phase: listening {quiet_secs:.0f}s with NO update requests…")
    start = time.monotonic()
    last = start
    counts: dict[int, int] = {}
    try:
        while time.monotonic() - start < quiet_secs:
            try:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=quiet_secs - (time.monotonic() - start))
            except TimeoutError:
                break
            if not chunk:
                print("[probe] server closed the connection")
                break
            now = time.monotonic()
            t = chunk[0]
            counts[t] = counts.get(t, 0) + 1
            print(f"[+{now - start:6.1f}s, +{now - last:5.2f}s] msg type={t:3d} len={len(chunk):4d}  {_hexdump(chunk)}")
            last = now
    finally:
        print(f"[probe] unsolicited message types seen: {counts or 'NONE (server pushes nothing)'}")
        writer.close()


def _mk_pixelformat(raw: bytes):  # retained for interactive debugging
    from pyvnc.pyvnc_common import PixelFormat

    return PixelFormat.deserialize(raw)


if __name__ == "__main__":
    asyncio.run(main())
