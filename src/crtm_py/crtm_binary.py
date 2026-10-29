"""Fortran sequential-unformatted reader for CRTM coefficient files.

CRTM ``Open_Binary_File`` writes a 4-byte file id ``123456789`` and then
ordinary unformatted records. ``Little_Endian`` and ``Big_Endian`` copies
differ only in byte order, including the record markers. ``REAL(fp)`` in
the v2.4 files used here is IEEE-754 binary64. The ODPS regression vector
``C`` is ``REAL(Single)``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

FILE_ID = 123456789


class FortranFile:
    """Sequential unformatted file, little- or big-endian."""

    def __init__(self, path, endian: str | None = None):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if endian is None:
            endian = _detect_endian(raw)
        self.endian = endian
        self._records = _split_records(raw, endian)
        self._i = 0
        magic = self.read_i32()
        if int(magic) != FILE_ID:
            raise ValueError(f"{self.path} file id {magic} is not {FILE_ID}")

    def read_record(self) -> bytes:
        if self._i >= len(self._records):
            raise EOFError(self.path)
        rec = self._records[self._i]
        self._i += 1
        return rec

    def read_i32(self, n: int | None = None):
        rec = self.read_record()
        if n is None:
            if len(rec) != 4:
                raise ValueError(f"expected 4-byte integer, got {len(rec)} in {self.path}")
            return struct.unpack(self.endian + "i", rec)[0]
        count = len(rec) // 4
        if n != count:
            raise ValueError(f"expected {n} int32, got {count} in {self.path}")
        return np.frombuffer(rec, dtype=np.dtype(self.endian + "i4")).astype(np.int64, copy=False)

    def read_i32s(self, shape, order="F"):
        rec = self.read_record()
        flat = np.frombuffer(rec, dtype=np.dtype(self.endian + "i4")).astype(np.int64, copy=True)
        return flat.reshape(shape, order=order)

    def read_f64(self, shape=None, order="F"):
        rec = self.read_record()
        flat = np.frombuffer(rec, dtype=np.dtype(self.endian + "f8")).astype(np.float64, copy=True)
        if shape is None:
            return flat
        return flat.reshape(shape, order=order)

    def read_f32(self, shape=None, order="F"):
        rec = self.read_record()
        flat = np.frombuffer(rec, dtype=np.dtype(self.endian + "f4")).astype(np.float64, copy=True)
        if shape is None:
            return flat
        return flat.reshape(shape, order=order)

    def read_string(self, n: int | None = None) -> str:
        rec = self.read_record()
        if n is not None and len(rec) != n:
            raise ValueError(f"expected {n}-byte string, got {len(rec)} in {self.path}")
        return rec.decode("ascii", errors="replace").rstrip()

    def read_chars(self, n_items: int, width: int) -> list[str]:
        rec = self.read_record()
        if len(rec) != n_items * width:
            raise ValueError(
                f"expected {n_items}x{width} characters, got {len(rec)} bytes in {self.path}"
            )
        return [rec[i * width : (i + 1) * width].decode("ascii", errors="replace").rstrip() for i in range(n_items)]


def _detect_endian(raw: bytes) -> str:
    if len(raw) < 12:
        raise ValueError("coefficient file is too short")
    little = struct.unpack_from("<i", raw, 0)[0]
    if little == 4 and struct.unpack_from("<i", raw, 4)[0] == FILE_ID:
        return "<"
    big = struct.unpack_from(">i", raw, 0)[0]
    if big == 4 and struct.unpack_from(">i", raw, 4)[0] == FILE_ID:
        return ">"
    raise ValueError("not a CRTM unformatted coefficient file")


def _split_records(raw: bytes, endian: str) -> list[bytes]:
    out = []
    i = 0
    n = len(raw)
    while i < n:
        if i + 4 > n:
            raise ValueError("truncated record marker")
        mark = struct.unpack_from(endian + "i", raw, i)[0]
        if mark < 0 or i + 8 + mark > n:
            raise ValueError(f"bad record length {mark} at byte {i}")
        end = struct.unpack_from(endian + "i", raw, i + 4 + mark)[0]
        if end != mark:
            raise ValueError(f"record marker mismatch at byte {i}")
        out.append(raw[i + 4 : i + 4 + mark])
        i += 8 + mark
    return out
