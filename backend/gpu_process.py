"""プロセスごとの VRAM 使用量（Windows のパフォーマンスカウンター）

nvidia-smi は Windows（WDDM）ではプロセスごとの使用量を返さないが、
Windows の「GPU Process Memory\\Dedicated Usage」カウンターなら取れる。
PowerShell の Get-Counter は起動に1秒以上かかるので、pdh.dll を直接呼ぶ。
"""
from __future__ import annotations

import ctypes
import re
import sys
from ctypes import wintypes

_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_LARGE = 0x00000400
_COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"


class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _FmtItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FmtValue)]


class GpuProcessMemory:
    """pid → 専用VRAM使用量(bytes)。Windows 以外・取得失敗時は空の dict を返す"""

    def __init__(self):
        self._query = None
        self._counter = None
        self.available = False
        if sys.platform != "win32":
            return
        try:
            self._pdh = ctypes.WinDLL("pdh")
            q, c = wintypes.HANDLE(), wintypes.HANDLE()
            if self._pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                return
            if self._pdh.PdhAddEnglishCounterW(q, _COUNTER, None, ctypes.byref(c)) != 0:
                self._pdh.PdhCloseQuery(q)
                return
            self._query, self._counter = q, c
            self.available = True
        except Exception:
            self.available = False

    def read(self) -> dict[int, int]:
        if not self.available:
            return {}
        pdh = self._pdh
        if pdh.PdhCollectQueryData(self._query) != 0:
            return {}
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(self._counter, _PDH_FMT_LARGE,
                                                  ctypes.byref(size), ctypes.byref(count), None)
        if (status & 0xFFFFFFFF) != _PDH_MORE_DATA or size.value == 0:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        status = pdh.PdhGetFormattedCounterArrayW(self._counter, _PDH_FMT_LARGE,
                                                  ctypes.byref(size), ctypes.byref(count), buf)
        if status != 0:
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(_FmtItem))
        out: dict[int, int] = {}
        for i in range(count.value):
            m = re.match(r"pid_(\d+)_", items[i].szName or "")
            if m:
                pid = int(m.group(1))
                out[pid] = out.get(pid, 0) + max(0, items[i].FmtValue.largeValue)
        return out

    def close(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = None
            self.available = False