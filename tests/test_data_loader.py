"""Databento loader tests against synthetic CSV files."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from backtest.data_loader import inspect_databento_file, load_databento_ohlcv, shape_of


@pytest.fixture
def databento_csv(tmp_path: Path):
    """Factory that writes a small CSV (optionally zst-compressed) and returns its path."""
    def _make(rows: list[dict], *, compressed: bool = False, name: str = "data") -> Path:
        df = pd.DataFrame(rows)
        if compressed:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            data = buf.getvalue().encode()
            cctx = zstd.ZstdCompressor()
            compressed_bytes = cctx.compress(data)
            path = tmp_path / f"{name}.csv.zst"
            path.write_bytes(compressed_bytes)
        else:
            path = tmp_path / f"{name}.csv"
            df.to_csv(path, index=False)
        return path
    return _make


def _row(ts: str, o: float, h: float, l: float, c: float, v: int, sym: str = "NQM6") -> dict:
    return {
        "ts_event": ts, "rtype": 33, "publisher_id": 1, "instrument_id": 42004058,
        "open": o, "high": h, "low": l, "close": c, "volume": v, "symbol": sym,
    }


def test_load_uncompressed_csv(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100.5, 128),
        _row("2026-05-01T00:01:00.000000000Z", 100.5, 102, 100, 101.5, 241),
        _row("2026-05-01T00:02:00.000000000Z", 101.5, 103, 101, 102.5, 274),
    ]
    path = databento_csv(rows)
    bars = load_databento_ohlcv(path)
    assert len(bars) == 3
    assert bars[0].open == 100
    assert bars[0].volume == 128
    assert bars[0].ts.tz is not None  # tz-aware


def test_load_compressed_csv(databento_csv):
    rows = [_row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100.5, 128)]
    path = databento_csv(rows, compressed=True)
    bars = load_databento_ohlcv(path)
    assert len(bars) == 1
    assert bars[0].open == 100


def test_missing_columns_raises(databento_csv):
    rows = [{"ts_event": "2026-05-01T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1}]
    path = databento_csv(rows)
    with pytest.raises(ValueError, match="missing columns"):
        load_databento_ohlcv(path)


def test_multiple_symbols_without_filter_raises(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100, 128, sym="NQM6"),
        _row("2026-05-01T00:01:00.000000000Z", 100, 101, 99, 100, 128, sym="NQU6"),
    ]
    path = databento_csv(rows)
    with pytest.raises(ValueError, match="Multiple symbols"):
        load_databento_ohlcv(path)


def test_symbol_filter_explicit(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100, 128, sym="NQM6"),
        _row("2026-05-01T00:01:00.000000000Z", 200, 201, 199, 200, 9999, sym="NQU6"),
    ]
    path = databento_csv(rows)
    bars = load_databento_ohlcv(path, symbol_filter="NQM6")
    assert len(bars) == 1
    assert bars[0].open == 100


def test_symbol_filter_auto_picks_highest_volume(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100, 128, sym="NQM6"),
        _row("2026-05-01T00:01:00.000000000Z", 200, 201, 199, 200, 9999, sym="NQU6"),
        _row("2026-05-01T00:02:00.000000000Z", 200, 201, 199, 200, 8888, sym="NQU6"),
    ]
    path = databento_csv(rows)
    bars = load_databento_ohlcv(path, symbol_filter="auto")
    assert len(bars) == 2
    assert all(b.open == 200 for b in bars)


def test_bars_sorted_by_ts(databento_csv):
    """Loader sorts by timestamp even if input is shuffled."""
    rows = [
        _row("2026-05-01T00:02:00.000000000Z", 102, 103, 101, 102.5, 274),
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100.5, 128),
        _row("2026-05-01T00:01:00.000000000Z", 100.5, 102, 100, 101.5, 241),
    ]
    path = databento_csv(rows)
    bars = load_databento_ohlcv(path)
    assert bars[0].open == 100
    assert bars[1].open == 100.5
    assert bars[2].open == 102


def test_shape_of_bars(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100, 100),
        _row("2026-05-01T00:01:00.000000000Z", 100, 101, 99, 100, 100),
        # Gap: skip 00:02
        _row("2026-05-01T00:05:00.000000000Z", 100, 101, 99, 100, 100),
    ]
    path = databento_csv(rows)
    bars = load_databento_ohlcv(path)
    s = shape_of(bars)
    assert s.n_bars == 3
    assert s.has_gaps is True


def test_inspect_file_summary(databento_csv):
    rows = [
        _row("2026-05-01T00:00:00.000000000Z", 100, 101, 99, 100, 100, sym="NQM6"),
        _row("2026-05-01T00:01:00.000000000Z", 100, 101, 99, 100, 100, sym="NQM6"),
        _row("2026-05-01T00:02:00.000000000Z", 100, 101, 99, 100, 100, sym="NQU6"),
    ]
    path = databento_csv(rows)
    s = inspect_databento_file(path)
    assert s.n_bars == 3
    assert s.symbols == {"NQM6": 2, "NQU6": 1}
    assert not s.has_gaps
