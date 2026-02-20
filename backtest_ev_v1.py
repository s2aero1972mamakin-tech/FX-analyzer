# backtest_ev_v1_fixed.py
# Ver1: EVゲート簡易ウォークフォワード検証
# - Streamlit Cloud / yfinance レート制限を避けるため、基本は Stooq (daily) を使用
# - 依存: pandas（streamlit環境なら通常入っています）
# 使い方: main.py から
#   import backtest_ev_v1
#   wf_df, summ = backtest_ev_v1.run_backtest(...)
# を呼べるように、"run_backtest" を提供します。

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd


# -----------------------------
# Data fetch (Stooq daily)
# -----------------------------
_STOOQ_MAP = {
    # yfinance FX symbols -> stooq symbols
    "JPY=X": "usdjpy",
    "EURUSD=X": "eurusd",
    "GBPUSD=X": "gbpusd",
    "AUDUSD=X": "audusd",
    "EURJPY=X": "eurjpy",
    "GBPJPY=X": "gbpjpy",
    "AUDJPY=X": "audjpy",
    # sometimes users pass these:
    "USDJPY=X": "usdjpy",
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "AUDUSD": "audusd",
    "EURJPY": "eurjpy",
    "GBPJPY": "gbpjpy",
    "AUDJPY": "audjpy",
}


def _stooq_symbol(pair_symbol: str) -> Optional[str]:
    s = (pair_symbol or "").strip().upper()
    return _STOOQ_MAP.get(s)


def _parse_period_to_days(period: str) -> int:
    p = (period or "").strip().lower()
    if p.endswith("y"):
        try:
            y = int(p[:-1])
            return y * 365 + 30
        except Exception:
            return 3650
    if p.endswith("mo"):
        try:
            m = int(p[:-2])
            return m * 31 + 10
        except Exception:
            return 365
    if p.endswith("d"):
        try:
            d = int(p[:-1])
            return d
        except Exception:
            return 365
    if p == "max":
        return 365 * 30
    return 3650


def _fetch_ohlc_stooq(pair_symbol: str, period: str = "10y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"ok": False, "source": "stooq", "error": None}
    stq = _stooq_symbol(pair_symbol)
    if not stq:
        meta["error"] = "stooq_symbol_not_found"
        return pd.DataFrame(), meta

    url = f"https://stooq.com/q/d/l/?s={stq}&i=d"
    try:
        df = pd.read_csv(url)
        if df is None or df.empty or "Date" not in df.columns:
            meta["error"] = "stooq_bad_csv"
            return pd.DataFrame(), meta

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        need = ["Open", "High", "Low", "Close"]
        for c in need:
            if c not in df.columns:
                meta["error"] = f"missing_col:{c}"
                return pd.DataFrame(), meta

        df = df[need].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            meta["error"] = "stooq_empty_after_clean"
            return pd.DataFrame(), meta

        days = _parse_period_to_days(period)
        end = pd.Timestamp(datetime.utcnow().date())
        start = end - pd.Timedelta(days=days)
        df = df.loc[df.index >= start]

        if df.empty:
            meta["error"] = "stooq_empty_after_slice"
            return pd.DataFrame(), meta

        meta["ok"] = True
        meta["rows"] = int(len(df))
        meta["from"] = str(df.index.min().date())
        meta["to"] = str(df.index.max().date())
        return df, meta

    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta


def _fetch_ohlc_yfinance(pair_symbol: str, period: str = "10y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fallback. yfinanceはStreamlit Cloudでrate-limitしやすいので完全ガード。"""
    meta: Dict[str, Any] = {"ok": False, "source": "yfinance", "error": None}
    try:
        import yfinance as yf
        df = yf.Ticker(pair_symbol).history(period=period, interval="1d")
        if df is None or df.empty:
            meta["error"] = "yfinance_empty"
            return pd.DataFrame(), meta

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        need = ["Open", "High", "Low", "Close"]
        for c in need:
            if c not in df.columns:
                meta["error"] = f"missing_col:{c}"
                return pd.DataFrame(), meta

        df = df[need].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            meta["error"] = "yfinance_empty_after_clean"
            return pd.DataFrame(), meta

        meta["ok"] = True
        meta["rows"] = int(len(df))
        meta["from"] = str(pd.to_datetime(df.index.min()).date())
        meta["to"] = str(pd.to_datetime(df.index.max()).date())
        return df, meta

    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta


def fetch_ohlc(pair_symbol: str, period: str = "10y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """優先: Stooq。ダメなら yfinance。"""
    df, meta = _fetch_ohlc_stooq(pair_symbol, period=period)
    if meta.get("ok") and not df.empty:
        return df, meta

    df2, meta2 = _fetch_ohlc_yfinance(pair_symbol, period=period)
    if meta2.get("ok") and not df2.empty:
        meta2["fallback_from"] = meta
        return df2, meta2

    return pd.DataFrame(), {"ok": False, "source": "none", "error": {"stooq": meta, "yfinance": meta2}}


# -----------------------------
# Indicators
# -----------------------------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
    rs = gain / loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["SMA25"] = d["Close"].rolling(25).mean()
    d["SMA75"] = d["Close"].rolling(75).mean()
    d["RSI"] = _rsi(d["Close"], 14)
    d["ATR"] = _atr(d, 14)
    d["DEV_SMA25_ATR"] = (d["Close"] - d["SMA25"]) / d["ATR"].replace(0, pd.NA)
    d["TREND_ATR"] = (d["SMA25"] - d["SMA75"]).abs() / d["ATR"].replace(0, pd.NA)
    d["ATR_RATIO"] = d["ATR"] / d["Close"].replace(0, pd.NA)
    return d


# -----------------------------
# State probabilities (simple softmax)
# -----------------------------

_STATES = ["trend_up", "trend_down", "range", "risk_off"]


def _softmax(logits: List[float]) -> List[float]:
    import math

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [x / s for x in exps]


def state_probs_row(r: pd.Series) -> Dict[str, float]:
    """数値のみ（外部特徴量無し）の簡易版 Softmax。"""
    try:
        trend = float(r.get("TREND_ATR", 0.0) or 0.0)
        dev = float(r.get("DEV_SMA25_ATR", 0.0) or 0.0)
        rsi = float(r.get("RSI", 50.0) or 50.0)
        atr_ratio = float(r.get("ATR_RATIO", 0.0) or 0.0)
    except Exception:
        trend, dev, rsi, atr_ratio = 0.0, 0.0, 50.0, 0.0

    dir_term = (rsi - 50.0) / 10.0 + (dev * 0.6)

    up = +1.2 * trend + 0.7 * dir_term
    down = +1.2 * trend - 0.7 * dir_term
    rng = -0.9 * trend - 0.8 * abs(dir_term) + 0.6
    risk = +12.0 * atr_ratio + 0.2 * abs(dev)

    logits = [max(-8.0, min(8.0, x)) for x in [up, down, rng, risk]]
    ps = _softmax(logits)
    return {k: float(v) for k, v in zip(_STATES, ps)}


def dominant_state(ps: Dict[str, float]) -> str:
    if not ps:
        return "range"
    return max(ps.items(), key=lambda kv: kv[1])[0]


# -----------------------------
# Mean-R estimation (train)
# -----------------------------

def _forward_return(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0


def _estimate_mean_R_by_state(train: pd.DataFrame, horizon_days: int) -> Dict[str, float]:
    """trainで状態別のmean_R（ATRをリスク単位としたR倍率）を推定。"""
    eps = 1e-9
    fr = _forward_return(train["Close"], horizon_days)
    atr = train["ATR"].replace(0, pd.NA)

    out: Dict[str, List[float]] = {s: [] for s in _STATES}

    for i in range(len(train) - horizon_days):
        r = train.iloc[i]
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
        price = float(r.get("Close") or 0.0)
        if atr_i <= 0 or price <= 0:
            continue

        ps = state_probs_row(r)
        st = dominant_state(ps)

        ret = float(fr.iloc[i]) if pd.notna(fr.iloc[i]) else 0.0
        base_R = (ret * price) / max(atr_i, eps)
        dev = float(r.get("DEV_SMA25_ATR") or 0.0)

        if st == "trend_up":
            R = base_R
        elif st == "trend_down":
            R = -base_R
        elif st == "range":
            R = (-1.0 if dev > 0 else 1.0) * base_R
            R *= 0.75
        else:
            R = -abs(base_R) * 0.25

        out[st].append(float(R))

    meanR: Dict[str, float] = {}
    for s in _STATES:
        xs = out[s]
        meanR[s] = float(pd.Series(xs).mean()) if xs else 0.0
    return meanR


# -----------------------------
# Trading simulation (test)
# -----------------------------

@dataclass
class DayResult:
    date: str
    decision: str
    dominant_state: str
    expected_R_ev: float
    realized_R: float


def _simulate_test_period(test: pd.DataFrame, meanR: Dict[str, float], horizon_days: int, min_expected_R: float) -> List[DayResult]:
    eps = 1e-9
    fr = _forward_return(test["Close"], horizon_days)
    out: List[DayResult] = []

    for i in range(len(test) - horizon_days):
        r = test.iloc[i]
        dt = str(pd.to_datetime(test.index[i]).date())

        atr_i = float(r.get("ATR") or 0.0)
        price = float(r.get("Close") or 0.0)
        if atr_i <= 0 or price <= 0:
            out.append(DayResult(dt, "NO_TRADE", "na", 0.0, 0.0))
            continue

        ps = state_probs_row(r)
        ev = 0.0
        for s in _STATES:
            ev += float(ps.get(s, 0.0)) * float(meanR.get(s, 0.0))

        dom = dominant_state(ps)

        if ev < float(min_expected_R):
            out.append(DayResult(dt, "NO_TRADE", dom, float(ev), 0.0))
            continue

        ret = float(fr.iloc[i]) if pd.notna(fr.iloc[i]) else 0.0
        base_R = (ret * price) / max(atr_i, eps)
        dev = float(r.get("DEV_SMA25_ATR") or 0.0)

        if dom == "trend_up":
            realized = base_R
        elif dom == "trend_down":
            realized = -base_R
        elif dom == "range":
            realized = (-1.0 if dev > 0 else 1.0) * base_R
            realized *= 0.75
        else:
            out.append(DayResult(dt, "NO_TRADE", dom, float(ev), 0.0))
            continue

        out.append(DayResult(dt, "TRADE", dom, float(ev), float(realized)))

    return out


# -----------------------------
# Walk-forward orchestration
# -----------------------------

def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # clamp day
    dim = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    day = min(d.day, dim)
    return date(y, m, day)


def _max_drawdown(equity: List[float]) -> float:
    peak = -1e18
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = peak - x
        max_dd = max(max_dd, dd)
    return float(max_dd)


def run_backtest(
    pair_symbol: str,
    period: str = "10y",
    horizon_days: int = 5,
    train_years: int = 3,
    test_months: int = 6,
    min_expected_R: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    main.py 側の呼び出しに合わせたシグネチャ。

    Returns:
      - wf_df: per-window metrics
      - summary: overall metrics
    """
    df, meta = fetch_ohlc(pair_symbol, period=period)
    if df is None or df.empty:
        raise RuntimeError(f"Price fetch failed: {meta}")

    df = compute_indicators(df).dropna()
    if df.empty or len(df) < 260:
        raise RuntimeError("Not enough data after indicator calc")

    start_date = pd.to_datetime(df.index.min()).date()
    end_date = pd.to_datetime(df.index.max()).date()

    windows = []
    cur = start_date
    while True:
        train_start = cur
        # train_years後の同月日（存在しない日は28日に寄せる）
        train_end = date(train_start.year + int(train_years), train_start.month, min(train_start.day, 28))
        test_end = _add_months(train_end, int(test_months))
        if test_end > end_date:
            break
        windows.append((train_start, train_end, test_end))
        cur = _add_months(cur, int(test_months))

    if not windows:
        raise RuntimeError("No walk-forward windows possible with given params")

    rows = []
    all_results: List[DayResult] = []

    idx_dates = pd.to_datetime(df.index).date

    for (tr0, tr1, te1) in windows:
        tr_mask = (idx_dates >= tr0) & (idx_dates < tr1)
        te_mask = (idx_dates >= tr1) & (idx_dates < te1)

        train = df.loc[tr_mask]
        test = df.loc[te_mask]

        if len(train) < 200 or len(test) < (int(horizon_days) + 20):
            continue

        meanR = _estimate_mean_R_by_state(train, int(horizon_days))
        results = _simulate_test_period(test, meanR, int(horizon_days), float(min_expected_R))
        all_results.extend(results)

        trades = [r for r in results if r.decision == "TRADE"]
        rs = [r.realized_R for r in trades]
        n = len(rs)

        avgR = float(pd.Series(rs).mean()) if rs else 0.0
        sumR = float(pd.Series(rs).sum()) if rs else 0.0
        win = sum(1 for x in rs if x > 0)
        winrate = (win / n) if n > 0 else 0.0

        eq = []
        c = 0.0
        for x in rs:
            c += float(x)
            eq.append(c)
        mdd = _max_drawdown(eq) if eq else 0.0

        rows.append({
            "train_start": str(tr0),
            "train_end": str(tr1),
            "test_end": str(te1),
            "n_trades": int(n),
            "avg_R": float(avgR),
            "sum_R": float(sumR),
            "win_rate": float(winrate),
            "max_dd_R": float(mdd),
            "meanR_trend_up": float(meanR.get("trend_up", 0.0)),
            "meanR_trend_down": float(meanR.get("trend_down", 0.0)),
            "meanR_range": float(meanR.get("range", 0.0)),
            "meanR_risk_off": float(meanR.get("risk_off", 0.0)),
        })

    wf_df = pd.DataFrame(rows)
    if wf_df.empty:
        raise RuntimeError("All windows skipped. Try smaller train_years/test_months or larger period.")

    overall_trades = [r for r in all_results if r.decision == "TRADE"]
    rs_all = [r.realized_R for r in overall_trades]
    n_all = len(rs_all)

    avg_all = float(pd.Series(rs_all).mean()) if rs_all else 0.0
    sum_all = float(pd.Series(rs_all).sum()) if rs_all else 0.0
    win_all = sum(1 for x in rs_all if x > 0)
    winrate_all = (win_all / n_all) if n_all > 0 else 0.0

    eq2 = []
    c2 = 0.0
    for x in rs_all:
        c2 += float(x)
        eq2.append(c2)
    mdd_all = _max_drawdown(eq2) if eq2 else 0.0

    summary = {
        "pair_symbol": pair_symbol,
        "period": period,
        "horizon_days": int(horizon_days),
        "train_years": int(train_years),
        "test_months": int(test_months),
        "min_expected_R": float(min_expected_R),
        "data_meta": meta,
        "windows": int(len(wf_df)),
        "n_trades": int(n_all),
        "avg_R": float(avg_all),
        "sum_R": float(sum_all),
        "win_rate": float(winrate_all),
        "max_dd_R": float(mdd_all),
    }
    return wf_df, summary
