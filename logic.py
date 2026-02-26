

# logic_fixed28_trend_entry_engine_compat.py
# Drop-in replacement for logic.py
# - Adds self-contained: HH/HL detection, breakout strength, continuation probability, phase-aware EV thresholding,
#   momentum bonus (capped), and breakout gate.
# - Designed to be backward-compatible with existing main.py callers.
#
# NOTE: This module does not depend on ctx keys being passed from main.py; it computes needed signals from price history.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Economic calendar / event risk (optional, free feed)
# ---------------------------------------------------------------------
# This tool is swing-oriented but macro events can cause gaps/slippage.
# We therefore compute an "event_risk_score" from upcoming releases and
# optionally block trades within a high-impact window.
#
# Default feed: Forex Factory weekly export (JSON)
# - https://nfs.faireconomy.media/ff_calendar_thisweek.json
#
# Notes:
# - If the feed is unavailable, we fail safe (event_risk_score=0) and
#   expose status in ctx_out so the UI can warn the operator.
#
import json
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

_FF_CAL_URL_DEFAULT = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_EVENT_CACHE = {
    "ts": 0.0,     # epoch seconds
    "url": None,
    "data": None,  # list
    "err": None,   # str
}

def _pair_to_ccys(pair: str) -> Tuple[str, str]:
    s = (pair or "").upper().replace(" ", "")
    s = s.replace("_", "/")
    if "/" in s:
        a, b = s.split("/", 1)
        return (a[:3], b[:3])
    # fallback: "USDJPY"
    if len(s) >= 6:
        return (s[:3], s[3:6])
    return ("", "")

def _fetch_ff_calendar(url: str, timeout: int = 12, ttl_sec: int = 1800) -> Tuple[Optional[List[dict]], str]:
    """
    Fetch weekly economic calendar JSON with very small cache to avoid rate limits.
    Returns (data, status_string).
    """
    now = time.time()
    if (_EVENT_CACHE["data"] is not None) and (_EVENT_CACHE["url"] == url) and (now - float(_EVENT_CACHE["ts"]) < ttl_sec):
        return _EVENT_CACHE["data"], "cache"

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (fx-analyzer; event-guard)"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(data, list):
            raise ValueError("calendar json is not a list")
        _EVENT_CACHE.update({"ts": now, "url": url, "data": data, "err": None})
        return data, "ok"
    except Exception as e:
        _EVENT_CACHE.update({"ts": now, "url": url, "data": None, "err": f"{type(e).__name__}: {e}"})
        return None, "fail"

def _parse_event_dt(item: dict) -> Optional[datetime]:
    """
    Try multiple schemas:
    - timestamp (epoch seconds)
    - datetime / date+time string
    """
    try:
        ts = item.get("timestamp", None)
        if isinstance(ts, (int, float)) and math.isfinite(float(ts)) and float(ts) > 0:
            # Many feeds use seconds; if it's too big, assume ms.
            tsv = float(ts)
            if tsv > 3e12:
                tsv = tsv / 1000.0
            return datetime.fromtimestamp(tsv, tz=timezone.utc)
    except Exception:
        pass

    # Common FF export fields: "date" + "time" (strings)
    dt_str = None
    for k in ("datetime", "date_time", "dt", "dateTime"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            dt_str = v.strip()
            break

    if dt_str is None:
        date_s = item.get("date", None)
        time_s = item.get("time", None)
        if isinstance(date_s, str) and date_s.strip():
            if isinstance(time_s, str) and time_s.strip():
                dt_str = f"{date_s.strip()} {time_s.strip()}"
            else:
                dt_str = date_s.strip()

    if not dt_str:
        return None

    # Parse with pandas (robust) then assume UTC if no tz
    try:
        dt = pd.to_datetime(dt_str, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def _compute_event_risk(
    pair: str,
    *,
    now_tz: str = "Asia/Tokyo",
    horizon_hours: int = 72,
    impacts: Optional[List[str]] = None,
    high_window_minutes: int = 60,
    url: str = _FF_CAL_URL_DEFAULT,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "ok": bool,
        "status": "ok|cache|fail",
        "err": str|None,
        "score": float,
        "factor": float (0..1),
        "window_high": bool,
        "next_high_hours": float|None,
        "upcoming": [ {dt_utc, currency, impact, title} ... ]  (<=10)
      }
    """
    impacts = impacts or ["High", "Medium"]
    a, b = _pair_to_ccys(pair)
    ccys = {a, b}

    data, status = _fetch_ff_calendar(url)
    if not data:
        return {
            "ok": False,
            "status": status,
            "err": _EVENT_CACHE.get("err"),
            "score": 0.0,
            "factor": 0.0,
            "window_high": False,
            "next_high_hours": None,
            "upcoming": [],
        }

    tz = ZoneInfo(now_tz)
    now_local = datetime.now(tz=tz)
    now_utc = now_local.astimezone(timezone.utc)

    # weights by impact
    w = {"High": 1.0, "Medium": 0.6, "Low": 0.3, "Holiday": 0.8, "Non-Economic": 0.2}

    upcoming = []
    for it in data:
        if not isinstance(it, dict):
            continue
        cur = str(it.get("currency") or it.get("ccy") or it.get("cur") or "").upper().strip()
        if not cur:
            # some schemas use "country" as currency code
            ctry = str(it.get("country") or "").upper().strip()
            if len(ctry) == 3:
                cur = ctry
        if cur and (cur not in ccys):
            continue

        impact = str(it.get("impact") or "").strip()
        if impacts and (impact not in impacts) and not (impact == "Holiday" and "Holiday" in impacts):
            continue

        dt = _parse_event_dt(it)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)

        hrs = (dt_utc - now_utc).total_seconds() / 3600.0
        if hrs < -1.0:
            continue
        if hrs > float(horizon_hours):
            continue

        title = str(it.get("title") or it.get("event") or it.get("name") or "").strip()
        upcoming.append({
            "dt_utc": dt_utc,
            "hours": float(hrs),
            "currency": cur,
            "impact": impact,
            "title": title,
        })

    upcoming.sort(key=lambda x: x["hours"])

    # score: sum(weight/(1+hours)) over upcoming events
    score = 0.0
    next_high = None
    window_high = False
    for ev in upcoming:
        impact = ev["impact"]
        hrs = float(ev["hours"])
        score += float(w.get(impact, 0.4)) / (1.0 + max(0.0, hrs))
        if impact == "High":
            if next_high is None:
                next_high = hrs
            # window check ±minutes
            if abs(hrs) * 60.0 <= float(high_window_minutes):
                window_high = True

    # normalize to factor 0..1 (heuristic)
    factor = _clamp(score / 1.25, 0.0, 1.0)

    # trim upcoming list for UI
    upcoming_ui = []
    for ev in upcoming[:10]:
        upcoming_ui.append({
            "dt_utc": ev["dt_utc"].isoformat(),
            "hours": float(ev["hours"]),
            "currency": ev["currency"],
            "impact": ev["impact"],
            "title": ev["title"],
        })

    return {
        "ok": True,
        "status": status,
        "err": None,
        "score": float(score),
        "factor": float(factor),
        "window_high": bool(window_high),
        "next_high_hours": (float(next_high) if next_high is not None else None),
        "upcoming": upcoming_ui,
    }

def _compute_weekend_risk(now_tz: str = "Asia/Tokyo") -> float:
    """
    Simple weekend gap risk proxy:
    - Fri evening (>=18:00 local) => 1.0
    - Sat/Sun => 1.0
    else 0.0
    """
    try:
        tz = ZoneInfo(now_tz)
        now = datetime.now(tz=tz)
        wd = now.weekday()  # Mon=0..Sun=6
        if wd >= 5:
            return 1.0
        if wd == 4 and now.hour >= 18:
            return 1.0
        return 0.0
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n//2)).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Simple Wilder ADX
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([(high - low).abs(),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx.fillna(0.0)

def _slope_norm(s: pd.Series, lookback: int = 10) -> float:
    # Normalized slope over lookback (last - first) / (std * sqrt(n))
    if len(s) < lookback + 1:
        return 0.0
    y = s.iloc[-lookback:].astype(float).values
    if np.all(np.isfinite(y)) is False:
        return 0.0
    dy = y[-1] - y[0]
    sd = float(np.std(y))
    if sd <= 1e-9:
        return 0.0
    return float(dy / (sd * math.sqrt(lookback)))

def _hh_hl_ok(df: pd.DataFrame, n: int = 20) -> bool:
    # crude HH/HL: compare last 2 swing highs and lows via rolling window peaks/valleys
    if len(df) < n + 5:
        return False
    close = df["Close"].astype(float)
    # detect local maxima/minima with a small window
    w = 3
    highs = (close.shift(w) < close) & (close.shift(-w) < close)
    lows = (close.shift(w) > close) & (close.shift(-w) > close)
    hi_idx = close[highs].tail(4).index
    lo_idx = close[lows].tail(4).index
    if len(hi_idx) < 2 or len(lo_idx) < 2:
        return False
    h1, h2 = close.loc[hi_idx[-2]], close.loc[hi_idx[-1]]
    l1, l2 = close.loc[lo_idx[-2]], close.loc[lo_idx[-1]]
    return (h2 > h1) and (l2 > l1)

def _breakout_strength(df: pd.DataFrame, n: int = 20) -> Tuple[bool, float]:
    # breakout if last close exceeds prior n-day high by >= 0.2*ATR
    if len(df) < n + 5:
        return False, 0.0
    close = df["Close"].astype(float)
    hi = df["High"].astype(float).rolling(n).max()
    atr = _atr(df, 14)
    last = close.iloc[-1]
    prev_hi = hi.iloc[-2]  # prior window high
    a = float(atr.iloc[-1]) if len(atr) else 0.0
    if not np.isfinite(prev_hi) or not np.isfinite(last) or a <= 0:
        return False, 0.0
    excess = last - prev_hi
    ok = excess > 0.2 * a
    strength = _clamp(excess / (1.5 * a), 0.0, 1.0) if a > 0 else 0.0
    return bool(ok), float(strength)

def _continuation_prob(df: pd.DataFrame, horizon: int = 5) -> Tuple[float, float]:
    """
    Simple continuation probability model (not ML-heavy):
    - Uses trend slope, adx, and breakout strength to estimate p(continue up) / p(continue down).
    """
    if len(df) < 60:
        return 0.5, 0.5
    close = df["Close"].astype(float)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    slope = _slope_norm(ema20, 12)
    adx = float(_adx(df, 14).iloc[-1])  # 0..100
    breakout_ok, bstr = _breakout_strength(df, 20)

    # Map to 0..1 features
    f_adx = _clamp((adx - 15.0) / 25.0, 0.0, 1.0)
    f_slope = _clamp((slope + 2.0) / 4.0, 0.0, 1.0)  # slope ~[-2,2] -> [0,1]
    f_trend = 0.55 * f_slope + 0.35 * f_adx + 0.10 * bstr
    # direction bias
    bias = 1.0 if ema20.iloc[-1] >= ema50.iloc[-1] else 0.0
    # base probabilities
    p_up = 0.35 + 0.55 * f_trend
    p_dn = 0.35 + 0.55 * (1.0 - f_trend)

    # apply direction bias and breakout
    if bias > 0.5:
        p_up += 0.05 + 0.10 * bstr
        p_dn -= 0.05
    else:
        p_dn += 0.05 + 0.10 * bstr
        p_up -= 0.05

    if breakout_ok:
        if bias > 0.5:
            p_up += 0.06
            p_dn -= 0.03
        else:
            p_dn += 0.06
            p_up -= 0.03

    p_up = _clamp(p_up, 0.05, 0.95)
    p_dn = _clamp(p_dn, 0.05, 0.95)
    return float(p_up), float(p_dn)

def _phase_label(df: pd.DataFrame) -> Tuple[str, float, float]:
    """
    Returns (phase_label, trend_strength 0..1, momentum_score -1..+1)
    """
    if len(df) < 60:
        return "UNKNOWN", 0.0, 0.0
    close = df["Close"].astype(float)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200) if len(close) >= 210 else _ema(close, 100)

    slope20 = _slope_norm(ema20, 12)  # roughly -? .. +?
    adx = float(_adx(df, 14).iloc[-1])
    breakout_ok, bstr = _breakout_strength(df, 20)
    hhhl = _hh_hl_ok(df, 30)

    # strength from ADX and breakout strength and slope magnitude
    s_adx = _clamp((adx - 12.0) / 28.0, 0.0, 1.0)
    s_slope = _clamp(abs(slope20) / 2.0, 0.0, 1.0)
    strength = _clamp(0.55 * s_adx + 0.30 * s_slope + 0.15 * bstr, 0.0, 1.0)

    # momentum: signed slope + small ema alignment
    align = 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
    mom = _clamp((slope20 / 2.0) + 0.20 * align, -1.0, 1.0)

    # classify
    if breakout_ok and strength >= 0.35:
        phase = "BREAKOUT_UP" if mom > 0 else "BREAKOUT_DOWN"
    else:
        if strength < 0.25:
            phase = "RANGE"
        else:
            phase = "UP_TREND" if mom > 0 else "DOWN_TREND"

    # refine by HH/HL
    if phase == "UP_TREND" and not hhhl and strength < 0.40:
        phase = "TRANSITION_UP"
    if phase == "DOWN_TREND" and strength < 0.40:
        phase = "TRANSITION_DOWN"

    return phase, float(strength), float(mom)

# ---------------------------------------------------------------------
# Core: strategy output
# ---------------------------------------------------------------------

@dataclass
class StrategyPlan:
    decision: str
    direction: str
    entry: float
    sl: float
    tp: float
    ev_raw: float
    ev_adj: float
    dynamic_threshold: float
    confidence: float
    why: str
    veto: List[str]
    ctx: Dict[str, Any]

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Backward compatible helper. main.py may call this."""
    if df is None or len(df) < 5:
        return {}
    close = df["Close"].astype(float)
    atr = _atr(df, 14)
    adx = _adx(df, 14)
    return {
        "ema20": float(_ema(close, 20).iloc[-1]),
        "ema50": float(_ema(close, 50).iloc[-1]),
        "atr14": float(atr.iloc[-1]) if len(atr) else 0.0,
        "adx14": float(adx.iloc[-1]) if len(adx) else 0.0,
    }

def get_ai_order_strategy(
    price_df: pd.DataFrame = None,
    pair: str = "",
    budget_yen: int = 0,
    context_data: Optional[Dict[str, Any]] = None,
    ext_features: Optional[Dict[str, Any]] = None,
    prefer_long_only: bool = False,
    api_key: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """
    main.py 互換の“完全版”エントリー判定（ctx依存を排除し、内部で必要な特徴量を計算します）。

    ✅ 互換性:
    - main.py が期待する key を全て返します（expected_R_ev / p_win_ev / veto_reasons / state_probs / ev_contribs など）
    - 旧key（entry/sl/tp, ev_raw/ev_adj, veto）も残します

    ✅ この関数が必ず使う入力:
    - 価格DF: price_df または kwargs(df/price_history/price_data) または context_data 内の (_df/df/price_df/price_history)
    - pair: pair または kwargs(pair_label/symbol/pair)

    価格DFが無い（or 60本未満）の場合は NO_TRADE で "データ不足" を返します。
    """

    # -----------------------------------------------------------------
    # 1) 引数の吸収（互換）
    # -----------------------------------------------------------------
    if not pair:
        pair = (kwargs.get("pair_label") or kwargs.get("symbol") or kwargs.get("pair") or "") or ""

    if ext_features is None:
        ext_features = kwargs.get("ext") or kwargs.get("external_features") or kwargs.get("ext_meta") or None

    if context_data is None:
        context_data = kwargs.get("ctx") or kwargs.get("context") or kwargs.get("_ctx") or None

    ctx_in = context_data or {}
    ext = ext_features or {}

    df = price_df
    if df is None:
        df = kwargs.get("df") or kwargs.get("price_history") or kwargs.get("price_data")

    # さらに、ctx内にdfを仕込んでいる場合（main側で ctx['_df']=df.tail(...) など）
    if df is None and isinstance(ctx_in, dict):
        df = ctx_in.get("_df") or ctx_in.get("df") or ctx_in.get("price_df") or ctx_in.get("price_history")

    # DataFrame 化
    if df is not None and not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = None

    # OHLC列名の正規化（lowercase等を吸収）
    if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
        cols = {c.lower(): c for c in df.columns}
        rename = {}
        for need in ["open", "high", "low", "close", "volume"]:
            if need in cols and cols[need] != need.capitalize():
                rename[cols[need]] = need.capitalize()
        if rename:
            df = df.rename(columns=rename)
        # たまに "Adj Close" のみ等があるので Close を補完
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

    # 必須列チェック
    need_cols = {"High", "Low", "Close"}
    if df is None or (not isinstance(df, pd.DataFrame)) or len(df) < 60 or (not need_cols.issubset(set(df.columns))):
        debug_cols = []
        try:
            debug_cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        except Exception:
            debug_cols = []
        return {
            "decision": "NO_TRADE",
            "direction": "LONG",
            "side": "BUY",
            "order_type": "—",
            "entry_type": "—",
            "entry": 0.0,
            "entry_price": 0.0,
            "sl": 0.0, "stop_loss": 0.0,
            "tp": 0.0, "take_profit": 0.0,
            "trail_sl": 0.0,
            "extend_factor": 1.0,
            "ev_raw": 0.0, "ev_adj": 0.0,
            "expected_R_ev_raw": 0.0,
            "expected_R_ev_adj": 0.0,
            "expected_R_ev": 0.0,
            "dynamic_threshold": float(_clamp(_safe_float(ctx_in.get("min_expected_R", 0.10), 0.10), 0.03, 0.30)),
            "gate_mode": "NO_DATA",
            "confidence": 0.0,
            "p_win": 0.0,
            "p_win_ev": 0.0,
            "why": "データ不足",
            "veto": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "veto_reasons": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "state_probs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "ev_contribs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "_ctx": {"pair": pair, "len": int(len(df)) if isinstance(df, pd.DataFrame) else 0, "cols": debug_cols},
        }

    # 念のためコピー
    df = df.copy()

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # -----------------------------------------------------------------
    # 2) 内部特徴量（ctx依存排除）
    # -----------------------------------------------------------------
    phase, strength, mom = _phase_label(df)
    horizon = int(_safe_float(ctx_in.get("horizon_days", 5), 5))
    p_up, p_dn = _continuation_prob(df, horizon=max(3, horizon))

    breakout_ok, breakout_strength = _breakout_strength(df, 20)
    hhhl_ok = _hh_hl_ok(df, 30)

    # -----------------------------------------------------------------
    # 3) 方向選択（トレンドを取り逃がさない）
    # -----------------------------------------------------------------
    direction = "LONG" if mom >= 0 else "SHORT"
    if prefer_long_only:
        direction = "LONG"
    # 強い上昇局面は “ショート禁止”
    if phase in ("UP_TREND", "BREAKOUT_UP", "TRANSITION_UP") and strength >= 0.33:
        direction = "LONG"
    if phase in ("DOWN_TREND", "BREAKOUT_DOWN", "TRANSITION_DOWN") and strength >= 0.33:
        direction = "SHORT"

    side = "BUY" if direction == "LONG" else "SELL"

    # -----------------------------------------------------------------
    # 4) リスクモデル（SL/TP）: ATRベース
    # -----------------------------------------------------------------
    atr14 = float(_atr(df, 14).iloc[-1])
    atr14 = max(atr14, 1e-6)

    lookback = 20
    recent_low = float(df["Low"].astype(float).tail(lookback).min())
    recent_high = float(df["High"].astype(float).tail(lookback).max())

    if direction == "LONG":
        sl = min(last - 1.2 * atr14, recent_low - 0.15 * atr14)
        tp = last + (2.2 * atr14 if phase in ("UP_TREND", "BREAKOUT_UP") else 1.6 * atr14)
    else:
        sl = max(last + 1.2 * atr14, recent_high + 0.15 * atr14)
        tp = last - (2.2 * atr14 if phase in ("DOWN_TREND", "BREAKOUT_DOWN") else 1.6 * atr14)

    entry = last

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 1e-9:
        risk = atr14
    rr = reward / risk

    # -----------------------------------------------------------------
    # 5) 勝率 proxy: 継続確率 × 強度（0.2..0.8にクリップ）
    # -----------------------------------------------------------------
    cont_best = max(p_up, p_dn)
    if direction == "LONG":
        p_win = 0.46 + 0.42 * _clamp((p_up - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (strength - 0.5)
    else:
        p_win = 0.46 + 0.42 * _clamp((p_dn - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (strength - 0.5)
    p_win = _clamp(p_win, 0.20, 0.80)

    # EV (R): EV = p*RR - (1-p)*1
    ev_raw = p_win * rr - (1.0 - p_win) * 1.0

    # -----------------------------------------------------------------
    # 6) 外部リスク（global_risk/war等）を macro_risk に畳み込み
    # -----------------------------------------------------------------
    # ext_features は main 側の feats を想定（global_risk_index / war_probability 等）
    gr = _safe_float(ext.get("global_risk_index", ext.get("global_risk", ext.get("risk_off", 0.35))), 0.35)
    war = _safe_float(ext.get("war_probability", ext.get("war", 0.0)), 0.0)
    macro_risk = _safe_float(ext.get("macro_risk_score", None), float("nan"))
    if not (isinstance(macro_risk, (int, float)) and math.isfinite(float(macro_risk))):
        macro_risk = _clamp(0.70 * gr + 0.30 * war, 0.0, 1.0)
    else:
        macro_risk = _clamp(float(macro_risk), 0.0, 1.0)

    
    # -----------------------------------------------------------------
    # 6.5) 経済指標/イベント & 週末ギャップのリスク（任意・無料フィード）
    # -----------------------------------------------------------------
    # 重要: 以前の版では macro_risk のみで、"指標の近接"（発表まで何時間か）を
    # 閾値や見送りに反映していませんでした。ここで明示的に加えます。
    event_guard_enable = bool(ctx_in.get("event_guard_enable", True))
    event_block_window = bool(ctx_in.get("event_block_high_impact_window", True))
    event_horizon_hours = int(_safe_float(ctx_in.get("event_horizon_hours", 72), 72))
    event_window_minutes = int(_safe_float(ctx_in.get("event_window_minutes", 60), 60))
    event_impacts = ctx_in.get("event_impacts", None)
    if not isinstance(event_impacts, list) or not event_impacts:
        event_impacts = ["High", "Medium"]
    event_calendar_url = str(ctx_in.get("event_calendar_url", _FF_CAL_URL_DEFAULT) or _FF_CAL_URL_DEFAULT)

    ev_meta = {"ok": False, "status": "off", "err": None, "score": 0.0, "factor": 0.0,
               "window_high": False, "next_high_hours": None, "upcoming": []}
    if event_guard_enable:
        try:
            ev_meta = _compute_event_risk(
                pair,
                now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"),
                horizon_hours=event_horizon_hours,
                impacts=[str(x) for x in event_impacts],
                high_window_minutes=event_window_minutes,
                url=event_calendar_url,
            )
        except Exception as e:
            ev_meta = {"ok": False, "status": "fail", "err": f"{type(e).__name__}: {e}", "score": 0.0, "factor": 0.0,
                       "window_high": False, "next_high_hours": None, "upcoming": []}

    weekend_risk = float(_compute_weekend_risk(now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo")))
# リスク調整 EV（表示用）
    risk_penalty = 0.12 + 0.55 * macro_risk   # 0.12..0.67
    ev_adj = ev_raw - 0.20 * risk_penalty

    # -----------------------------------------------------------------
    # 7) 動的閾値（フェーズ別） + リスク時に少し上げる
    # -----------------------------------------------------------------
    base_thr = _safe_float(ctx_in.get("dynamic_threshold_base", None), float("nan"))
    if not (isinstance(base_thr, (int, float)) and math.isfinite(float(base_thr))):
        base_thr = _safe_float(ctx_in.get("min_expected_R", 0.08), 0.08)
    base_thr = _clamp(float(base_thr), 0.03, 0.25)

    thr_mult = 1.0
    if phase in ("UP_TREND", "DOWN_TREND", "TRANSITION_UP", "TRANSITION_DOWN"):
        thr_mult -= 0.18 * strength
    if phase.startswith("BREAKOUT"):
        thr_mult -= 0.25 * max(strength, breakout_strength)
    if phase == "RANGE":
        thr_mult += 0.10

    dynamic_threshold = base_thr * thr_mult
    # リスク時は閾値を上げる（ただし上げ過ぎない）
    dynamic_threshold = dynamic_threshold + 0.04 * macro_risk
    dynamic_threshold = _clamp(dynamic_threshold, 0.02, 0.30)

    # イベント/週末リスクも閾値に反映（主ゲートが raw+mom の場合でも抑止できるように）
    try:
        dynamic_threshold = dynamic_threshold + 0.06 * float(ev_meta.get("factor", 0.0) or 0.0) + 0.03 * float(weekend_risk or 0.0)
    except Exception:
        pass
    dynamic_threshold = _clamp(dynamic_threshold, 0.02, 0.30)


    # -----------------------------------------------------------------
    # 8) モメンタム加点（上限 +0.06R）
    # -----------------------------------------------------------------
    mom_bonus = 0.0
    if direction == "LONG" and mom > 0:
        mom_bonus = 0.06 * _clamp(strength, 0.0, 1.0) * _clamp(p_up, 0.0, 1.0)
    if direction == "SHORT" and mom < 0:
        mom_bonus = 0.06 * _clamp(strength, 0.0, 1.0) * _clamp(p_dn, 0.0, 1.0)
    mom_bonus = _clamp(mom_bonus, 0.0, 0.06)

    ev_gate = ev_raw + mom_bonus  # gateに使うEV（従来要望の "raw+mom"）

    # -----------------------------------------------------------------
    # 9) ブレイク別ゲート（救済）：HH/HL または breakout + 継続確率 + 強度
    # -----------------------------------------------------------------
    breakout_pass = bool((breakout_ok or hhhl_ok) and (cont_best >= 0.58) and (max(strength, breakout_strength) >= 0.38) and (macro_risk <= 0.82))

    # -----------------------------------------------------------------
    # 10) 信頼度（0..1）
    # -----------------------------------------------------------------
    confidence = _clamp(0.32 + 0.38 * strength + 0.20 * (cont_best - 0.5) + 0.10 * (rr - 1.0), 0.0, 1.0)

    veto: List[str] = []
    why = ""
    gate_mode = "raw+mom"

    # 高インパクト指標の“直前直後”は、スリッページ/ギャップの不確実性が高いので強制見送り（任意）
    if event_guard_enable and event_block_window and bool(ev_meta.get("window_high", False)):
        decision = "NO_TRADE"
        gate_mode = "event_block"
        reason = f"高インパクト指標の前後（±{event_window_minutes}分）のため見送り"
        veto.append(reason)
        why = reason
    elif ev_gate >= dynamic_threshold:
        decision = "TRADE"
        why = f"EV通過(raw+mom): {ev_gate:+.3f} ≥ 動的閾値 {dynamic_threshold:.3f}"
    elif breakout_pass and (ev_gate >= dynamic_threshold - 0.04):
        decision = "TRADE"
        gate_mode = "breakout_rescue"
        why = f"BREAKOUT通過: EV {ev_gate:+.3f} / 閾値 {dynamic_threshold:.3f}（救済）"
    else:
        decision = "NO_TRADE"
        veto.append(f"EV不足(raw+mom): {ev_gate:+.3f} < 動的閾値 {dynamic_threshold:.3f}")
        veto.insert(0, f"EV不足(raw): {ev_raw:+.3f} < 動的閾値 {dynamic_threshold:.3f}")
        why = veto[0]

    # -----------------------------------------------------------------
    # 11) 状態確率 / EV内訳（UI用）
    # -----------------------------------------------------------------
    # score -> normalize
    s_up = max(0.0, p_up * (0.55 + 0.75 * strength) + max(0.0, mom) * 0.10)
    s_dn = max(0.0, p_dn * (0.55 + 0.75 * strength) + max(0.0, -mom) * 0.10)
    s_range = max(0.0, (1.0 - strength) * 0.95 + 0.05)
    s_risk = max(0.0, macro_risk * 1.15 + (1.0 - cont_best) * 0.10)

    tot = s_up + s_dn + s_range + s_risk
    if tot <= 1e-12:
        state_probs = {"trend_up": 0.25, "trend_down": 0.25, "range": 0.25, "risk_off": 0.25}
    else:
        state_probs = {
            "trend_up": float(s_up / tot),
            "trend_down": float(s_dn / tot),
            "range": float(s_range / tot),
            "risk_off": float(s_risk / tot),
        }

    # 状態別R（粗い近似。目的は“未計算に見えない”UIと、EVの説明可能性）
    if direction == "LONG":
        r_up = max(0.2, rr * 0.85)
        r_dn = -1.0
    else:
        r_dn = max(0.2, rr * 0.85)
        r_up = -1.0
    r_range = (0.12 * rr - 0.35)
    r_riskoff = -0.75

    ev_contribs = {
        "trend_up": float(state_probs["trend_up"] * r_up),
        "trend_down": float(state_probs["trend_down"] * r_dn),
        "range": float(state_probs["range"] * r_range),
        "risk_off": float(state_probs["risk_off"] * r_riskoff),
    }

    # -----------------------------------------------------------------
    # 12) ctx（デバッグ/可視化用）
    # -----------------------------------------------------------------
    ctx_out = {
        "pair": pair,
        "phase_label": phase,
        "trend_strength": float(strength),
        "momentum_score": float(mom),
        "cont_p_up": float(p_up),
        "cont_p_dn": float(p_dn),
        "hh_hl_ok": bool(hhhl_ok),
        "breakout_ok": bool(breakout_ok),
        "breakout_strength": float(breakout_strength),
        "rr": float(rr),
        "p_win": float(p_win),
        "macro_risk_score": float(macro_risk),
        "event_risk_score": float(ev_meta.get("score", 0.0) or 0.0),
        "event_risk_factor": float(ev_meta.get("factor", 0.0) or 0.0),
        "event_window_high": bool(ev_meta.get("window_high", False)),
        "event_next_high_hours": (float(ev_meta.get("next_high_hours")) if ev_meta.get("next_high_hours") is not None else None),
        "event_feed_status": str(ev_meta.get("status", "") or ""),
        "event_feed_error": str(ev_meta.get("err", "") or ""),
        "event_upcoming": (ev_meta.get("upcoming", []) or []),
        "weekend_risk": float(weekend_risk),
        "mom_bonus": float(mom_bonus),
        "dynamic_threshold": float(dynamic_threshold),
        "dynamic_threshold_base": float(base_thr),
        "dynamic_threshold_mult": float(thr_mult),
        "breakout_pass": bool(breakout_pass),
        "cont_best": float(cont_best),
        "len": int(len(df)),
    }

    # Trail SL: エントリーから0.5R戻し（見せ方用）
    trail_sl = sl
    try:
        dist_sl = abs(entry - sl)
        if dist_sl > 0:
            trail_sl = entry - 0.5*dist_sl if direction == "LONG" else (entry + 0.5*dist_sl)
    except Exception:
        trail_sl = sl

    # 返却（main互換キーを全て用意）
    plan = {
        "decision": decision,
        "direction": direction,
        "side": side,
        "order_type": "MARKET",
        "entry_type": "MARKET_NOW",

        "entry": float(entry),
        "entry_price": float(entry),
        "sl": float(sl),
        "stop_loss": float(sl),
        "tp": float(tp),
        "take_profit": float(tp),

        "trail_sl": float(trail_sl),
        "extend_factor": 1.0,

        "ev_raw": float(ev_raw),
        "ev_adj": float(ev_adj),

        "expected_R_ev_raw": float(ev_raw),
        "expected_R_ev_adj": float(ev_adj),
        "expected_R_ev": float(ev_gate),

        "dynamic_threshold": float(dynamic_threshold),
        "gate_mode": gate_mode,

        "confidence": float(confidence),
        "p_win": float(p_win),
        "p_win_ev": float(p_win),

        "why": why,
        "veto": list(veto),
        "veto_reasons": list(veto),

        "state_probs": state_probs,
        "ev_contribs": ev_contribs,

        "_ctx": ctx_out,
    }
    return plan
# End of file
