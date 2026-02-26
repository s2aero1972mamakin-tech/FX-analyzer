
# logic_fixed27_trend_entry_engine.py
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
    price_df: pd.DataFrame,
    pair: str = "",
    budget_yen: int = 0,
    context_data: Optional[Dict[str, Any]] = None,
    ext_features: Optional[Dict[str, Any]] = None,
    prefer_long_only: bool = False,
) -> Dict[str, Any]:
    """
    Returns dict compatible with existing main.py:
      {
        "decision": "TRADE"/"NO_TRADE",
        "direction": "LONG"/"SHORT",
        "entry": ..., "sl": ..., "tp": ...,
        "ev_raw": ..., "ev_adj": ...,
        "dynamic_threshold": ...,
        "confidence": ...,
        "why": ..., "veto": [...],
        "_ctx": {...}
      }
    """
    df = price_df.copy() if price_df is not None else None
    if df is None or len(df) < 60:
        return {
            "decision": "NO_TRADE",
            "direction": "LONG",
            "entry": 0.0, "sl": 0.0, "tp": 0.0,
            "ev_raw": 0.0, "ev_adj": 0.0,
            "dynamic_threshold": 0.10,
            "confidence": 0.0,
            "why": "データ不足",
            "veto": ["データ不足（最低60本必要）"],
            "_ctx": {},
        }

    ctx_in = context_data or {}
    ext = ext_features or {}

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # Compute phase, strength, momentum and continuation probabilities internally
    phase, strength, mom = _phase_label(df)
    p_up, p_dn = _continuation_prob(df, horizon=int(_safe_float(ctx_in.get("horizon_days", 5), 5)))

    breakout_ok, breakout_strength = _breakout_strength(df, 20)
    hhhl_ok = _hh_hl_ok(df, 30)

    # Direction: prefer with momentum/phase
    direction = "LONG" if mom >= 0 else "SHORT"
    if prefer_long_only:
        direction = "LONG"
    # Hard block: in strong uptrend/breakout, do not short (reduces "uptrend but short EV negative" issue)
    if phase in ("UP_TREND", "BREAKOUT_UP") and strength >= 0.35:
        direction = "LONG"

    # Basic risk model: SL at 1.2*ATR behind recent swing, TP = 2.0*ATR in trend, else 1.5*ATR
    atr14 = float(_atr(df, 14).iloc[-1])
    atr14 = max(atr14, 1e-6)
    lookback = 20
    recent_low = float(df["Low"].astype(float).tail(lookback).min())
    recent_high = float(df["High"].astype(float).tail(lookback).max())

    if direction == "LONG":
        sl = min(last - 1.2 * atr14, recent_low - 0.15 * atr14)
        tp_base = last + (2.2 * atr14 if phase.startswith("UP") or phase.startswith("BREAKOUT") else 1.6 * atr14)
    else:
        sl = max(last + 1.2 * atr14, recent_high + 0.15 * atr14)
        tp_base = last - (2.2 * atr14 if phase.startswith("DOWN") or phase.startswith("BREAKOUT") else 1.6 * atr14)

    entry = last  # market entry assumption (main can convert to limit if desired)
    tp = tp_base

    # Compute R-multiples
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 1e-9:
        risk = atr14
    rr = reward / risk

    # Win probability proxy: combine continuation prob and strength
    if direction == "LONG":
        p_win = 0.45 + 0.40 * _clamp((p_up - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (strength - 0.5)
    else:
        p_win = 0.45 + 0.40 * _clamp((p_dn - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (strength - 0.5)
    p_win = _clamp(p_win, 0.20, 0.80)

    # EV in R: EV = p*RR - (1-p)*1
    ev_raw = p_win * rr - (1.0 - p_win) * 1.0

    # External risk penalty (0..1). If not present, assume moderate-low.
    macro_risk = _clamp(_safe_float(ext.get("macro_risk_score", ext.get("risk_off", 0.35)), 0.35), 0.0, 1.0)
    risk_penalty = 0.15 + 0.55 * macro_risk  # 0.15..0.70
    ev_adj = ev_raw - 0.20 * risk_penalty  # moderate penalty

    # Dynamic threshold base
    base_thr = _clamp(_safe_float(ctx_in.get("dynamic_threshold_base", 0.08), 0.08), 0.03, 0.20)

    # Phase-based threshold adjustment (your requested item #1)
    thr_mult = 1.0
    if phase in ("UP_TREND", "DOWN_TREND", "TRANSITION_UP", "TRANSITION_DOWN"):
        thr_mult -= 0.15 * strength  # loosen in trends
    if phase.startswith("BREAKOUT"):
        thr_mult -= 0.25 * max(strength, breakout_strength)  # more loosen in breakout
    if phase == "RANGE":
        thr_mult += 0.10  # tighten in range
    dynamic_threshold = _clamp(base_thr * thr_mult, 0.02, 0.25)

    # Momentum bonus capped (your requested item #2)
    mom_bonus = 0.0
    if direction == "LONG" and mom > 0:
        mom_bonus = 0.06 * _clamp(strength, 0.0, 1.0) * _clamp(p_up, 0.0, 1.0)
    if direction == "SHORT" and mom < 0:
        mom_bonus = 0.06 * _clamp(strength, 0.0, 1.0) * _clamp(p_dn, 0.0, 1.0)
    mom_bonus = _clamp(mom_bonus, 0.0, 0.06)
    ev_eff = ev_raw + mom_bonus  # use raw gate with bonus (conservative)

    # Breakout gate (your requested item #3)
    cont_best = max(p_up, p_dn)
    breakout_pass = bool((breakout_ok or hhhl_ok) and (cont_best >= 0.62) and (max(strength, breakout_strength) >= 0.45) and (macro_risk <= 0.75))

    # Confidence (0..1)
    confidence = _clamp(0.35 + 0.35 * strength + 0.20 * (cont_best - 0.5) + 0.10 * (rr - 1.0), 0.0, 1.0)

    veto: List[str] = []
    why = ""

    # Primary gate: EV_eff vs threshold OR breakout_pass (allow slight EV miss)
    if ev_eff >= dynamic_threshold:
        decision = "TRADE"
        why = f"EV通過(raw+mom): {ev_eff:+.3f} ≥ 動的閾値 {dynamic_threshold:.3f}"
    elif breakout_pass and (ev_eff >= dynamic_threshold - 0.03):
        decision = "TRADE"
        why = f"BREAKOUT通過: EV {ev_eff:+.3f} / 閾値 {dynamic_threshold:.3f}（緩和適用）"
    else:
        decision = "NO_TRADE"
        veto.append(f"EV不足(raw+mom): {ev_eff:+.3f} < 動的閾値 {dynamic_threshold:.3f}")
        # Keep older phrasing too, so existing UI matches:
        veto.insert(0, f"EV不足(raw): {ev_raw:+.3f} < 動的閾値 {dynamic_threshold:.3f}")
        why = veto[0]

    # Attach ctx for UI and debugging (always populated; avoids "ctx missing" silent failure)
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
        "mom_bonus": float(mom_bonus),
        "dynamic_threshold": float(dynamic_threshold),
        "dynamic_threshold_base": float(base_thr),
        "dynamic_threshold_mult": float(thr_mult),
        "breakout_pass": bool(breakout_pass),
        "cont_best": float(cont_best),
    }

    return {
        "decision": decision,
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "ev_raw": float(ev_raw),
        "ev_adj": float(ev_adj),
        "dynamic_threshold": float(dynamic_threshold),
        "confidence": float(confidence),
        "why": why,
        "veto": veto,
        "_ctx": ctx_out,
    }

# End of file
