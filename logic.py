


# logic_fixed28_trend_entry_engine_compat.py
# Drop-in replacement for logic.py
# - Adds self-contained: HH/HL detection, breakout strength, continuation probability, phase-aware EV thresholding,
#   momentum bonus (capped), and breakout gate.
# - Designed to be backward-compatible with existing main.py callers.
#
# NOTE: This module does not depend on ctx keys being passed from main.py; it computes needed signals from price history.

from __future__ import annotations





# ------------------- B-RANK + PROP AI PATCH -------------------
def _range_center_penalty(range_pos: float) -> float:
    try:
        d = abs(float(range_pos) - 0.5)
        if d < 0.15:
            return 0.35
        if d < 0.25:
            return 0.15
        return 0.0
    except Exception:
        return 0.0

def _event_unknown_adjust(ev_meta: dict) -> float:
    try:
        ok = bool((ev_meta or {}).get("ok", False))
        return 0.15 if not ok else 0.0
    except Exception:
        return 0.15

def _liquidity_sweep(df) -> bool:
    try:
        if df is None or len(df) < 5:
            return False
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        prev_high = float(h.iloc[-2])
        prev_low = float(l.iloc[-2])
        last_high = float(h.iloc[-1])
        last_low = float(l.iloc[-1])
        last_close = float(c.iloc[-1])
        if last_high > prev_high and last_close < prev_high:
            return True
        if last_low < prev_low and last_close > prev_low:
            return True
        return False
    except Exception:
        return False

def _volatility_expansion(df) -> float:
    try:
        c = df["Close"].astype(float)
        if len(c) < 30:
            return 0.0
        v10 = float(c.pct_change().rolling(10).std().iloc[-1] or 0.0)
        v30 = float(c.pct_change().rolling(30).std().iloc[-1] or 0.0)
        if v30 <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, v10 / v30))
    except Exception:
        return 0.0

def _market_regime(df) -> str:
    try:
        if df is None or len(df) < 100:
            return "UNKNOWN"
        c = df["Close"].astype(float)
        ma50 = float(c.rolling(50).mean().iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float(c.rolling(100).mean().iloc[-1])
        if ma50 > ma200:
            return "BULL"
        if ma50 < ma200:
            return "BEAR"
        return "RANGE"
    except Exception:
        return "UNKNOWN"
# ---------------------------------------------------------------

# --- AI Quality / Timing / Failure Feature Patch (Auto-added) ---
def _quality_decay(strength, breakout_ok, hhhl_ok, confidence):
    q = (
        0.40 * float(strength) +
        0.30 * float(confidence) +
        0.20 * (1.0 if breakout_ok else 0.0) +
        0.10 * (1.0 if hhhl_ok else 0.0)
    )
    q = max(0.35, min(1.0, q))
    return q

def _entry_timing_filter(df, direction):
    try:
        o = df["Open"].astype(float)
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        if len(c) < 3:
            return True
        if direction == "LONG":
            cond = (c.iloc[-1] > c.iloc[-2]) and (c.iloc[-1] > o.iloc[-1])
        else:
            cond = (c.iloc[-1] < c.iloc[-2]) and (c.iloc[-1] < o.iloc[-1])
        return bool(cond)
    except Exception:
        return True

def _failure_features(df):
    try:
        c = df["Close"].astype(float)
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        o = df["Open"].astype(float)
        if len(c) < 5:
            return 0.0
        body = abs(c.iloc[-1] - o.iloc[-1])
        rng = max(1e-9, h.iloc[-1] - l.iloc[-1])
        wick_ratio = (rng - body) / rng
        pullback = abs(c.iloc[-1] - c.iloc[-3]) / max(1e-9, rng)
        penalty = min(0.4, wick_ratio * 0.5 + pullback * 0.3)
        return penalty
    except Exception:
        return 0.0
# --- END PATCH ---

def _liquidity_pool_tp(df: pd.DataFrame, direction: str, lookback: int = 40) -> Optional[float]:
    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())
        return recent_high if str(direction).upper() == "LONG" else recent_low
    except Exception:
        return None

def _regime_tp_multiple(phase_label: str) -> float:
    try:
        phase = str(phase_label or "")
        if phase.startswith("BREAKOUT"):
            return 3.0
        if phase in ("UP_TREND", "DOWN_TREND"):
            return 2.5
        if "TRANSITION" in phase:
            return 1.8
        if phase == "RANGE":
            return 1.3
        return 2.0
    except Exception:
        return 2.0

def _resolve_trade_profile(ctx_in: Dict[str, Any]) -> Dict[str, Any]:
    try:
        interval = str((ctx_in or {}).get("price_interval") or (ctx_in or {}).get("interval") or "").strip().lower()
    except Exception:
        interval = ""
    try:
        tf_label = str((ctx_in or {}).get("timeframe_mode") or (ctx_in or {}).get("trade_profile") or "")
    except Exception:
        tf_label = ""
    horizon_days = max(1, int(_safe_float((ctx_in or {}).get("horizon_days", 5), 5)))

    is_intraday = bool(interval in ("1h", "60m", "30m", "15m") or ("デイトレ" in tf_label) or str((ctx_in or {}).get("is_intraday", False)).lower() in ("1", "true", "yes"))
    is_position = bool(interval == "1wk" or ("中長期" in tf_label) or horizon_days >= 21)

    if is_intraday:
        profile = {
            "name": "DAYTRADE",
            "interval": interval or "1h",
            "is_intraday": True,
            "lookback": 12,
            "liquidity_lookback": 18,
            "sl_atr_mult": 0.95,
            "sl_buffer_atr": 0.10,
            "tp_breakout": 1.85,
            "tp_trend": 1.55,
            "tp_transition": 1.30,
            "tp_range": 1.05,
            "event_preblock_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_preblock_hours", 6.0), 6.0), 2.0, 24.0)),
            "event_market_ban_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_market_ban_hours", 4.0), 4.0), 1.0, 12.0)),
            "threshold_bias": 0.015,
            "rr_floor": max(1.05, float(_safe_float((ctx_in or {}).get("rr_min_floor", 1.0), 1.0))),
            "partial_r": 0.60,
            "trail_trigger_r": 0.90,
            "be_trigger_r": 0.40,
            "max_hold_hours": float(_clamp(_safe_float((ctx_in or {}).get("max_hold_hours", 12.0), 12.0), 4.0, 36.0)),
            "stale_after_hours": float(_clamp(_safe_float((ctx_in or {}).get("stale_after_hours", 4.0), 4.0), 1.0, 12.0)),
        }
        profile["max_hold_bars"] = int(max(4, round(profile["max_hold_hours"])))
        return profile

    if is_position:
        return {
            "name": "POSITION",
            "interval": interval or "1wk",
            "is_intraday": False,
            "lookback": 26,
            "liquidity_lookback": 52,
            "sl_atr_mult": 1.35,
            "sl_buffer_atr": 0.18,
            "tp_breakout": 3.20,
            "tp_trend": 2.80,
            "tp_transition": 2.10,
            "tp_range": 1.40,
            "event_preblock_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_preblock_hours", 36.0), 36.0), 6.0, 96.0)),
            "event_market_ban_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_market_ban_hours", 18.0), 18.0), 6.0, 48.0)),
            "threshold_bias": -0.005,
            "rr_floor": max(1.00, float(_safe_float((ctx_in or {}).get("rr_min_floor", 1.0), 1.0))),
            "partial_r": 0.80,
            "trail_trigger_r": 1.30,
            "be_trigger_r": 0.55,
            "max_hold_hours": float(_clamp(_safe_float((ctx_in or {}).get("max_hold_hours", 24.0 * 20.0), 24.0 * 20.0), 24.0 * 5.0, 24.0 * 90.0)),
            "stale_after_hours": float(_clamp(_safe_float((ctx_in or {}).get("stale_after_hours", 24.0 * 5.0), 24.0 * 5.0), 24.0, 24.0 * 20.0)),
            "max_hold_bars": int(max(4, round(max(5.0, horizon_days / 5.0)))),
        }

    return {
        "name": "SWING",
        "interval": interval or "1d",
        "is_intraday": False,
        "lookback": 20,
        "liquidity_lookback": 40,
        "sl_atr_mult": 1.20,
        "sl_buffer_atr": 0.15,
        "tp_breakout": 3.00,
        "tp_trend": 2.50,
        "tp_transition": 1.80,
        "tp_range": 1.30,
        "event_preblock_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_preblock_hours", 24.0), 24.0), 6.0, 72.0)),
        "event_market_ban_hours": float(_clamp(_safe_float((ctx_in or {}).get("event_market_ban_hours", 12.0), 12.0), 3.0, 24.0)),
        "threshold_bias": 0.0,
        "rr_floor": max(1.00, float(_safe_float((ctx_in or {}).get("rr_min_floor", 1.0), 1.0))),
        "partial_r": 0.70,
        "trail_trigger_r": 1.10,
        "be_trigger_r": 0.50,
        "max_hold_hours": float(_clamp(_safe_float((ctx_in or {}).get("max_hold_hours", 24.0 * 20.0), 24.0 * 20.0), 24.0 * 3.0, 24.0 * 60.0)),
        "stale_after_hours": float(_clamp(_safe_float((ctx_in or {}).get("stale_after_hours", 24.0 * 3.0), 24.0 * 3.0), 12.0, 24.0 * 14.0)),
        "max_hold_bars": int(max(4, horizon_days)),
    }


def _profile_tp_multiple(profile: Dict[str, Any], phase_label: str, breakout_ok: bool, strength: float) -> float:
    try:
        p = str((profile or {}).get("name") or "SWING")
        phase = str(phase_label or "")
        if breakout_ok or phase.startswith("BREAKOUT"):
            mult = float((profile or {}).get("tp_breakout", _regime_tp_multiple(phase_label)))
        elif phase in ("UP_TREND", "DOWN_TREND"):
            mult = float((profile or {}).get("tp_trend", _regime_tp_multiple(phase_label)))
        elif "TRANSITION" in phase:
            mult = float((profile or {}).get("tp_transition", _regime_tp_multiple(phase_label)))
        elif phase == "RANGE":
            mult = float((profile or {}).get("tp_range", _regime_tp_multiple(phase_label)))
        else:
            mult = float(_regime_tp_multiple(phase_label))
        if p == "DAYTRADE":
            mult += 0.15 * float(_clamp(strength - 0.55, -0.30, 0.40))
        return float(_clamp(mult, 0.90, 4.20))
    except Exception:
        return float(_regime_tp_multiple(phase_label))


def _session_structure_levels(df: Optional[pd.DataFrame], fallback_bars: int = 32) -> Dict[str, Optional[float]]:
    try:
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
            return {"session_high": None, "session_low": None, "recent_high": None, "recent_low": None}
        d = df.copy().tail(max(24, int(fallback_bars)))
        recent = d.tail(max(12, min(len(d), int(fallback_bars))))
        session = recent
        try:
            idx = d.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                idx_jst = idx.tz_convert("Asia/Tokyo") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("Asia/Tokyo")
                day_mask = pd.Series(idx_jst.date == idx_jst[-1].date(), index=d.index)
                same_day = d.loc[day_mask]
                if isinstance(same_day, pd.DataFrame) and len(same_day) >= 8:
                    session = same_day.tail(max(8, min(len(same_day), int(fallback_bars))))
        except Exception:
            session = recent
        return {
            "session_high": float(session["High"].astype(float).max()) if not session.empty else None,
            "session_low": float(session["Low"].astype(float).min()) if not session.empty else None,
            "recent_high": float(recent["High"].astype(float).max()) if not recent.empty else None,
            "recent_low": float(recent["Low"].astype(float).min()) if not recent.empty else None,
        }
    except Exception:
        return {"session_high": None, "session_low": None, "recent_high": None, "recent_low": None}


def _compute_daytrade_tp1(
    entry: float,
    sl: float,
    tp2: float,
    direction: str,
    fast_df: Optional[pd.DataFrame],
    fast_probe: Optional[Dict[str, Any]],
    liq_tp: Optional[float],
    phase_label: str,
    strength: float,
) -> Optional[float]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp2 = float(tp2)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return None
        phase = str(phase_label or "")
        prof_strength = float(_clamp(strength, 0.0, 1.0))
        fast_atr = max(float((fast_probe or {}).get("atr14") or risk), 1e-9)
        struct = _session_structure_levels(fast_df, fallback_bars=32)
        min_move = max(0.30 * risk, 0.22 * fast_atr)
        base_r = 0.52 + 0.12 * prof_strength
        if phase == "RANGE":
            base_r -= 0.08
        if phase.startswith("BREAKOUT"):
            base_r += 0.06
        base_r = float(_clamp(base_r, 0.42, 0.82))
        if str(direction).upper() == "LONG":
            fallback_tp1 = entry + base_r * risk
            cap = entry + 0.88 * max(0.0, tp2 - entry)
            candidates = [struct.get("recent_high"), struct.get("session_high"), float(liq_tp) if liq_tp is not None else None, fallback_tp1]
            valid = []
            for v in candidates:
                if v is None:
                    continue
                vv = float(v)
                if vv >= entry + min_move and vv < tp2:
                    valid.append(vv)
            tp1 = min(valid) if valid else min(fallback_tp1, cap)
            tp1 = min(tp1, cap)
            tp1 = max(tp1, entry + min_move)
        else:
            fallback_tp1 = entry - base_r * risk
            cap = entry - 0.88 * max(0.0, entry - tp2)
            candidates = [struct.get("recent_low"), struct.get("session_low"), float(liq_tp) if liq_tp is not None else None, fallback_tp1]
            valid = []
            for v in candidates:
                if v is None:
                    continue
                vv = float(v)
                if vv <= entry - min_move and vv > tp2:
                    valid.append(vv)
            tp1 = max(valid) if valid else max(fallback_tp1, cap)
            tp1 = max(tp1, cap)
            tp1 = min(tp1, entry - min_move)
        if str(direction).upper() == "LONG" and tp1 >= tp2:
            tp1 = entry + 0.55 * (tp2 - entry)
        if str(direction).upper() != "LONG" and tp1 <= tp2:
            tp1 = entry - 0.55 * (entry - tp2)
        return float(tp1)
    except Exception:
        return None


def _cap_daytrade_tp1_to_liquidity(entry: float, tp1: Optional[float], tp2: float, direction: str, liq_tp: Optional[float]) -> Tuple[Optional[float], bool]:
    try:
        if tp1 is None or liq_tp is None:
            return (float(tp1) if tp1 is not None else None), False
        tp1 = float(tp1)
        tp2 = float(tp2)
        liq_tp = float(liq_tp)
        if str(direction).upper() == "LONG":
            capped = min(tp1, liq_tp, tp2)
            return float(capped), bool(capped < tp1 - 1e-12)
        capped = max(tp1, liq_tp, tp2)
        return float(capped), bool(capped > tp1 + 1e-12)
    except Exception:
        return (float(tp1) if tp1 is not None else None), False


def _compress_daytrade_tp2(entry: float, sl: float, tp2: float, direction: str, mtf_alignment_score: float, fast_tf_dir_ok: Optional[bool], micro_tf_dir_ok: Optional[bool], fast_df: Optional[pd.DataFrame], liq_tp: Optional[float]) -> Tuple[float, float, bool, bool, str]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp2 = float(tp2)
        risk = abs(entry - sl)
        dist = abs(tp2 - entry)
        if risk <= 1e-9 or dist <= 1e-9:
            return float(tp2), 1.0, False, False, "none"
        hard_cap = bool(float(mtf_alignment_score) <= -0.40 and fast_tf_dir_ok is False and micro_tf_dir_ok is False and liq_tp is not None)
        if hard_cap:
            liq = float(liq_tp)
            capped = min(tp2, liq) if str(direction).upper() == "LONG" else max(tp2, liq)
            return float(capped), float(abs(capped - entry) / max(dist, 1e-9)), True, True, "intraday_liquidity_cap"
        compress_needed = (micro_tf_dir_ok is False) or (float(mtf_alignment_score) < 0.35)
        if not compress_needed:
            return float(tp2), 1.0, False, False, "none"
        factor = 1.0
        if micro_tf_dir_ok is False:
            factor -= 0.22
        if fast_tf_dir_ok is False:
            factor -= 0.10
        if float(mtf_alignment_score) < 0.35:
            factor -= min(0.24, (0.35 - float(mtf_alignment_score)) * 0.28)
        factor = float(_clamp(factor, 0.45, 0.90))
        struct = _session_structure_levels(fast_df, fallback_bars=24)
        min_reward = max(0.90 * risk, 0.58 * dist)
        if str(direction).upper() == "LONG":
            compressed = entry + max(min_reward, factor * dist)
            overheads = [struct.get("recent_high"), struct.get("session_high"), float(liq_tp) if liq_tp is not None else None]
            overheads = [float(v) for v in overheads if v is not None and float(v) > entry + 0.80 * risk]
            if overheads:
                compressed = min(compressed, min(overheads))
            compressed = min(compressed, tp2)
            compressed = max(compressed, entry + 0.90 * risk)
        else:
            compressed = entry - max(min_reward, factor * dist)
            supports = [struct.get("recent_low"), struct.get("session_low"), float(liq_tp) if liq_tp is not None else None]
            supports = [float(v) for v in supports if v is not None and float(v) < entry - 0.80 * risk]
            if supports:
                compressed = max(compressed, max(supports))
            compressed = max(compressed, tp2)
            compressed = min(compressed, entry - 0.90 * risk)
        return float(compressed), float(factor), True, False, "soft_compress"
    except Exception:
        return float(tp2), 1.0, False, False, "none"


def _compute_profile_partial_tp(entry: float, sl: float, tp: float, direction: str, profile: Dict[str, Any], phase_label: str, strength: float, fast_df: Optional[pd.DataFrame] = None, fast_probe: Optional[Dict[str, Any]] = None, liq_tp: Optional[float] = None) -> Optional[float]:
    try:
        if str((profile or {}).get("name") or "") == "DAYTRADE":
            tp1 = _compute_daytrade_tp1(entry, sl, tp, direction, fast_df, fast_probe or {}, liq_tp, phase_label, strength)
            if tp1 is not None:
                return float(tp1)
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return None
        trigger_r = float((profile or {}).get("partial_r", 0.70) or 0.70)
        if str((profile or {}).get("name") or "") == "DAYTRADE":
            if str(phase_label).startswith("BREAKOUT") or str(phase_label) in ("UP_TREND", "DOWN_TREND"):
                trigger_r = min(0.75, trigger_r + 0.05 * float(_clamp(strength, 0.0, 1.0)))
            elif str(phase_label) == "RANGE":
                trigger_r = max(0.45, trigger_r - 0.10)
        trigger_r = float(_clamp(trigger_r, 0.40, 1.20))
        if str(direction).upper() == "LONG":
            tp1 = entry + trigger_r * risk
            if tp1 >= tp:
                tp1 = entry + 0.50 * (tp - entry)
        else:
            tp1 = entry - trigger_r * risk
            if tp1 <= tp:
                tp1 = entry - 0.50 * (entry - tp)
        return float(tp1)
    except Exception:
        return _compute_partial_tp(entry, sl, tp, direction)


def _ctx_dataframe(ctx_in: Dict[str, Any], *keys: str) -> Optional[pd.DataFrame]:
    for k in keys:
        try:
            df = (ctx_in or {}).get(k)
        except Exception:
            df = None
        if isinstance(df, pd.DataFrame) and (not df.empty):
            cols = set(df.columns)
            if {"Open", "High", "Low", "Close"}.issubset(cols):
                return df.copy()
    return None


def _probe_intraday_frame(df: Optional[pd.DataFrame], direction: str, lookback: int = 16) -> Dict[str, Any]:
    try:
        if df is None or (not isinstance(df, pd.DataFrame)) or df.empty or len(df) < max(24, lookback + 6):
            return {}
        d = df.copy().tail(max(lookback + 30, 60))
        o = d["Open"].astype(float)
        h = d["High"].astype(float)
        l = d["Low"].astype(float)
        c = d["Close"].astype(float)
        atr = float(_atr(d, 14).iloc[-1])
        atr = max(atr, 1e-9)
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        last_close = float(c.iloc[-1])
        recent_low = float(l.tail(lookback).min())
        recent_high = float(h.tail(lookback).max())
        breakout_ok, breakout_strength = _breakout_strength(d, max(10, min(lookback, 18)))
        hhhl_ok = _hh_hl_ok(d, max(12, min(lookback + 6, 24)))
        lllh_ok = _ll_lh_ok(d, max(12, min(lookback + 6, 24)))
        mom_r = float((c.iloc[-1] - c.iloc[-4]) / atr) if len(c) >= 4 else 0.0
        candle_r = float((c.iloc[-1] - o.iloc[-1]) / atr)
        long_ok = bool((ema8.iloc[-1] >= ema21.iloc[-1]) and (mom_r >= -0.20) and (last_close >= float(ema8.iloc[-1]) or breakout_ok or hhhl_ok))
        short_ok = bool((ema8.iloc[-1] <= ema21.iloc[-1]) and (mom_r <= 0.20) and (last_close <= float(ema8.iloc[-1]) or breakout_ok or lllh_ok))
        dir_ok = long_ok if str(direction).upper() == "LONG" else short_ok
        score = 0.0
        score += 0.55 if dir_ok else -0.70
        score += 0.18 if breakout_ok else 0.0
        if str(direction).upper() == "LONG":
            score += 0.12 if hhhl_ok else -0.05
            score += 0.10 if candle_r > 0 else -0.08
            score += 0.10 if mom_r > 0 else -0.08
        else:
            score += 0.12 if lllh_ok else -0.05
            score += 0.10 if candle_r < 0 else -0.08
            score += 0.10 if mom_r < 0 else -0.08
        score = float(_clamp(score, -1.0, 1.0))
        return {
            "last_close": float(last_close),
            "atr14": float(atr),
            "recent_low": float(recent_low),
            "recent_high": float(recent_high),
            "breakout_ok": bool(breakout_ok),
            "breakout_strength": float(breakout_strength),
            "hhhl_ok": bool(hhhl_ok),
            "lllh_ok": bool(lllh_ok),
            "mom_r": float(mom_r),
            "candle_r": float(candle_r),
            "dir_ok": bool(dir_ok),
            "score": float(score),
        }
    except Exception:
        return {}


def _daytrade_refine_levels(entry: float, sl: float, tp: float, direction: str, trade_profile: Dict[str, Any], atr14: float,
                            recent_low: float, recent_high: float, liq_lookback: int,
                            fast_df: Optional[pd.DataFrame], fast_probe: Dict[str, Any],
                            micro_df: Optional[pd.DataFrame], micro_probe: Dict[str, Any]) -> Tuple[float, float, float, float, float, Optional[float]]:
    try:
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
        risk0 = abs(entry - sl)
        fast_atr = max(float((fast_probe or {}).get("atr14") or atr14), 1e-9)
        ref_entry = float((micro_probe or {}).get("last_close") or (fast_probe or {}).get("last_close") or entry)
        fast_low = float((fast_probe or {}).get("recent_low") or recent_low)
        fast_high = float((fast_probe or {}).get("recent_high") or recent_high)
        if str(direction).upper() == "LONG":
            sl_fast = min(ref_entry - max(0.85 * fast_atr, 0.55 * atr14), fast_low - 0.08 * fast_atr)
            fast_tp_base = ref_entry + max(float(trade_profile.get("tp_trend", 1.55) or 1.55) * fast_atr, 1.25 * abs(ref_entry - sl_fast))
            liq_tp = None
            if isinstance(fast_df, pd.DataFrame) and (not fast_df.empty):
                liq_tp = _liquidity_pool_tp(fast_df.tail(max(12, min(len(fast_df), liq_lookback + 6))), "LONG", lookback=max(12, min(liq_lookback, max(12, len(fast_df) - 2))))
            tp_fast = max(fast_tp_base, liq_tp) if liq_tp is not None else fast_tp_base
            sl_new = min(sl, sl_fast)
            tp_new = max(tp_fast, ref_entry + 1.05 * abs(ref_entry - sl_new))
        else:
            sl_fast = max(ref_entry + max(0.85 * fast_atr, 0.55 * atr14), fast_high + 0.08 * fast_atr)
            fast_tp_base = ref_entry - max(float(trade_profile.get("tp_trend", 1.55) or 1.55) * fast_atr, 1.25 * abs(ref_entry - sl_fast))
            liq_tp = None
            if isinstance(fast_df, pd.DataFrame) and (not fast_df.empty):
                liq_tp = _liquidity_pool_tp(fast_df.tail(max(12, min(len(fast_df), liq_lookback + 6))), "SHORT", lookback=max(12, min(liq_lookback, max(12, len(fast_df) - 2))))
            tp_fast = min(fast_tp_base, liq_tp) if liq_tp is not None else fast_tp_base
            sl_new = max(sl, sl_fast)
            tp_new = min(tp_fast, ref_entry - 1.05 * abs(ref_entry - sl_new))
        entry_new = float(ref_entry)
        risk = abs(entry_new - sl_new)
        reward = abs(tp_new - entry_new)
        if risk <= 1e-9:
            risk = max(risk0, fast_atr)
        rr = reward / max(risk, 1e-9)
        return float(entry_new), float(sl_new), float(tp_new), float(risk), float(rr), (float(liq_tp) if liq_tp is not None else None)
    except Exception:
        risk = abs(float(entry) - float(sl))
        rr = abs(float(tp) - float(entry)) / max(risk, 1e-9)
        return float(entry), float(sl), float(tp), float(risk), float(rr), None


def _parse_utc_like(ts: Any) -> Optional[datetime]:
    try:
        if ts is None:
            return None
        s = str(ts).strip()
        if not s:
            return None
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _compute_partial_tp(entry: float, sl: float, tp: float, direction: str):
    try:
        entry = float(entry)
        sl = float(sl)
        tp = float(tp)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return None

        if str(direction).upper() == "LONG":
            tp1 = entry + 0.7 * risk
            if tp1 >= tp:
                tp1 = entry + 0.5 * (tp - entry)
        else:
            tp1 = entry - 0.7 * risk
            if tp1 <= tp:
                tp1 = entry - 0.5 * (entry - tp)

        return float(tp1)
    except Exception:
        return None

        if str(direction).upper() == "LONG":
            return float(entry) + risk
        return float(entry) - risk
    except Exception:
        return None


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
import ssl
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

_FF_CAL_URL_DEFAULT = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_EVENT_CACHE = {
    "ts": 0.0,     # epoch seconds
    "url": None,
    "data": None,  # list
    "err": None,   # str
}



# Persistent file cache (survives Streamlit reruns / transient rate limits)
_EVENT_FILE_CACHE_PATH = Path(tempfile.gettempdir()) / "fx_analyzer_ff_calendar_cache.json"

def _read_event_file_cache(max_age_sec: int = 24 * 3600) -> Optional[List[dict]]:
    try:
        p = _EVENT_FILE_CACHE_PATH
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(obj, dict):
            return None
        ts = float(obj.get("ts", 0.0) or 0.0)
        data = obj.get("data", None)
        if not isinstance(data, list):
            return None
        if ts > 0 and (time.time() - ts) > float(max_age_sec):
            return None
        return data
    except Exception:
        return None

def _write_event_file_cache(url: str, data: List[dict]) -> None:
    try:
        p = _EVENT_FILE_CACHE_PATH
        payload = {"ts": time.time(), "url": url, "data": data}
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _derive_xml_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    if ".json" in u:
        return u.replace(".json", ".xml")
    if u.endswith("/"):
        u = u[:-1]
    # fallback
    return "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

def _fetch_ff_calendar_xml(url: str, timeout: int = 12) -> Optional[List[dict]]:
    """Parse ForexFactory weekly XML export into list[dict] compatible with _compute_event_risk."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (fx-analyzer; event-guard)"})
        ctx = None
        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = None
        try:
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read()
        except TypeError:
            # python without context param
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()

        # XML is often windows-1252
        txt = raw.decode("utf-8", errors="replace")
        # Some servers send the XML declaration with windows-1252; ElementTree can parse bytes too.
        root = ET.fromstring(txt)
        out: List[dict] = []
        ny_tz = ZoneInfo("America/New_York")
        for ev in root.findall(".//event"):
            title = (ev.findtext("title") or "").strip()
            ctry = (ev.findtext("country") or "").strip().upper()
            impact = (ev.findtext("impact") or "").strip()
            date_s = (ev.findtext("date") or "").strip()
            time_s = (ev.findtext("time") or "").strip()
            # Date is typically MM-DD-YYYY in FF XML export
            dt_obj = None
            try:
                mm, dd, yy = date_s.split("-")
                hh, mi = (time_s.split(":") + ["0"])[:2] if time_s else ("0","0")
                dt_obj = datetime(int(yy), int(mm), int(dd), int(hh), int(mi), 0, tzinfo=ny_tz)
            except Exception:
                dt_obj = None
            if dt_obj is None:
                continue
            out.append({
                "title": title,
                "country": ctry,
                "impact": impact,
                "timestamp": float(dt_obj.astimezone(timezone.utc).timestamp()),
            })
        return out if out else None
    except Exception:
        return None
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


def _pip_size(pair: str) -> float:
    s = (pair or "").upper()
    # Heuristic: JPY pairs use 0.01, others 0.0001
    return 0.01 if "JPY" in s else 0.0001

def _round_to_pip(x: float, pair: str) -> float:
    try:
        p = _pip_size(pair)
        return float(round(float(x) / p) * p)
    except Exception:
        return float(x)

def _fetch_ff_calendar(url: str, timeout: int = 12, ttl_sec: int = 1800) -> Tuple[Optional[List[dict]], str]:
    """Fetch weekly economic calendar (JSON preferred) with robust caching + XML/file fallback.

    Streamlit Cloud reruns the script frequently. Also, the free FF weekly export can occasionally
    return 429/5xx or time out. We therefore:
      1) use in-memory TTL cache
      2) try JSON fetch
      3) on failure, try XML export (same host)
      4) on failure, fall back to a local file cache (<=24h)
    """
    now = time.time()

    # 1) in-memory cache
    if (_EVENT_CACHE.get("data") is not None) and (_EVENT_CACHE.get("url") == url) and (now - float(_EVENT_CACHE.get("ts", 0.0)) < float(ttl_sec)):
        return _EVENT_CACHE["data"], "cache"

    def _urlopen_bytes(req: Request) -> bytes:
        # Default SSL context; on TLS issues fall back to unverified context
        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = None
        try:
            try:
                with urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read()
            except TypeError:
                with urlopen(req, timeout=timeout) as resp:
                    return resp.read()
        except Exception:
            # last resort (some environments)
            try:
                uctx = ssl._create_unverified_context()
                with urlopen(req, timeout=timeout, context=uctx) as resp:
                    return resp.read()
            except Exception:
                raise

    # 2) JSON fetch
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (fx-analyzer; event-guard)", "Accept": "application/json,text/plain,*/*"})
        raw = _urlopen_bytes(req)
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(data, list):
            raise ValueError("calendar json is not a list")
        _EVENT_CACHE.update({"ts": now, "url": url, "data": data, "err": None})
        _write_event_file_cache(url, data)
        return data, "ok"
    except Exception as e_json:
        # 3) XML fallback
        try:
            xml_url = _derive_xml_url(url)
            data_xml = _fetch_ff_calendar_xml(xml_url, timeout=timeout)
            if isinstance(data_xml, list) and data_xml:
                _EVENT_CACHE.update({"ts": now, "url": url, "data": data_xml, "err": f"json_fail={type(e_json).__name__}"})
                _write_event_file_cache(url, data_xml)
                return data_xml, "ok"
        except Exception:
            pass

        # 4) file cache fallback
        cached = _read_event_file_cache(max_age_sec=24 * 3600)
        if isinstance(cached, list) and cached:
            _EVENT_CACHE.update({"ts": now, "url": url, "data": cached, "err": f"json_fail={type(e_json).__name__}: {e_json}"})
            return cached, "cache"

        _EVENT_CACHE.update({"ts": now, "url": url, "data": None, "err": f"{type(e_json).__name__}: {e_json}"})
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
    past_lookback_hours: int = 24,
    hours_scale: float = 24.0,
    norm: float = 3.0,
    impacts: Optional[List[str]] = None,
    high_window_minutes: int = 60,
    url: str = _FF_CAL_URL_DEFAULT,
) -> Dict[str, Any]:
    """
    Swing-oriented economic calendar risk model.

    Returns:
      {
        "ok": bool,
        "status": "ok|cache|fail",
        "err": str|None,
        "score": float,                  # upcoming-only risk score (heuristic)
        "factor": float (0..1),          # normalized risk factor (upcoming-only)
        "window_high": bool,             # high-impact within ±high_window_minutes
        "next_high_hours": float|None,   # hours until next High impact (>=0)
        "last_high_hours": float|None,   # hours since last High impact (>=0)
        "next_any_hours": float|None,    # hours until next (any impact)
        "last_any_hours": float|None,    # hours since last (any impact)
        "impact_ccys": { "USD": {"upcoming": n, "recent": n}, ... },
        "upcoming": [ {dt_utc, currency, impact, title, hours} ... ] (<=10),
        "recent":   [ {dt_utc, currency, impact, title, hours} ... ] (<=10),  # hours is negative (in the past)
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
            "last_high_hours": None,
            "next_any_hours": None,
            "last_any_hours": None,
            "impact_ccys": {},
            "upcoming": [],
            "recent": [],
        }

    tz = ZoneInfo(now_tz)
    now_local = datetime.now(tz=tz)
    now_utc = now_local.astimezone(timezone.utc)

    # weights by impact
    w = {"High": 1.0, "Medium": 0.6, "Low": 0.3, "Holiday": 0.8, "Non-Economic": 0.2}

    events = []
    for it in data:
        if not isinstance(it, dict):
            continue
        cur = str(it.get("currency") or it.get("ccy") or it.get("cur") or "").upper().strip()
        if not cur:
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
        if hrs < -float(max(1, int(past_lookback_hours))):
            continue
        if hrs > float(horizon_hours):
            continue

        title = str(it.get("title") or it.get("event") or it.get("name") or "").strip()
        events.append({
            "dt_utc": dt_utc,
            "hours": float(hrs),
            "currency": cur,
            "impact": impact,
            "title": title,
        })

    events.sort(key=lambda x: x["hours"])

    # next/last helpers
    next_high = None
    last_high = None
    next_any = None
    last_any = None
    window_high = False

    # score: upcoming-only
    score = 0.0

    impact_ccys: Dict[str, Dict[str, int]] = {}
    def _inc(cur: str, k: str) -> None:
        if not cur:
            return
        if cur not in impact_ccys:
            impact_ccys[cur] = {"upcoming": 0, "recent": 0}
        impact_ccys[cur][k] = int(impact_ccys[cur].get(k, 0)) + 1

    for ev in events:
        impact = ev["impact"]
        hrs = float(ev["hours"])

        # window check for High impact (±minutes)
        if impact == "High":
            if abs(hrs) * 60.0 <= float(high_window_minutes):
                window_high = True

        if hrs >= 0:
            # upcoming
            if next_any is None:
                next_any = hrs
            if impact == "High" and next_high is None:
                next_high = hrs
            _inc(str(ev.get("currency") or ""), "upcoming")

            denom = 1.0 + (max(0.0, hrs) / max(1e-6, float(hours_scale)))
            score += float(w.get(impact, 0.4)) / denom
        else:
            # recent
            if last_any is None:
                last_any = abs(hrs)
            if impact == "High" and last_high is None:
                last_high = abs(hrs)
            _inc(str(ev.get("currency") or ""), "recent")

    # normalize to factor 0..1 (heuristic)
    factor = _clamp(score / max(1e-6, float(norm)), 0.0, 1.0)

    # trim lists for UI
    upcoming_ui = []
    recent_ui = []
    for ev in events:
        rec = {
            "dt_utc": ev["dt_utc"].isoformat(),
            "hours": float(ev["hours"]),
            "currency": ev["currency"],
            "impact": ev["impact"],
            "title": ev["title"],
        }
        if float(ev["hours"]) >= 0 and len(upcoming_ui) < 10:
            upcoming_ui.append(rec)
        if float(ev["hours"]) < 0 and len(recent_ui) < 10:
            recent_ui.append(rec)

    return {
        "ok": True,
        "status": status,
        "err": None,
        "score": float(score),
        "factor": float(factor),
        "window_high": bool(window_high),
        "next_high_hours": (float(next_high) if next_high is not None else None),
        "last_high_hours": (float(last_high) if last_high is not None else None),
        "next_any_hours": (float(next_any) if next_any is not None else None),
        "last_any_hours": (float(last_any) if last_any is not None else None),
        "impact_ccys": impact_ccys,
        "upcoming": upcoming_ui,
        "recent": recent_ui,
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



def _compute_unrealized_R(side: str, entry: float, sl: float, price: float) -> float:
    """Unrealized PnL in R units (R = initial stop distance). Positive is profit."""
    try:
        side = str(side or "").upper()
        entry = float(entry)
        sl = float(sl)
        price = float(price)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return 0.0
        if side == "SELL":
            return (entry - price) / risk
        return (price - entry) / risk
    except Exception:
        return 0.0


def _extract_hold_state(
    pair: str,
    df: pd.DataFrame,
    ctx_in: Dict[str, Any],
    plan_like: Dict[str, Any],
    ev_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        pos = ctx_in.get("position") or ctx_in.get("pos") or {}
        if not isinstance(pos, dict):
            pos = {}
        pos_open = bool(ctx_in.get("position_open", False) or pos.get("open") or pos.get("is_open") or (len(pos) > 0))
        if not pos_open:
            return None
        side = str(pos.get("side") or pos.get("pos_side") or plan_like.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            side = str(plan_like.get("side") or "BUY").upper()
        entry = float(pos.get("entry") or pos.get("entry_price") or pos.get("pos_entry") or plan_like.get("entry") or plan_like.get("entry_price") or 0.0)
        sl = float(pos.get("sl") or pos.get("stop_loss") or pos.get("pos_sl") or plan_like.get("sl") or plan_like.get("stop_loss") or 0.0)
        tp = float(pos.get("tp") or pos.get("take_profit") or pos.get("pos_tp") or plan_like.get("tp") or plan_like.get("take_profit") or 0.0)
        price = None
        for k in ("current_price", "price", "last_price", "mark_price"):
            if k in pos and pos.get(k) is not None:
                price = pos.get(k)
                break
        if price is None:
            price = ctx_in.get("pos_current_price", None)
        if price is None:
            try:
                price = float(df["Close"].astype(float).iloc[-1]) if isinstance(df, pd.DataFrame) and (not df.empty) else float(entry)
            except Exception:
                price = float(entry)
        price = float(price)
        unrealized_R = pos.get("unrealized_R", None)
        if unrealized_R is None:
            unrealized_R = _compute_unrealized_R(side, entry, sl, price)
        else:
            unrealized_R = float(unrealized_R)
        dd_R = float(pos.get("dd_R") or pos.get("max_dd_R") or pos.get("drawdown_R") or 0.0)
        peak_R = pos.get("peak_R") or pos.get("peak_unrealized_R") or pos.get("max_favorable_R")
        peak_R = float(peak_R) if peak_R is not None else float(unrealized_R)
        nh = ev_meta.get("next_high_hours", None)
        try:
            nh_f = (float(nh) if nh is not None else None)
        except Exception:
            nh_f = None
        time_in_trade_h = None
        try:
            opened_at = _parse_utc_like(pos.get("opened_at_utc") or pos.get("opened_at") or ctx_in.get("opened_at_utc"))
            if opened_at is not None:
                time_in_trade_h = max(0.0, (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600.0)
        except Exception:
            time_in_trade_h = None
        trade_profile = _resolve_trade_profile(ctx_in or {})
        return {
            "pair": str(pair),
            "pos": pos,
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "price": float(price),
            "unrealized_R": float(unrealized_R),
            "dd_R": float(dd_R),
            "peak_R": float(peak_R),
            "time_in_trade_h": (float(time_in_trade_h) if time_in_trade_h is not None else None),
            "trade_profile": trade_profile,
            "nh_f": nh_f,
            "event_factor": float(ev_meta.get("factor", 0.0) or 0.0),
            "window_high": bool(ev_meta.get("window_high", False)),
        }
    except Exception:
        return None


def _swing_hold_v1(state: Dict[str, Any], ctx_in: Dict[str, Any], weekend_risk: float, weekcross_risk: float) -> Dict[str, Any]:
    try:
        side = str(state.get("side") or "BUY").upper()
        entry = float(state.get("entry") or 0.0)
        sl = float(state.get("sl") or 0.0)
        tp = float(state.get("tp") or 0.0)
        price = float(state.get("price") or entry)
        unrealized_R = float(state.get("unrealized_R") or 0.0)
        dd_R = float(state.get("dd_R") or 0.0)
        nh_f = state.get("nh_f")
        event_factor = float(state.get("event_factor") or 0.0)
        window_high = bool(state.get("window_high", False))
        time_in_trade_h = state.get("time_in_trade_h")
        trade_profile = dict(state.get("trade_profile") or {})
        no_add_h = float(ctx_in.get("hold_no_add_hours", 48.0) or 48.0)
        reduce_h = float(ctx_in.get("hold_reduce_hours", 18.0) or 18.0)
        be_h = float(ctx_in.get("hold_breakeven_hours", 12.0) or 12.0)
        partial_h = float(ctx_in.get("hold_partial_tp_hours", 18.0) or 18.0)
        no_add_h = max(6.0, min(168.0, no_add_h))
        reduce_h = max(3.0, min(72.0, reduce_h))
        be_h = max(1.0, min(48.0, be_h))
        partial_h = max(1.0, min(72.0, partial_h))
        notes: List[str] = []
        actions: List[str] = []
        no_add = False
        if (nh_f is not None) and (nh_f <= no_add_h):
            no_add = True
            notes.append(f"高インパクト指標が{nh_f:.1f}時間以内 → 追加建て禁止（スイングでも実行リスク回避）")
        if float(weekend_risk or 0.0) > 0.0:
            no_add = True
            notes.append("週末ギャップリスク → 追加建て禁止")
        if float(weekcross_risk or 0.0) > 0.0:
            no_add = True
            notes.append("週跨ぎ（木金）リスク → 追加建て禁止")
        reduce_mult = 1.0
        if (nh_f is not None) and (nh_f <= reduce_h):
            reduce_mult = min(reduce_mult, float(_clamp(1.0 - 0.60 * event_factor, 0.20, 1.00)))
            if unrealized_R >= 0.20:
                reduce_mult = min(reduce_mult, 0.50)
                actions.append("REDUCE_SIZE")
                notes.append("イベント接近＆含み益あり → 建玉の一部縮退（例：半分）を推奨")
            else:
                reduce_mult = min(reduce_mult, 0.70)
                actions.append("REDUCE_SIZE")
                notes.append("イベント接近 → 建玉縮退を推奨（リスク低減）")
        if float(weekend_risk or 0.0) > 0.0:
            reduce_mult = min(reduce_mult, 0.60)
            if "REDUCE_SIZE" not in actions:
                actions.append("REDUCE_SIZE")
            notes.append("週末前 → 建玉縮退を推奨（ギャップ対策）")
        partial_tp = 0.0
        if (nh_f is not None) and (nh_f <= partial_h) and (unrealized_R >= 0.60):
            partial_tp = max(partial_tp, 0.50)
            actions.append("PARTIAL_TP")
            notes.append("含み益0.6R以上＆イベント近接 → 半分利確を推奨")
        if window_high and (unrealized_R >= 0.30):
            partial_tp = max(partial_tp, 0.50)
            if "PARTIAL_TP" not in actions:
                actions.append("PARTIAL_TP")
            notes.append("高インパクト窓内 → 半分利確を推奨（スリッページ/乱高下対策）")
        move_be = False
        new_sl = None
        if (nh_f is not None) and (nh_f <= be_h) and (unrealized_R >= 0.15):
            move_be = True
            new_sl = float(entry)
            actions.append("MOVE_SL_TO_BE")
            notes.append("イベント接近 → ストップを建値（BE）へ移動を推奨（勝ちを負けにしない）")
        if float(weekend_risk or 0.0) > 0.0 and (unrealized_R >= 0.15):
            move_be = True
            if new_sl is None:
                new_sl = float(entry)
            if "MOVE_SL_TO_BE" not in actions:
                actions.append("MOVE_SL_TO_BE")
            notes.append("週末前 → 建値/浅い利確で防御を推奨")
        try:
            if new_sl is not None:
                new_sl = _round_to_pip(float(new_sl), state.get("pair", ""))
        except Exception:
            pass
        return {
            "version": "swing_hold_v1",
            "trade_profile": str(trade_profile.get("name", "SWING")),
            "pair": str(state.get("pair", "")),
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "current_price": float(price),
            "unrealized_R": float(unrealized_R),
            "dd_R": float(dd_R),
            "event_next_high_hours": nh_f,
            "event_window_high": bool(window_high),
            "event_risk_factor": float(event_factor),
            "weekend_risk": float(weekend_risk or 0.0),
            "weekcross_risk": float(weekcross_risk or 0.0),
            "time_in_trade_hours": (float(time_in_trade_h) if time_in_trade_h is not None else None),
            "no_add": bool(no_add),
            "reduce_size_mult": float(_clamp(reduce_mult, 0.20, 1.00)),
            "partial_tp_ratio": float(_clamp(partial_tp, 0.0, 1.0)),
            "move_sl_to_be": bool(move_be),
            "new_sl_reco": (float(new_sl) if new_sl is not None else None),
            "actions": list(dict.fromkeys(actions)),
            "notes": notes,
        }
    except Exception:
        return {}


def daytrade_hold_v1(state: Dict[str, Any], ctx_in: Dict[str, Any], weekend_risk: float, weekcross_risk: float) -> Dict[str, Any]:
    try:
        side = str(state.get("side") or "BUY").upper()
        entry = float(state.get("entry") or 0.0)
        sl = float(state.get("sl") or 0.0)
        tp = float(state.get("tp") or 0.0)
        price = float(state.get("price") or entry)
        unrealized_R = float(state.get("unrealized_R") or 0.0)
        dd_R = float(state.get("dd_R") or 0.0)
        peak_R = float(state.get("peak_R") or unrealized_R)
        nh_f = state.get("nh_f")
        event_factor = float(state.get("event_factor") or 0.0)
        window_high = bool(state.get("window_high", False))
        time_in_trade_h = state.get("time_in_trade_h")
        trade_profile = dict(state.get("trade_profile") or {})
        time_in_trade_m = (float(time_in_trade_h) * 60.0) if time_in_trade_h is not None else None
        trail_trigger_r = float(trade_profile.get("trail_trigger_r", 0.90) or 0.90)
        be_trigger_r = float(trade_profile.get("be_trigger_r", 0.40) or 0.40)
        fast_df = _ctx_dataframe(ctx_in, "_df_fast_15m", "df_fast_15m", "_df_15m")
        micro_df = _ctx_dataframe(ctx_in, "_df_micro_5m", "df_micro_5m", "_df_5m")
        direction = "LONG" if side == "BUY" else "SHORT"
        fast_probe = _probe_intraday_frame(fast_df, direction, lookback=16)
        micro_probe = _probe_intraday_frame(micro_df, direction, lookback=12)
        fast_bad = bool(fast_probe) and (fast_probe.get("dir_ok") is False)
        micro_bad = bool(micro_probe) and (micro_probe.get("dir_ok") is False)
        alignment = 0.0
        if fast_probe:
            alignment += 0.65 * float(fast_probe.get("score", 0.0) or 0.0)
        if micro_probe:
            alignment += 0.35 * float(micro_probe.get("score", 0.0) or 0.0)
        fade_R = max(0.0, float(peak_R) - float(unrealized_R))
        notes: List[str] = []
        actions: List[str] = []
        no_add = bool(float(weekend_risk or 0.0) > 0.0 or float(weekcross_risk or 0.0) > 0.0)
        reduce_mult = 1.0
        partial_tp = 0.0
        move_be = False
        new_sl = None
        time_exit_stage = None
        if unrealized_R >= be_trigger_r:
            move_be = True
            new_sl = float(entry)
            actions.append("MOVE_SL_TO_BE")
            notes.append("デイトレで0.4R超 → 建値防御を優先")
        if unrealized_R >= trail_trigger_r:
            partial_tp = max(partial_tp, 0.35)
            if "PARTIAL_TP" not in actions:
                actions.append("PARTIAL_TP")
            notes.append("デイトレで0.9R超 → 一部利確で利益保護")
        if (nh_f is not None) and (nh_f <= 3.0):
            no_add = True
            reduce_mult = min(reduce_mult, float(_clamp(1.0 - 0.55 * max(0.35, event_factor), 0.20, 1.0)))
            if "REDUCE_SIZE" not in actions:
                actions.append("REDUCE_SIZE")
            notes.append(f"高インパクト指標が{nh_f:.1f}時間以内 → デイトレ建玉を軽くする")
        if float(weekend_risk or 0.0) > 0.0:
            reduce_mult = min(reduce_mult, 0.60)
            if "REDUCE_SIZE" not in actions:
                actions.append("REDUCE_SIZE")
            notes.append("週末前 → デイトレでも建玉縮小")
        if float(weekcross_risk or 0.0) > 0.0:
            reduce_mult = min(reduce_mult, 0.75)
            if "REDUCE_SIZE" not in actions:
                actions.append("REDUCE_SIZE")
            notes.append("週跨ぎ前 → 建玉縮小を優先")
        def _stage_exit(stage_label: str, detail: str, reduce_to: float = 0.35):
            nonlocal time_exit_stage, reduce_mult
            time_exit_stage = stage_label
            reduce_mult = min(reduce_mult, float(reduce_to))
            if "TIME_EXIT" not in actions:
                actions.append("TIME_EXIT")
            notes.append(detail)
        if time_in_trade_m is not None:
            if time_in_trade_m >= 30.0:
                if (unrealized_R <= 0.02 and micro_bad) or (fade_R >= 0.18 and micro_bad) or (alignment <= -0.25 and unrealized_R < 0.10):
                    _stage_exit("30m", "30分経過でも伸びず、5分足が逆向き → 失速撤退を優先", reduce_to=0.50)
            if time_in_trade_m >= 60.0:
                if (unrealized_R < 0.10 and (micro_bad or fast_bad)) or (fade_R >= 0.22 and alignment <= -0.10):
                    _stage_exit("60m", "60分経過で伸び不足/戻し発生 → デイトレ撤退を優先", reduce_to=0.35)
            if time_in_trade_m >= 120.0:
                if unrealized_R < 0.20 or alignment <= 0.0:
                    _stage_exit("120m", "120分経過でも利が伸びないため、時間撤退を優先", reduce_to=0.20)
        if window_high and unrealized_R >= 0.25:
            partial_tp = max(partial_tp, 0.50)
            if "PARTIAL_TP" not in actions:
                actions.append("PARTIAL_TP")
            notes.append("高インパクト窓内 → デイトレは半分利確を優先")
        try:
            if new_sl is not None:
                new_sl = _round_to_pip(float(new_sl), state.get("pair", ""))
        except Exception:
            pass
        return {
            "version": "daytrade_hold_v1",
            "trade_profile": str(trade_profile.get("name", "DAYTRADE")),
            "pair": str(state.get("pair", "")),
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "current_price": float(price),
            "unrealized_R": float(unrealized_R),
            "peak_R": float(peak_R),
            "fade_R": float(fade_R),
            "dd_R": float(dd_R),
            "event_next_high_hours": nh_f,
            "event_window_high": bool(window_high),
            "event_risk_factor": float(event_factor),
            "weekend_risk": float(weekend_risk or 0.0),
            "weekcross_risk": float(weekcross_risk or 0.0),
            "time_in_trade_hours": (float(time_in_trade_h) if time_in_trade_h is not None else None),
            "time_in_trade_minutes": (float(time_in_trade_m) if time_in_trade_m is not None else None),
            "time_exit_stage": time_exit_stage,
            "mtf_alignment_score": float(alignment),
            "fast_tf_dir_ok": (bool(fast_probe.get("dir_ok")) if fast_probe else None),
            "micro_tf_dir_ok": (bool(micro_probe.get("dir_ok")) if micro_probe else None),
            "no_add": bool(no_add),
            "reduce_size_mult": float(_clamp(reduce_mult, 0.20, 1.00)),
            "partial_tp_ratio": float(_clamp(partial_tp, 0.0, 1.0)),
            "move_sl_to_be": bool(move_be),
            "new_sl_reco": (float(new_sl) if new_sl is not None else None),
            "actions": list(dict.fromkeys(actions)),
            "notes": notes,
        }
    except Exception:
        return {}


def daytrade_hold_selfcheck() -> Dict[str, Any]:
    try:
        now = datetime.now(timezone.utc)
        idx = pd.date_range(end=now, periods=120, freq="5min", tz="UTC")
        base = np.linspace(100.0, 100.8, len(idx))
        d = pd.DataFrame({"Open": base - 0.03, "High": base + 0.06, "Low": base - 0.06, "Close": base}, index=idx)
        fast_df = d.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        scenarios = {
            "30m": {"opened_at_utc": (now - timedelta(minutes=31)).strftime("%Y-%m-%dT%H:%M:%SZ"), "current_price": 100.01, "peak_R": 0.24},
            "60m": {"opened_at_utc": (now - timedelta(minutes=61)).strftime("%Y-%m-%dT%H:%M:%SZ"), "current_price": 100.03, "peak_R": 0.34},
            "120m": {"opened_at_utc": (now - timedelta(minutes=121)).strftime("%Y-%m-%dT%H:%M:%SZ"), "current_price": 100.05, "peak_R": 0.36},
        }
        out = {"ok": True, "results": {}}
        for label, pos_extra in scenarios.items():
            ctx = {"price_interval": "1h", "trade_profile": "DAYTRADE", "_df_fast_15m": fast_df.copy(), "_df_micro_5m": d.copy()}
            ctx["_df_fast_15m"].iloc[-3:, ctx["_df_fast_15m"].columns.get_loc("Close")] = [100.12, 99.98, 99.90]
            ctx["_df_fast_15m"].iloc[-3:, ctx["_df_fast_15m"].columns.get_loc("Open")] = [100.16, 100.05, 100.00]
            ctx["_df_micro_5m"].iloc[-4:, ctx["_df_micro_5m"].columns.get_loc("Close")] = [100.14, 100.02, 99.96, 99.90]
            ctx["_df_micro_5m"].iloc[-4:, ctx["_df_micro_5m"].columns.get_loc("Open")] = [100.18, 100.08, 100.00, 99.96]
            state = {
                "pair": "AUD/JPY",
                "side": "BUY",
                "entry": 100.00,
                "sl": 99.50,
                "tp": 100.80,
                "price": float(pos_extra["current_price"]),
                "unrealized_R": float((float(pos_extra["current_price"]) - 100.00) / 0.50),
                "dd_R": 0.0,
                "peak_R": float(pos_extra["peak_R"]),
                "time_in_trade_h": (now - _parse_utc_like(pos_extra["opened_at_utc"])) .total_seconds() / 3600.0,
                "trade_profile": _resolve_trade_profile(ctx),
                "nh_f": None,
                "event_factor": 0.0,
                "window_high": False,
            }
            res = daytrade_hold_v1(state, ctx, 0.0, 0.0)
            out["results"][label] = {"actions": res.get("actions", []), "time_exit_stage": res.get("time_exit_stage"), "passed": bool("TIME_EXIT" in (res.get("actions", []) or []) and str(res.get("time_exit_stage")) == label)}
        out["passed_all"] = all(v.get("passed") for v in out["results"].values())
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _hold_manage_reco(
    pair: str,
    df: pd.DataFrame,
    ctx_in: Dict[str, Any],
    plan_like: Dict[str, Any],
    ev_meta: Dict[str, Any],
    weekend_risk: float,
    weekcross_risk: float,
) -> Dict[str, Any]:
    try:
        state = _extract_hold_state(pair, df, ctx_in, plan_like, ev_meta)
        if not isinstance(state, dict):
            return {}
        trade_profile = dict(state.get("trade_profile") or {})
        if str(trade_profile.get("name")) == "DAYTRADE":
            return daytrade_hold_v1(state, ctx_in, weekend_risk, weekcross_risk)
        return _swing_hold_v1(state, ctx_in, weekend_risk, weekcross_risk)
    except Exception:
        return {}
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


# --- PATCH: SHORT structure separation ---
def _ll_lh_ok(df, n: int = 20) -> bool:
    try:
        if len(df) < n + 5:
            return False
        close = df["Close"].astype(float)
        w = 3
        highs = (close.shift(w) < close) & (close.shift(-w) < close)
        lows = (close.shift(w) > close) & (close.shift(-w) > close)
        hi_idx = close[highs].tail(4).index
        lo_idx = close[lows].tail(4).index
        if len(hi_idx) < 2 or len(lo_idx) < 2:
            return False
        h1, h2 = close.loc[hi_idx[-2]], close.loc[hi_idx[-1]]
        l1, l2 = close.loc[lo_idx[-2]], close.loc[lo_idx[-1]]
        return (h2 < h1) and (l2 < l1)
    except Exception:
        return False
# --- END PATCH ---

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
    main.py 互換の“完成版”エントリー判定（ctx依存を排除し、内部で必要な特徴量を計算します）。

    完成要件（要点）:
      - 統合スコア（rank_score）で最終ランキングできる
      - 価格構造（HH/HL, ブレイク, トレンド強度, クローズ構造）を最優先
      - 通貨強弱は補助（単独で方向決定しない）
      - マクロは弱いバイアス（NO連発の主因にしない）
      - イベントは「直前の実行リスク」＋「直後の捕獲」を必須化
        * 直前: 成行禁止/縮退/閾値引上げ（ただし見送り地獄にしない）
        * 直後: 0-1h様子見、1-24hブレイク専用ゲートで積極再評価
      - veto乱立を抑え、主因が説明できる
      - NameError / SyntaxError ゼロ（phase_label等の未定義を根絶）
      - RLは出口専用（本関数は入口のみ）

    互換性:
      - main.py が期待する key を返します（expected_R_ev / p_win_ev / veto_reasons / state_probs / ev_contribs 等）
      - 旧key（entry/sl/tp, ev_raw/ev_adj, veto）も残します
      - 追加key: rank_score / final_score / p_eff / event_mode / event_last_high_hours 等
    """

    # --- safety init (NameError防止) ---
    phase = "UNKNOWN"
    phase_label = "UNKNOWN"

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

    if df is None and isinstance(ctx_in, dict):
        df = ctx_in.get("_df") or ctx_in.get("df") or ctx_in.get("price_df") or ctx_in.get("price_history")

    if df is not None and not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = None

    # OHLC列名の正規化
    if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
        cols = {c.lower(): c for c in df.columns}
        rename = {}
        for need in ["open", "high", "low", "close", "volume"]:
            if need in cols and cols[need] != need.capitalize():
                rename[cols[need]] = need.capitalize()
        if rename:
            df = df.rename(columns=rename)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

    need_cols = {"High", "Low", "Close"}
    if df is None or (not isinstance(df, pd.DataFrame)) or len(df) < 60 or (not need_cols.issubset(set(df.columns))):
        debug_cols = []
        try:
            debug_cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
        except Exception:
            debug_cols = []
        thr = float(_clamp(_safe_float((ctx_in or {}).get("min_expected_R", 0.10), 0.10), 0.03, 0.30))
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
            "rank_score": 0.0,
            "final_score": 0.0,
            "dynamic_threshold": thr,
            "gate_mode": "NO_DATA",
            "confidence": 0.0,
            "p_win": 0.0,
            "p_eff": 0.0,
            "p_win_ev": 0.0,
            "why": "データ不足",
            "veto": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "veto_reasons": ["データ不足（最低60本必要 / High・Low・Close必須）"],
            "event_mode": "NO_DATA",
            "event_next_high_hours": None,
            "event_last_high_hours": None,
            "state_probs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "ev_contribs": {"trend_up": 0.0, "trend_down": 0.0, "range": 0.0, "risk_off": 0.0},
            "_ctx": {"pair": pair, "len": int(len(df)) if isinstance(df, pd.DataFrame) else 0, "cols": debug_cols},
        }

    df = df.copy()
    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # -----------------------------------------------------------------
    # 2) 内部特徴量（ctx依存排除）
    # -----------------------------------------------------------------
    phase, strength, mom = _phase_label(df)
    trade_profile = _resolve_trade_profile(ctx_in or {})
    horizon = int(_safe_float(ctx_in.get("horizon_days", 5), 5))
    model_horizon = int(_safe_float(ctx_in.get("model_horizon_bars", max(3, horizon)), max(3, horizon)))
    if bool(trade_profile.get("is_intraday")):
        model_horizon = max(4, min(24, model_horizon))
    p_up, p_dn = _continuation_prob(df, horizon=max(3, model_horizon))

    lookback = int(max(10, _safe_float(trade_profile.get("lookback", 20), 20)))
    breakout_ok, breakout_strength = _breakout_strength(df, lookback)
    hhhl_ok = _hh_hl_ok(df, max(20, int(lookback * 1.5)))
    lllh_ok = _ll_lh_ok(df, max(20, int(lookback * 1.5)))
    structure_long = bool(hhhl_ok or breakout_ok)
    structure_short = bool(lllh_ok or breakout_ok)

    # 表示用ラベル（NameError根絶）
    if str(phase) == "RANGE":
        phase_label = "RANGE"
    elif str(phase) in ("UP_TREND", "BREAKOUT_UP", "TRANSITION_UP"):
        phase_label = "UP_TREND"
    elif str(phase) in ("DOWN_TREND", "BREAKOUT_DOWN", "TRANSITION_DOWN"):
        phase_label = "DOWN_TREND"
    else:
        phase_label = str(phase or "UNKNOWN")

    # -----------------------------------------------------------------
    # 3) 方向選択（通貨強弱“単独”は不可。価格構造主導）
    # -----------------------------------------------------------------
    direction = "LONG" if mom >= 0 else "SHORT"
    if prefer_long_only:
        direction = "LONG"
    if phase_label == "UP_TREND" and float(strength) >= 0.28:
        direction = "LONG"
    if phase_label == "DOWN_TREND" and float(strength) >= 0.28:
        direction = "SHORT"

    # RANGEは“端”なら逆張り優先（ただし厳格条件）
    recent_low = float(df["Low"].astype(float).tail(lookback).min())
    recent_high = float(df["High"].astype(float).tail(lookback).max())
    span = float(max(1e-9, recent_high - recent_low))
    range_pos = float(_clamp((last - recent_low) / span, 0.0, 1.0))
    if phase_label == "RANGE":
        if range_pos <= 0.30:
            direction = "LONG"
        elif range_pos >= 0.70:
            direction = "SHORT"

    side = "BUY" if direction == "LONG" else "SELL"
    structure_dir_ok = bool(structure_long if direction == "LONG" else structure_short)

    # -----------------------------------------------------------------
    # 4) リスクモデル（SL/TP）: ATRベース
    # -----------------------------------------------------------------
    atr14 = float(_atr(df, 14).iloc[-1])
    atr14 = max(atr14, 1e-6)
    sl_atr_mult = float(trade_profile.get("sl_atr_mult", 1.2) or 1.2)
    sl_buffer_atr = float(trade_profile.get("sl_buffer_atr", 0.15) or 0.15)
    regime_mult = _profile_tp_multiple(trade_profile, phase_label, breakout_ok, strength)
    liq_lookback = int(max(10, _safe_float(trade_profile.get("liquidity_lookback", lookback * 2), lookback * 2)))

    if direction == "LONG":
        sl = min(last - sl_atr_mult * atr14, recent_low - sl_buffer_atr * atr14)
        atr_tp = last + regime_mult * atr14
        liq_tp = _liquidity_pool_tp(df.tail(max(liq_lookback + 2, 12)), "LONG", lookback=liq_lookback)
        tp = max(atr_tp, liq_tp) if liq_tp is not None else atr_tp
    else:
        sl = max(last + sl_atr_mult * atr14, recent_high + sl_buffer_atr * atr14)
        atr_tp = last - regime_mult * atr14
        liq_tp = _liquidity_pool_tp(df.tail(max(liq_lookback + 2, 12)), "SHORT", lookback=liq_lookback)
        tp = min(atr_tp, liq_tp) if liq_tp is not None else atr_tp

    entry = last
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 1e-9:
        risk = atr14
    rr = reward / risk
    liq_tp_intraday = None
    mtf_fast = {}
    mtf_micro = {}
    mtf_alignment_score = 0.0
    if bool(trade_profile.get("is_intraday")):
        fast_df = _ctx_dataframe(ctx_in, "_df_fast_15m", "df_fast_15m", "_df_15m")
        micro_df = _ctx_dataframe(ctx_in, "_df_micro_5m", "df_micro_5m", "_df_5m")
        mtf_fast = _probe_intraday_frame(fast_df, direction, lookback=max(12, lookback))
        mtf_micro = _probe_intraday_frame(micro_df, direction, lookback=12)
        if mtf_fast:
            mtf_alignment_score += 0.65 * float(mtf_fast.get("score", 0.0) or 0.0)
        if mtf_micro:
            mtf_alignment_score += 0.35 * float(mtf_micro.get("score", 0.0) or 0.0)
        entry, sl, tp, risk, rr, liq_tp_intraday = _daytrade_refine_levels(
            entry, sl, tp, direction, trade_profile, atr14, recent_low, recent_high, liq_lookback,
            fast_df if isinstance(fast_df, pd.DataFrame) else None, mtf_fast,
            micro_df if isinstance(micro_df, pd.DataFrame) else None, mtf_micro,
        )
        reward = abs(tp - entry)
    tp2 = tp
    tp2_compress_factor = 1.0
    tp2_compressed = False
    tp2_hard_capped = False
    tp2_cap_reason = "none"
    if bool(trade_profile.get("is_intraday")):
        tp2, tp2_compress_factor, tp2_compressed, tp2_hard_capped, tp2_cap_reason = _compress_daytrade_tp2(
            entry, sl, tp2, direction, float(mtf_alignment_score or 0.0),
            (bool(mtf_fast.get("dir_ok")) if mtf_fast else None),
            (bool(mtf_micro.get("dir_ok")) if mtf_micro else None),
            fast_df if isinstance(fast_df, pd.DataFrame) else None,
            liq_tp_intraday,
        )
        tp = float(tp2)
        reward = abs(tp - entry)
        rr = reward / max(risk, 1e-9)
    rr_min = float(trade_profile.get("rr_floor", ctx_in.get("rr_min_floor", 1.0)) or 1.0)
    if bool(trade_profile.get("is_intraday")) and mtf_alignment_score >= 0.35:
        rr_min = max(0.95, rr_min - 0.10)
    rr_floor_fail = bool(rr < rr_min)
    partial_tp = _compute_profile_partial_tp(
        entry, sl, tp2, direction, trade_profile, phase_label, strength,
        fast_df=(fast_df if 'fast_df' in locals() and isinstance(fast_df, pd.DataFrame) else None),
        fast_probe=(mtf_fast if isinstance(mtf_fast, dict) else None),
        liq_tp=liq_tp_intraday,
    )
    tp1 = partial_tp if partial_tp is not None else tp2
    tp1_liquidity_capped = False
    if bool(trade_profile.get("is_intraday")):
        tp1, tp1_liquidity_capped = _cap_daytrade_tp1_to_liquidity(entry, tp1, tp2, direction, liq_tp_intraday)

    # -----------------------------------------------------------------
    # 5) 勝率 proxy（モデル）→ confidenceで縮退（p_eff）
    # -----------------------------------------------------------------
    cont_best = max(float(p_up), float(p_dn))
    if direction == "LONG":
        p_win_model = 0.46 + 0.42 * _clamp((float(p_up) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    else:
        p_win_model = 0.46 + 0.42 * _clamp((float(p_dn) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    p_win_model = float(_clamp(p_win_model - _failure_features(df), 0.20, 0.80))
    if bool(trade_profile.get("is_intraday")):
        p_win_model = float(_clamp(p_win_model + 0.10 * float(mtf_alignment_score or 0.0), 0.20, 0.85))

    # 信頼度（0..1）
    structure_ok_dir = (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))
    structure_flag = 1.0 if structure_ok_dir else 0.0
    confidence = float(_clamp(
        0.30
        + 0.40 * float(strength)
        + 0.18 * (float(cont_best) - 0.5)
        + 0.0
        + 0.04 * float(structure_flag),
        0.0, 1.0
    ))
    # Direction/structure alignment penalty
    if not structure_dir_ok:
        confidence = float(_clamp(confidence * 0.70, 0.0, 1.0))
    if bool(trade_profile.get("is_intraday")):
        confidence = float(_clamp(confidence + 0.14 * float(mtf_alignment_score or 0.0), 0.0, 1.0))

    # p_eff: confidenceが低いほど0.5に寄せる（整合崩れ対策）
    conf_k = float(_clamp(confidence / 0.75, 0.0, 1.0))
    p_eff = float(_clamp(0.5 + (p_win_model - 0.5) * conf_k, 0.20, 0.80))
    if bool(trade_profile.get("is_intraday")):
        p_eff = float(_clamp(p_eff + 0.08 * float(mtf_alignment_score or 0.0), 0.20, 0.82))

    # EV (R): EV = p*RR - (1-p)*1
    ev_raw = float(p_eff * float(rr) - (1.0 - p_eff) * 1.0)

    # -----------------------------------------------------------------
    # 6) 外部リスク（macro）: 弱いバイアスとして統合（NO連発の主因にしない）
    # -----------------------------------------------------------------
    gr = _safe_float(ext.get("global_risk_index", ext.get("global_risk", ext.get("risk_off", 0.35))), 0.35)
    war = _safe_float(ext.get("war_probability", ext.get("war", 0.0)), 0.0)
    macro_risk = _safe_float(ext.get("macro_risk_score", None), float("nan"))
    if not (isinstance(macro_risk, (int, float)) and math.isfinite(float(macro_risk))):
        macro_risk = _clamp(0.70 * gr + 0.30 * war, 0.0, 1.0)
    else:
        macro_risk = _clamp(float(macro_risk), 0.0, 1.0)

    # 表示用（ev_adj）は弱いペナルティに留める
    risk_penalty = 0.10 + 0.50 * float(macro_risk)   # 0.10..0.60
    ev_adj = float(ev_raw - 0.18 * float(risk_penalty))

    # -----------------------------------------------------------------
    # 6.5) 経済指標/イベント（直前の実行リスク + 直後の捕獲）
    # -----------------------------------------------------------------
    event_guard_enable = bool(ctx_in.get('event_guard_enable', True))
    event_block_window = bool(ctx_in.get('event_block_high_impact_window', True))
    event_horizon_hours = int(_safe_float(ctx_in.get("event_horizon_hours", 168), 168))
    event_past_lookback_hours = int(_safe_float(ctx_in.get("event_past_lookback_hours", 24), 24))
    event_window_minutes = int(_safe_float(ctx_in.get("event_window_minutes", 60), 60))
    event_impacts = ctx_in.get("event_impacts", None)
    if not isinstance(event_impacts, list) or not event_impacts:
        event_impacts = ["High", "Medium"]
    event_calendar_url = str(ctx_in.get("event_calendar_url", _FF_CAL_URL_DEFAULT) or _FF_CAL_URL_DEFAULT)

    ev_meta = {"ok": False, "status": "off", "err": None, "score": 0.0, "factor": 0.0,
               "window_high": False, "next_high_hours": None, "last_high_hours": None,
               "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}
    if event_guard_enable:
        try:
            ev_meta = _compute_event_risk(
                pair,
                now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"),
                horizon_hours=event_horizon_hours,
                past_lookback_hours=event_past_lookback_hours,
                hours_scale=float(ctx_in.get("event_hours_scale", 24.0) or 24.0),
                norm=float(ctx_in.get("event_norm", 3.0) or 3.0),
                impacts=[str(x) for x in event_impacts],
                high_window_minutes=event_window_minutes,
                url=event_calendar_url,
            )
        except Exception as e:
            ev_meta = {"ok": False, "status": "unknown", "err": f"{type(e).__name__}: {e}", "score": 0.0, "factor": 0.0,
                       "window_high": False, "next_high_hours": None, "last_high_hours": None,
                       "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}

    weekend_risk = float(_compute_weekend_risk(now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo")))

    # Thu/Fri are special for swing entries (weekend gap approaches + event clusters).
    try:
        _tz = ZoneInfo(str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"))
        _now_local = datetime.now(tz=_tz)
        _wd = int(_now_local.weekday())  # Mon=0..Sun=6
        weekcross_risk = 1.0 if _wd in (3, 4) else 0.0  # Thu/Fri
        weekcross_weekday = _wd
    except Exception:
        weekcross_risk = 0.0
        weekcross_weekday = None

    # event mode classification (swing)
    try:
        next_high = ev_meta.get("next_high_hours", None)
        last_high = ev_meta.get("last_high_hours", None)
        pre_h = float(ctx_in.get("event_preblock_hours", trade_profile.get("event_preblock_hours", 24.0)) or trade_profile.get("event_preblock_hours", 24.0))
        if bool(trade_profile.get("is_intraday")):
            pre_h = float(_clamp(pre_h, 2.0, 24.0))
        else:
            pre_h = float(_clamp(pre_h, 6.0, 72.0))
    except Exception:
        next_high = None
        last_high = None
        pre_h = 24.0

    event_mode = "NORMAL"
    if bool(ev_meta.get("window_high", False)) and event_block_window:
        event_mode = "EVENT_WINDOW"
    elif (last_high is not None) and (float(last_high) <= 1.0):
        event_mode = "POST_WAIT"          # 0-1h: wait
    elif (last_high is not None) and (float(last_high) <= 24.0):
        event_mode = "POST_BREAKOUT"      # 1-24h: breakout-only gate
    elif (next_high is not None) and (float(next_high) <= float(pre_h)):
        event_mode = "PRE_EVENT"          # upcoming high-impact is close
    else:
        event_mode = "NORMAL"

    # -----------------------------------------------------------------
    # 7) 動的閾値（フェーズ/構造優先 + リスク時に軽く上げる）
    # -----------------------------------------------------------------
    base_thr = _safe_float(ctx_in.get("dynamic_threshold_base", None), float("nan"))
    if not (isinstance(base_thr, (int, float)) and math.isfinite(float(base_thr))):
        base_thr = _safe_float(ctx_in.get("min_expected_R", 0.08), 0.08)
    base_thr = float(_clamp(float(base_thr), 0.03, 0.25))

    thr_mult = 1.0
    if phase_label in ("UP_TREND", "DOWN_TREND"):
        thr_mult -= 0.16 * float(strength)
    if str(phase).startswith("BREAKOUT"):
        thr_mult -= 0.22 * max(float(strength), float(breakout_strength))
    if phase_label == "RANGE":
        thr_mult += 0.10

    dynamic_threshold = float(_clamp(base_thr * thr_mult, 0.02, 0.30))

    # macro bias is weak
    dynamic_threshold = float(_clamp(dynamic_threshold + 0.03 * float(macro_risk), 0.02, 0.30))
    dynamic_threshold = float(_clamp(dynamic_threshold + float(trade_profile.get("threshold_bias", 0.0) or 0.0), 0.02, 0.30))
    if bool(trade_profile.get("is_intraday")):
        dynamic_threshold = float(_clamp(dynamic_threshold - 0.015 * max(0.0, float(mtf_alignment_score or 0.0)) + 0.020 * max(0.0, -float(mtf_alignment_score or 0.0)), 0.02, 0.30))

    # upcoming event / weekend / weekcross: threshold add (but do not cause perpetual NO)
    try:
        event_thr_add = float(ctx_in.get("event_threshold_add", 0.18) or 0.18)
        event_thr_add = float(_clamp(event_thr_add, 0.10, 0.30))
        weekend_thr_add = float(ctx_in.get("weekend_threshold_add", 0.03) or 0.03)
        weekend_thr_add = float(_clamp(weekend_thr_add, 0.0, 0.20))
        weekcross_thr_add = float(ctx_in.get("weekcross_threshold_add", 0.03) or 0.03)
        weekcross_thr_add = float(_clamp(weekcross_thr_add, 0.0, 0.20))

        ef = float(ev_meta.get("factor", 0.0) or 0.0)
        # POST_BREAKOUTでは“捕獲”を優先し、閾値上乗せを弱める
        if event_mode == "POST_BREAKOUT":
            ef *= 0.40
        dynamic_threshold = float(_clamp(
            dynamic_threshold
            + event_thr_add * ef
            + weekend_thr_add * float(weekend_risk or 0.0)
            + weekcross_thr_add * float(weekcross_risk or 0.0),
            0.02, 0.30
        ))
    except Exception:
        pass


    # B-rank: condition-specific threshold optimization
    try:
        if phase_label == "RANGE":
            dynamic_threshold = float(_clamp(dynamic_threshold + 0.04, 0.02, 0.30))
        elif phase_label in ("UP_TREND", "DOWN_TREND"):
            dynamic_threshold = float(_clamp(dynamic_threshold - 0.02, 0.02, 0.30))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 8) モメンタム/通貨強弱（補助のみ、上限あり）
    # -----------------------------------------------------------------
    mom_bonus = 0.0
    if direction == "LONG" and mom > 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_up), 0.0, 1.0)
    if direction == "SHORT" and mom < 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_dn), 0.0, 1.0)
    mom_bonus = float(_clamp(mom_bonus, 0.0, 0.06))

    ccy_strength_proxy = 0.0
    try:
        c = df["Close"].astype(float)
        r20 = (c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) >= 21 else 0.0
        r60 = (c.iloc[-1] / c.iloc[-61] - 1.0) if len(c) >= 61 else 0.0
        vol20 = float(c.pct_change().rolling(20).std().iloc[-1]) if len(c) >= 21 else 0.0
        vol60 = float(c.pct_change().rolling(60).std().iloc[-1]) if len(c) >= 61 else 0.0
        z20 = (r20 / (vol20 + 1e-9)) if vol20 > 0 else 0.0
        z60 = (r60 / (vol60 + 1e-9)) if vol60 > 0 else 0.0
        ccy_strength_proxy = float(_clamp(0.5 * z20 + 0.5 * z60, -1.0, 1.0))
    except Exception:
        ccy_strength_proxy = 0.0

    ccy_bonus = 0.0
    if direction == "LONG" and ccy_strength_proxy > 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    elif direction == "SHORT" and ccy_strength_proxy < 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    ccy_bonus = float(_clamp(ccy_bonus, 0.0, 0.04))

    ev_gate = float(ev_raw + mom_bonus + ccy_bonus)
    ev_gate = ev_gate * _quality_decay(strength, breakout_ok, hhhl_ok, confidence)
    # B-rank + prop AI adjustments
    ev_gate -= float(_range_center_penalty(range_pos))
    ev_gate -= float(_event_unknown_adjust(ev_meta))
    if _liquidity_sweep(df):
        ev_gate -= 0.25
    vol_exp = float(_volatility_expansion(df))
    ev_gate += 0.08 * vol_exp
    regime = _market_regime(df)
    if regime == "RANGE":
        ev_gate -= 0.15
    if not structure_dir_ok:
        ev_gate -= 0.10

    # -----------------------------------------------------------------
    # 9) 構造ゲート（最優先）
    # -----------------------------------------------------------------
    breakout_pass = bool(_entry_timing_filter(df, direction) and 
        (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))
        and (float(cont_best) >= 0.57)
        and (max(float(strength), float(breakout_strength)) >= 0.35)
        and (float(macro_risk) <= 0.90)
    )

    # RANGE 端の逆張り（厳格）
    range_edge_setup = False
    try:
        # イベント直前はレンジ逆張りを避ける（事故回避）。直後捕獲はブレイク専用。
        in_pre = (event_mode == "PRE_EVENT")
        if phase_label == "RANGE" and (not in_pre) and (event_mode not in ("EVENT_WINDOW", "POST_WAIT")):
            near_edge = (range_pos <= 0.25) if direction == "LONG" else (range_pos >= 0.75)
            range_edge_setup = bool(
                near_edge
                and (float(rr) >= 1.40)
                and (float(confidence) >= 0.45)
                and (float(cont_best) >= 0.54)
                and (float(macro_risk) <= 0.85)
                and (ev_gate >= float(dynamic_threshold) - 0.02)
            )
    except Exception:
        range_edge_setup = False

    # 全体の構造妥当性
    structure_ok = True
    if phase_label == "RANGE":
        center_avoid = bool(abs(float(range_pos) - 0.5) >= 0.18)
        structure_ok = bool((breakout_pass or range_edge_setup) and center_avoid)
    else:
        if (float(strength) < 0.18) and not (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok)):
            structure_ok = False

    # POST_BREAKOUTはブレイク根拠必須（取り逃がし防止と事故回避を両立）
    if event_mode == "POST_BREAKOUT":
        structure_ok = bool(breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))
    if bool(trade_profile.get("is_intraday")) and float(mtf_alignment_score or 0.0) <= -0.20:
        structure_ok = False

    # -----------------------------------------------------------------
    # 10) veto/decision（veto乱立を抑える）
    # -----------------------------------------------------------------
    veto: List[str] = []
    def _veto(msg: str) -> None:
        s = str(msg or "").strip()
        if not s:
            return
        if s not in veto:
            veto.append(s)

    why = ""
    gate_mode = "raw+mom"

    if rr_floor_fail:
        _veto(f"RR不足: {rr:.2f} < {rr_min:.2f}")
        decision = "NO_TRADE"

    # mandatory event window block
    if rr_floor_fail:
        pass
    elif event_guard_enable and event_block_window and event_mode == "EVENT_WINDOW":
        gate_mode = "event_block"
        why = f"高インパクト指標の前後（±{event_window_minutes}分）のため見送り"
        _veto(why)
        decision = "NO_TRADE"
    elif event_guard_enable and event_mode == "POST_WAIT":
        gate_mode = "post_wait"
        why = "高インパクト直後0〜1hは様子見（スプレッド/再反転の不確実性）"
        _veto(why)
        decision = "NO_TRADE"
    elif not structure_ok:
        gate_mode = "structure_veto"
        if phase_label == "RANGE":
            why = "レンジ優勢で構造根拠が不足（ブレイク or 端の逆張り条件が未達）"
        else:
            why = "価格構造の根拠が弱い（トレンド強度/HHHL/ブレイクが不足）"
        _veto(why)
        decision = "NO_TRADE"
    else:
        # EV gate (post-breakout has its own rescue)
        if ev_gate >= float(dynamic_threshold):
            decision = "TRADE"
            if bool(ctx_in.get("sbi_min_lot_guard", True)) and int(ctx_in.get("sbi_min_lot", 1) or 1) >= 1 and float(confidence) < 0.42:
                decision = "NO_TRADE"
                _veto("SBI最小1建リスク")
            if not _entry_timing_filter(df, direction):
                decision = "NO_TRADE"
                veto.append("Entry timing filter rejected")
            why = f"EV通過: {ev_gate:+.3f} ≥ 動的閾値 {float(dynamic_threshold):.3f}"
        elif event_mode == "POST_BREAKOUT" and (ev_gate >= float(dynamic_threshold) - 0.08) and float(confidence) >= 0.42:
            decision = "TRADE"
            gate_mode = "post_breakout_rescue"
            why = f"イベント後捕獲（1〜24hブレイク専用）: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        elif breakout_pass and (ev_gate >= float(dynamic_threshold) - 0.04):
            decision = "TRADE"
            gate_mode = "breakout_rescue"
            why = f"BREAKOUT通過: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        else:
            decision = "NO_TRADE"
            _veto(f"EV不足: {ev_gate:+.3f} < 動的閾値 {float(dynamic_threshold):.3f}")

    # -----------------------------------------------------------------
    # 11) 状態確率 / EV内訳（UI用）
    # -----------------------------------------------------------------
    s_up = max(0.0, float(p_up) * (0.55 + 0.75 * float(strength)) + max(0.0, float(mom)) * 0.10)
    s_dn = max(0.0, float(p_dn) * (0.55 + 0.75 * float(strength)) + max(0.0, -float(mom)) * 0.10)
    s_range = max(0.0, (1.0 - float(strength)) * 0.95 + 0.05)
    s_risk = max(0.0, float(macro_risk) * 1.15 + (1.0 - float(cont_best)) * 0.10)

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

    if direction == "LONG":
        r_up = max(0.2, float(rr) * 0.85)
        r_dn = -1.0
    else:
        r_dn = max(0.2, float(rr) * 0.85)
        r_up = -1.0
    r_range = (0.12 * float(rr) - 0.35)
    r_riskoff = -0.75
    ev_contribs = {
        "trend_up": float(state_probs["trend_up"] * r_up),
        "trend_down": float(state_probs["trend_down"] * r_dn),
        "range": float(state_probs["range"] * r_range),
        "risk_off": float(state_probs["risk_off"] * r_riskoff),
    }

    # -----------------------------------------------------------------
    # 12) 統合スコア（単一ランキング指標）
    #   - 価格構造を最優先（structure_weight大）
    #   - EVは次点
    #   - イベント影響（通貨別）はペナルティとして統合 → 非影響通貨ペアが相対的に上位化
    # -----------------------------------------------------------------
    structure_score = (
        0.60 * float(strength)
        + (0.25 * float(breakout_strength) if bool(breakout_ok) else 0.0)
        + (0.15 if bool(hhhl_ok) else 0.0)
        + 0.10 * float(_clamp(abs(float(mom)), 0.0, 1.0))
    )
    if phase_label == "RANGE":
        structure_score *= 0.85
    structure_scaled = float(_clamp(structure_score, 0.0, 1.2) / 1.2)

    ev_scaled = float(_clamp((ev_gate + 0.15) / 1.35, 0.0, 1.0))

    ef_up = float(ev_meta.get("factor", 0.0) or 0.0)
    event_pen = 0.20 * ef_up
    if event_mode == "PRE_EVENT":
        event_pen = 0.28 * ef_up
    if event_mode == "POST_BREAKOUT":
        event_pen = 0.10 * ef_up

    event_pen += 0.08 * float(weekend_risk or 0.0) + 0.08 * float(weekcross_risk or 0.0)
    macro_pen = 0.08 * float(macro_risk)

    rank_score = float(
        2.00 * structure_scaled
        + 1.40 * ev_scaled
        + 0.40 * float(confidence)
        - float(event_pen)
        - float(macro_pen)
    )
    final_score = rank_score
    decision_score = float(ev_gate)
    ranking_score = float(rank_score)
    execution_score = float(confidence * (1.0 + max(0.0, float(strength))))

    # -----------------------------------------------------------------
    # 13) ctx（デバッグ/可視化用）
    # -----------------------------------------------------------------
    ctx_out = {
        "pair": pair,
        "trade_profile": str(trade_profile.get("name", "SWING")),
        "price_interval": str(trade_profile.get("interval", ctx_in.get("price_interval", "1d"))),
        "model_horizon_bars": int(model_horizon),
        "phase_label": phase_label,
        "trend_strength": float(strength),
        "momentum_score": float(mom),
        "range_pos": float(range_pos),
        "range_edge_setup": bool(range_edge_setup),
        "market_regime": str(regime),
        "volatility_expansion": float(vol_exp),
        "liquidity_sweep": bool(_liquidity_sweep(df)),
        "ccy_strength_proxy": float(ccy_strength_proxy),
        "ccy_bonus": float(ccy_bonus),
        "cont_p_up": float(p_up),
        "cont_p_dn": float(p_dn),
        "hh_hl_ok": bool(hhhl_ok),
        "ll_lh_ok": bool(lllh_ok),
        "structure_long": bool(structure_long),
        "structure_short": bool(structure_short),
        "structure_dir_ok": bool(structure_dir_ok),
        "breakout_ok": bool(breakout_ok),
        "breakout_strength": float(breakout_strength),
        "breakout_pass": bool(breakout_pass),
        "rr": float(rr),
        "p_win_model": float(p_win_model),
        "p_eff": float(p_eff),
        "macro_risk_score": float(macro_risk),
        "event_mode": str(event_mode),
        "event_risk_score": float(ev_meta.get("score", 0.0) or 0.0),
        "event_risk_factor": float(ev_meta.get("factor", 0.0) or 0.0),
        "event_window_high": bool(ev_meta.get("window_high", False)),
        "event_next_high_hours": (float(ev_meta.get("next_high_hours")) if ev_meta.get("next_high_hours") is not None else None),
        "event_last_high_hours": (float(ev_meta.get("last_high_hours")) if ev_meta.get("last_high_hours") is not None else None),
        "event_feed_status": str(ev_meta.get("status", "") or ""),
        "event_feed_error": str(ev_meta.get("err", "") or ""),
        "event_upcoming": (ev_meta.get("upcoming", []) or []),
        "event_recent": (ev_meta.get("recent", []) or []),
        "event_impact_ccys": (ev_meta.get("impact_ccys", {}) or {}),
        "weekend_risk": float(weekend_risk),
        "weekcross_risk": float(weekcross_risk or 0.0),
        "weekcross_weekday": (int(weekcross_weekday) if weekcross_weekday is not None else None),
        "mom_bonus": float(mom_bonus),
        "dynamic_threshold": float(dynamic_threshold),
        "dynamic_threshold_base": float(base_thr),
        "dynamic_threshold_mult": float(thr_mult),
        "ev_gate": float(ev_gate),
        "structure_scaled": float(structure_scaled),
        "ev_scaled": float(ev_scaled),
        "rank_score": float(rank_score),
        "event_penalty": float(event_pen),
        "macro_penalty": float(macro_pen),
        "len": int(len(df)),
        "regime_tp_multiple": float(regime_mult),
        "liquidity_tp": (float(liq_tp) if liq_tp is not None else None),
        "tp1": (float(tp1) if tp1 is not None else None),
        "tp2": float(tp2),
        "mtf_alignment_score": float(mtf_alignment_score),
        "fast_tf_dir_ok": bool(mtf_fast.get("dir_ok", False)) if isinstance(mtf_fast, dict) and mtf_fast else None,
        "micro_tf_dir_ok": bool(mtf_micro.get("dir_ok", False)) if isinstance(mtf_micro, dict) and mtf_micro else None,
        "price_now": float((mtf_micro.get("last_close") if isinstance(mtf_micro, dict) and mtf_micro else entry)),
        "liquidity_tp_intraday": (float(liq_tp_intraday) if liq_tp_intraday is not None else None),
        "tp1_basis": ("intraday_structure" if bool(trade_profile.get("is_intraday")) else "risk_multiple"),
        "tp1_liquidity_capped": bool(tp1_liquidity_capped),
        "tp2_compress_factor": float(tp2_compress_factor),
        "tp2_compressed": bool(tp2_compressed),
        "tp2_hard_capped": bool(tp2_hard_capped),
        "tp2_cap_reason": str(tp2_cap_reason),
        "max_hold_hours": float(trade_profile.get("max_hold_hours", 0.0) or 0.0),
        "stale_after_hours": float(trade_profile.get("stale_after_hours", 0.0) or 0.0),
        "be_trigger_r": float(trade_profile.get("be_trigger_r", 0.0) or 0.0),
        "trail_trigger_r": float(trade_profile.get("trail_trigger_r", 0.0) or 0.0),
    }

    # -----------------------------------------------------------------
    # 14) 注文方式の提案（直前:成行禁止 / 直後:ブレイク専用）
    # -----------------------------------------------------------------
    order_type = "MARKET"
    entry_type = "MARKET_NOW"
    exec_guard_notes: List[str] = []

    # setup-based suggestion (even for NO_TRADE; UI上は参考として表示可能)
    setup_kind = "TREND"
    if phase_label == "RANGE" and bool(range_edge_setup):
        setup_kind = "RANGE_EDGE"
    elif bool(breakout_pass) or str(phase).startswith("BREAKOUT") or (event_mode == "POST_BREAKOUT"):
        setup_kind = "BREAKOUT"

    try:
        pip = _pip_size(pair)
        atr_for_entry = max(float(atr14), float(pip) * 10.0)
    except Exception:
        pip = 0.01
        atr_for_entry = float(atr14)

    # Base reco by setup
    if setup_kind == "RANGE_EDGE":
        if direction == "LONG":
            new_entry = entry - 0.25 * atr_for_entry
        else:
            new_entry = entry + 0.25 * atr_for_entry
        order_type = "LIMIT"
        entry_type = "LIMIT_PULLBACK"
        exec_guard_notes.append("レンジ端のため、押し目/戻りの指値を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    elif setup_kind == "BREAKOUT":
        if direction == "LONG":
            new_entry = entry + 0.10 * atr_for_entry
            entry_type = "STOP_BREAKOUT"
        else:
            new_entry = entry - 0.10 * atr_for_entry
            entry_type = "STOP_BREAKDOWN"
        order_type = "STOP"
        exec_guard_notes.append("ブレイク捕獲のため、逆指値（STOP）を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    # High-impact is close enough → ban MARKET entry (直前:成行禁止)
    event_market_ban_active = False
    event_market_ban_hours = float(ctx_in.get("event_market_ban_hours", trade_profile.get("event_market_ban_hours", 12.0)) or trade_profile.get("event_market_ban_hours", 12.0))
    if float(weekcross_risk or 0.0) > 0.0:
        event_market_ban_hours = max(event_market_ban_hours, float(ctx_in.get("weekcross_market_ban_hours", 18.0) or 18.0))
    try:
        nh = ev_meta.get("next_high_hours", None)
        if (nh is not None) and (float(nh) <= float(event_market_ban_hours)):
            event_market_ban_active = True
    except Exception:
        pass

    if decision == "TRADE" and bool(event_market_ban_active) and order_type == "MARKET":
        # If we still ended up MARKET, convert to pending
        try:
            nh = float(ev_meta.get("next_high_hours") or 0.0)
        except Exception:
            nh = None
        if phase_label == "RANGE" and (not breakout_pass):
            # pullback limit
            if direction == "LONG":
                new_entry = entry - 0.25 * atr_for_entry
            else:
                new_entry = entry + 0.25 * atr_for_entry
            order_type = "LIMIT"
            entry_type = "LIMIT_PULLBACK"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → 押し目/戻りの指値を提案"
        else:
            # breakout stop
            if direction == "LONG":
                new_entry = entry + 0.10 * atr_for_entry
                entry_type = "STOP_BREAKOUT"
            else:
                new_entry = entry - 0.10 * atr_for_entry
                entry_type = "STOP_BREAKDOWN"
            order_type = "STOP"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → ブレイク逆指値を提案"

        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

        exec_guard_notes.append(msg)
        if why:
            why = why + " / " + msg
        else:
            why = msg

    # lot shrink factors (UI/logging)
    try:
        ef = float(ctx_out.get("event_risk_factor", 0.0) or 0.0)
        ctx_out["event_market_ban_active"] = bool(event_market_ban_active)
        ctx_out["event_market_ban_hours"] = float(event_market_ban_hours)
        ctx_out["exec_guard_notes"] = list(exec_guard_notes)
        ctx_out["order_type_reco"] = str(order_type)
        ctx_out["entry_type_reco"] = str(entry_type)
        ctx_out["lot_shrink_event_factor"] = float(_clamp(1.0 - 0.60 * ef, 0.20, 1.00))
        ctx_out["lot_shrink_weekcross_factor"] = (0.75 if float(weekcross_risk or 0.0) > 0.0 else 1.0)
        ctx_out["lot_shrink_weekend_factor"] = (0.60 if float(weekend_risk or 0.0) > 0.0 else 1.0)
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 15) 保有中のイベント接近対応（縮退/一部利確/建値移動/追加禁止）
    # -----------------------------------------------------------------
    hold_manage = _hold_manage_reco(
        pair=str(pair),
        df=df,
        ctx_in=ctx_in,
        plan_like={"side": side, "entry": entry, "sl": sl, "tp": tp},
        ev_meta=(ev_meta or {}),
        weekend_risk=float(weekend_risk or 0.0),
        weekcross_risk=float(weekcross_risk or 0.0),
    )
    if isinstance(hold_manage, dict) and hold_manage:
        try:
            ctx_out["hold_manage"] = hold_manage
        except Exception:
            pass

    # Trail SL: エントリーから0.5R戻し（見せ方用）
    trail_sl = sl
    try:
        dist_sl = abs(entry - sl)
        if dist_sl > 0:
            trail_sl = entry - 0.5 * dist_sl if direction == "LONG" else (entry + 0.5 * dist_sl)
    except Exception:
        trail_sl = sl

    # 返却（main互換キー）
    plan = {
        "decision": str(decision),
        "direction": str(direction),
        "side": str(side),

        "order_type": str(order_type),
        "entry_type": str(entry_type),

        "entry": float(entry),
        "entry_price": float(entry),
        "sl": float(sl),
        "stop_loss": float(sl),
        "tp": float(tp),
        "take_profit": float(tp),
        "tp1": (float(tp1) if tp1 is not None else float(tp)),
        "tp2": float(tp2),
        "tp1_basis": ("intraday_structure" if bool(trade_profile.get("is_intraday")) else "risk_multiple"),
        "tp1_liquidity_capped": bool(tp1_liquidity_capped),
        "tp2_compress_factor": float(tp2_compress_factor),
        "tp2_compressed": bool(tp2_compressed),
        "tp2_hard_capped": bool(tp2_hard_capped),
        "tp2_cap_reason": str(tp2_cap_reason),
        "partial_tp_enabled": True,
        "price_now": float((mtf_micro.get("last_close") if isinstance(mtf_micro, dict) and mtf_micro else entry)),
        "trade_profile": str(trade_profile.get("name", "SWING")),
        "price_interval": str(trade_profile.get("interval", ctx_in.get("price_interval", "1d"))),
        "max_hold_hours": float(trade_profile.get("max_hold_hours", 0.0) or 0.0),
        "stale_after_hours": float(trade_profile.get("stale_after_hours", 0.0) or 0.0),
        "be_trigger_r": float(trade_profile.get("be_trigger_r", 0.0) or 0.0),
        "trail_trigger_r": float(trade_profile.get("trail_trigger_r", 0.0) or 0.0),

        "trail_sl": float(trail_sl),
        "extend_factor": 1.0,

        "ev_raw": float(ev_raw),
        "ev_adj": float(ev_adj),

        "expected_R_ev_raw": float(ev_raw),
        "expected_R_ev_adj": float(ev_adj),
        "expected_R_ev": float(ev_gate),

        "rank_score": float(rank_score),
        "final_score": float(final_score),
        "decision_score": float(decision_score),
        "ranking_score": float(ranking_score),
        "execution_score": float(execution_score),

        "dynamic_threshold": float(dynamic_threshold),
        "gate_mode": str(gate_mode),

        "confidence": float(confidence),
        "p_win": float(p_eff),       # UIには縮退後を提示（整合性を優先）
        "p_eff": float(p_eff),
        "p_win_ev": float(p_eff),

        "event_mode": str(event_mode),
        "event_next_high_hours": (float(next_high) if next_high is not None else None),
        "event_last_high_hours": (float(last_high) if last_high is not None else None),

        "why": str(why),
        "veto": list(veto),
        "veto_reasons": list(veto),

        "state_probs": state_probs,
        "ev_contribs": ev_contribs,

        "hold_manage": (hold_manage if isinstance(hold_manage, dict) else {}),
        "_ctx": ctx_out,
    }
    return plan

# End of file




# ===============================
# FX_AI_PRO_v7 FULL PATCH BLOCK
# (non-breaking additive upgrades)
# ===============================

# --- probability normalization ---
def _normalize_probs(p_up: float, p_dn: float):
    try:
        p_up = float(p_up)
        p_dn = float(p_dn)
        s = p_up + p_dn
        if s > 1.0 and s > 0:
            p_up = p_up / s
            p_dn = p_dn / s
        return p_up, p_dn
    except Exception:
        return 0.5, 0.5


# --- volatility regime detection ---
def _volatility_regime(df):
    try:
        atr = (df["High"] - df["Low"]).rolling(14).mean()
        atr_mean = atr.rolling(50).mean()
        if atr.iloc[-1] > 1.4 * atr_mean.iloc[-1]:
            return "VOL_EXPANSION"
        elif atr.iloc[-1] < 0.7 * atr_mean.iloc[-1]:
            return "VOL_COMPRESSION"
        return "VOL_NORMAL"
    except Exception:
        return "VOL_UNKNOWN"


# --- liquidity sweep detection ---
def _detect_liquidity_sweep(df):
    try:
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)

        recent_high = highs.tail(20).max()
        prev_high = highs.tail(40).head(20).max()

        recent_low = lows.tail(20).min()
        prev_low = lows.tail(40).head(20).min()

        sweep_up = recent_high > prev_high * 1.0005
        sweep_down = recent_low < prev_low * 0.9995

        return {"sweep_up": bool(sweep_up), "sweep_down": bool(sweep_down)}
    except Exception:
        return {"sweep_up": False, "sweep_down": False}


# --- no-lookahead helper (use closed bar) ---
def _last_closed(series):
    try:
        return series.iloc[-2]
    except Exception:
        return series.iloc[-1]


# --- improved trend strength model ---
def _trend_strength_v7(adx, slope, atr_expansion):
    try:
        return max(0.0, min(1.0, 0.5*adx + 0.3*slope + 0.2*atr_expansion))
    except Exception:
        return 0.0

# ===============================
# END PATCH
# ===============================



# ---------------------------------------------------------------------
# QUALITY FILTER PATCH (added automatically)
# Purpose: reduce low‑quality trades without reducing win rate.
# It attenuates EV when structure quality is weak.
# This patch is backward compatible and safe.
# ---------------------------------------------------------------------

def _apply_quality_filter(ev_gate, strength, confidence, breakout_ok, hhhl_ok):
    try:
        breakout_flag = 1.0 if breakout_ok else 0.0
        hhhl_flag = 1.0 if hhhl_ok else 0.0

        quality = (
            0.35 * float(strength)
            + 0.30 * float(confidence)
            + 0.20 * float(breakout_flag)
            + 0.15 * float(hhhl_flag)
        )

        quality = max(0.35, min(1.0, quality))
        return float(ev_gate) * quality
    except Exception:
        return ev_gate
