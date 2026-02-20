# logic.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import math

import pandas as pd


PAIR_MAP = {
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in exps}


def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {
            "sma25": 0.0, "sma75": 0.0, "rsi": 50.0, "atr": 0.0,
            "recent_high20": 0.0, "recent_low20": 0.0,
            "atr_ratio": 0.0, "trend_strength": 0.0,
        }

    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in d.columns:
            return {
                "sma25": 0.0, "sma75": 0.0, "rsi": 50.0, "atr": 0.0,
                "recent_high20": 0.0, "recent_low20": 0.0,
                "atr_ratio": 0.0, "trend_strength": 0.0,
            }

    d = d[["Open", "High", "Low", "Close"]].dropna()
    if d.empty:
        return {
            "sma25": 0.0, "sma75": 0.0, "rsi": 50.0, "atr": 0.0,
            "recent_high20": 0.0, "recent_low20": 0.0,
            "atr_ratio": 0.0, "trend_strength": 0.0,
        }

    close = d["Close"]
    high = d["High"]
    low = d["Low"]

    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    price = float(close.iloc[-1])
    atr_v = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    sma25_v = float(sma25.iloc[-1]) if pd.notna(sma25.iloc[-1]) else price
    sma75_v = float(sma75.iloc[-1]) if pd.notna(sma75.iloc[-1]) else price
    rsi_v = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    recent_high20 = float(high.tail(20).max())
    recent_low20 = float(low.tail(20).min())

    eps = 1e-9
    atr_ratio = atr_v / max(price, eps)
    trend_strength = abs(sma25_v - sma75_v) / max(atr_v, eps) if atr_v > 0 else 0.0

    return {
        "sma25": sma25_v,
        "sma75": sma75_v,
        "rsi": rsi_v,
        "atr": atr_v,
        "recent_high20": recent_high20,
        "recent_low20": recent_low20,
        "atr_ratio": float(atr_ratio),
        "trend_strength": float(trend_strength),
    }


def _state_probs(ctx: Dict[str, Any]) -> Dict[str, float]:
    price = float(ctx.get("price") or 0.0)
    sma25 = float(ctx.get("sma25") or price)
    sma75 = float(ctx.get("sma75") or price)
    rsi = float(ctx.get("rsi") or 50.0)
    atr_ratio = float(ctx.get("atr_ratio") or 0.0)
    trend_strength = float(ctx.get("trend_strength") or 0.0)

    news = float(ctx.get("news_sentiment") or 0.0)
    cpi = float(ctx.get("cpi_surprise") or 0.0)
    nfp = float(ctx.get("nfp_surprise") or 0.0)
    rate = float(ctx.get("rate_diff_change") or 0.0)

    up = 0.0
    down = 0.0
    rng = 0.0
    risk = 0.0

    ma_spread = (sma25 - sma75) / max(price, 1e-9)
    up += 6.0 * ma_spread + 0.03 * (rsi - 50.0) + 0.6 * trend_strength
    down += -6.0 * ma_spread + 0.03 * (50.0 - rsi) + 0.6 * trend_strength

    rng += -1.2 * trend_strength + 0.02 * (1.0 - abs(rsi - 50.0) / 50.0)

    risk += 80.0 * atr_ratio + 0.8 * abs(rate) - 0.3 * news
    risk += 0.02 * (abs(cpi) + abs(nfp))

    return _softmax({"trend_up": up, "trend_down": down, "range": rng, "risk_off": risk})


def _state_stats_ev_stub() -> Dict[str, Dict[str, float]]:
    return {
        "trend_up": {"mean_R": 0.08, "n": 120},
        "trend_down": {"mean_R": 0.08, "n": 120},
        "range": {"mean_R": 0.01, "n": 120},
        "risk_off": {"mean_R": -0.10, "n": 60},
    }


def _shrink_mean_R(mean_R: float, n: float, n_ref: float = 30.0) -> float:
    w = _clamp(float(n) / float(n_ref), 0.0, 1.0)
    return float(mean_R) * w


def _ev_from_probs(state_probs: Dict[str, float], state_stats: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    contribs: Dict[str, float] = {}
    ev = 0.0
    for st, p in state_probs.items():
        stats = state_stats.get(st, {"mean_R": 0.0, "n": 0.0})
        mean_R = float(stats.get("mean_R", 0.0))
        n = float(stats.get("n", 0.0))
        mean_R_adj = _shrink_mean_R(mean_R, n, n_ref=30.0)
        c = float(p) * mean_R_adj
        contribs[st] = c
        ev += c
    return float(ev), contribs


def _pwin_from_probs(state_probs: Dict[str, float], state_stats: Dict[str, Dict[str, float]]) -> float:
    pwin = 0.5
    for st, p in state_probs.items():
        mean_R = float(state_stats.get(st, {}).get("mean_R", 0.0))
        pwin += float(p) * _clamp(mean_R, -0.2, 0.2) / 2.0
    return float(_clamp(pwin, 0.05, 0.95))


def _build_order(ctx: Dict[str, Any], state_probs: Dict[str, float], expected_R_ev: float) -> Dict[str, Any]:
    price = float(ctx.get("price") or 0.0)
    atr = float(ctx.get("atr") or 0.0)
    rh = float(ctx.get("recent_high20") or price)
    rl = float(ctx.get("recent_low20") or price)

    dom = max(state_probs.items(), key=lambda kv: kv[1])[0]

    order = {
        "decision": "NO_TRADE",
        "side": None,
        "order_type": None,
        "entry_type": None,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
    }

    if dom == "risk_off":
        return order

    if expected_R_ev <= 0:
        return order

    if atr <= 0:
        atr = max(price * 0.005, 0.0001)

    if dom in ("trend_up", "trend_down"):
        if dom == "trend_up":
            side = "BUY"
            entry = rh + 0.2 * atr
            sl = entry - 1.5 * atr
            tp = entry + 2.0 * atr
        else:
            side = "SELL"
            entry = rl - 0.2 * atr
            sl = entry + 1.5 * atr
            tp = entry - 2.0 * atr
        order.update({
            "decision": "TRADE",
            "side": side,
            "order_type": "STOP",
            "entry_type": "BREAKOUT_STOP",
            "entry": float(entry),
            "stop_loss": float(sl),
            "take_profit": float(tp),
        })
        return order

    side = "BUY" if price <= (rl + rh) / 2.0 else "SELL"
    if side == "BUY":
        entry = rl + 0.2 * atr
        sl = entry - 1.2 * atr
        tp = entry + 1.5 * atr
    else:
        entry = rh - 0.2 * atr
        sl = entry + 1.2 * atr
        tp = entry - 1.5 * atr

    order.update({
        "decision": "TRADE",
        "side": side,
        "order_type": "LIMIT",
        "entry_type": "MEANREV_LIMIT",
        "entry": float(entry),
        "stop_loss": float(sl),
        "take_profit": float(tp),
    })
    return order


def get_ai_order_strategy(api_key: str, context_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    ctx = dict(context_data or {})
    min_expected_R = float(ctx.get("min_expected_R") or 0.07)

    state_probs = _state_probs(ctx)
    state_stats = _state_stats_ev_stub()

    news = float(ctx.get("news_sentiment") or 0.0)
    rate = float(ctx.get("rate_diff_change") or 0.0)
    state_stats["trend_up"]["mean_R"] += 0.02 * news
    state_stats["trend_down"]["mean_R"] += 0.02 * (-news)
    state_stats["risk_off"]["mean_R"] += -0.02 * abs(rate)

    expected_R_ev, ev_contribs = _ev_from_probs(state_probs, state_stats)
    p_win_ev = _pwin_from_probs(state_probs, state_stats)

    why = f"EVゲート: expected_R_ev={expected_R_ev:+.3f} < min_expected_R={min_expected_R:.2f}"
    confidence = float(_clamp(abs(expected_R_ev) / max(min_expected_R, 1e-9), 0.0, 1.0))

    order = _build_order(ctx, state_probs, expected_R_ev)

    if expected_R_ev >= min_expected_R and order.get("decision") == "TRADE":
        why = f"EVゲート通過: expected_R_ev={expected_R_ev:+.3f} ≥ min_expected_R={min_expected_R:.2f}"
        confidence = float(_clamp(expected_R_ev / max(min_expected_R, 1e-9), 0.0, 1.0))
        decision = "TRADE"
    else:
        decision = "NO_TRADE"

    out: Dict[str, Any] = {
        "decision": decision if decision == "NO_TRADE" else order.get("decision", "TRADE"),
        "side": order.get("side"),
        "order_type": order.get("order_type"),
        "entry_type": order.get("entry_type"),
        "entry": order.get("entry"),
        "stop_loss": order.get("stop_loss"),
        "take_profit": order.get("take_profit"),
        "confidence": confidence,
        "why": why,
        "state_probs": state_probs,
        "expected_R_ev": float(expected_R_ev),
        "p_win_ev": float(p_win_ev),
        "ev_contribs": ev_contribs,
        "state_stats_ev": state_stats,
    }
    return out
