# logic.py (v4 integrated, backward-compatible with v3 UI)
from __future__ import annotations

from typing import Dict, Any, Tuple, List
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

# -------------------------
# Small utils
# -------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in exps}

# -------------------------
# Indicators (same as v3)
# -------------------------
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

# -------------------------
# State probs + overlays
# -------------------------
def _state_probs_base(ctx: Dict[str, Any]) -> Dict[str, float]:
    price = _safe_float(ctx.get("price"), 0.0)
    sma25 = _safe_float(ctx.get("sma25"), price)
    sma75 = _safe_float(ctx.get("sma75"), price)
    rsi = _safe_float(ctx.get("rsi"), 50.0)
    atr_ratio = _safe_float(ctx.get("atr_ratio"), 0.0)
    trend_strength = _safe_float(ctx.get("trend_strength"), 0.0)

    news = _safe_float(ctx.get("news_sentiment"), 0.0)
    cpi = _safe_float(ctx.get("cpi_surprise"), 0.0)
    nfp = _safe_float(ctx.get("nfp_surprise"), 0.0)
    rate = _safe_float(ctx.get("rate_diff_change"), 0.0)

    up = 0.0
    down = 0.0
    rng = 0.0
    risk = 0.0

    ma_spread = (sma25 - sma75) / max(price, 1e-9)
    up += 6.0 * ma_spread + 0.03 * (rsi - 50.0) + 0.6 * trend_strength
    down += -6.0 * ma_spread + 0.03 * (50.0 - rsi) + 0.6 * trend_strength

    rng += -1.2 * trend_strength + 0.02 * (1.0 - abs(rsi - 50.0) / 50.0)

    # baseline risk_off: volatility + macro shocks
    risk += 80.0 * atr_ratio + 0.8 * abs(rate) - 0.3 * news
    risk += 0.02 * (abs(cpi) + abs(nfp))

    return _softmax({"trend_up": up, "trend_down": down, "range": rng, "risk_off": risk})

def _apply_overlays(state_probs: Dict[str, float], ctx: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Applies macro/geo stress overlays to state probabilities.
    Returns (new_probs, overlay_meta)
    """
    p = dict(state_probs)
    macro = _clamp(_safe_float(ctx.get("macro_risk_score"), 0.0), 0.0, 1.0)
    global_risk = _clamp(_safe_float(ctx.get("global_risk_index"), 0.0), 0.0, 1.0)
    war_prob = _clamp(_safe_float(ctx.get("war_probability"), 0.0), 0.0, 1.0)
    fin_stress = _clamp(_safe_float(ctx.get("financial_stress"), 0.0), 0.0, 1.0)

    # Overlay strength: conservative. push more weight into risk_off when risk rises.
    bump = 0.35 * global_risk + 0.20 * macro + 0.20 * war_prob + 0.15 * fin_stress
    bump = _clamp(bump, 0.0, 0.65)

    # Take from non-risk states proportionally
    non = max(1e-9, p["trend_up"] + p["trend_down"] + p["range"])
    take = bump
    p["risk_off"] = _clamp(p["risk_off"] + bump, 0.0, 1.0)
    p["trend_up"] = _clamp(p["trend_up"] * (1.0 - take), 0.0, 1.0)
    p["trend_down"] = _clamp(p["trend_down"] * (1.0 - take), 0.0, 1.0)
    p["range"] = _clamp(p["range"] * (1.0 - take), 0.0, 1.0)

    # Normalize
    s = sum(p.values()) or 1.0
    p = {k: v / s for k, v in p.items()}

    meta = {
        "macro_risk_score": macro,
        "global_risk_index": global_risk,
        "war_probability": war_prob,
        "financial_stress": fin_stress,
        "risk_off_bump": bump,
    }
    return p, meta

# -------------------------
# EV model (still stub, but now modulated)
# -------------------------
def _state_stats_ev_base() -> Dict[str, Dict[str, float]]:
    # These are priors; you can later replace with empirical from backtests.
    return {
        "trend_up": {"mean_R": 0.08, "n": 120},
        "trend_down": {"mean_R": 0.08, "n": 120},
        "range": {"mean_R": 0.01, "n": 120},
        "risk_off": {"mean_R": -0.12, "n": 60},
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

# -------------------------
# Black Swan Guard + Capital Governor
# -------------------------
def evaluate_black_swan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns dict: {flag: bool, level: 'green/yellow/red', reasons: [...], metrics: {...}}
    Uses:
      - atr_ratio
      - vix
      - global_risk_index / war_probability / financial_stress
      - gdelt counts
    """
    atr_ratio = _safe_float(ctx.get("atr_ratio"), 0.0)
    vix = _safe_float(ctx.get("vix"), float("nan"))
    global_risk = _clamp(_safe_float(ctx.get("global_risk_index"), 0.0), 0.0, 1.0)
    war = _clamp(_safe_float(ctx.get("war_probability"), 0.0), 0.0, 1.0)
    fin = _clamp(_safe_float(ctx.get("financial_stress"), 0.0), 0.0, 1.0)
    war_cnt = _safe_float(ctx.get("gdelt_war_count_1d"), 0.0)
    fin_cnt = _safe_float(ctx.get("gdelt_finance_count_1d"), 0.0)

    reasons: List[str] = []
    level = "green"

    # thresholds are conservative; adjust later
    if atr_ratio > 0.02:  # daily ATR >2% of price (FX daily is often <1%)
        reasons.append(f"ATR比が高い(atr_ratio={atr_ratio:.3f})")
        level = "yellow"
    if not math.isnan(vix) and vix >= 35:
        reasons.append(f"VIX高水準(vix={vix:.1f})")
        level = "yellow"
    if global_risk >= 0.70:
        reasons.append(f"GlobalRisk高(global_risk={global_risk:.2f})")
        level = "red"
    if war >= 0.75:
        reasons.append(f"War確率高(war_prob={war:.2f})")
        level = "red"
    if fin >= 0.75:
        reasons.append(f"FinancialStress高(fin_stress={fin:.2f})")
        level = "red"
    if war_cnt >= 200:
        reasons.append(f"戦争関連ニュース急増(count={int(war_cnt)})")
        level = "red"
    if fin_cnt >= 250:
        reasons.append(f"金融危機関連ニュース急増(count={int(fin_cnt)})")
        level = "red"

    flag = level == "red"
    return {
        "flag": flag,
        "level": level,
        "reasons": reasons,
        "metrics": {
            "atr_ratio": atr_ratio,
            "vix": vix,
            "global_risk_index": global_risk,
            "war_probability": war,
            "financial_stress": fin,
            "gdelt_war_count_1d": war_cnt,
            "gdelt_finance_count_1d": fin_cnt,
        }
    }

def capital_governor(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Purely local risk governor using user-provided limits in ctx.
    This does NOT look at broker positions; it gates new trades.
    """
    max_dd = _safe_float(ctx.get("max_drawdown_limit"), 0.15)
    daily_stop = _safe_float(ctx.get("daily_loss_limit"), 0.03)
    cur_dd = _safe_float(ctx.get("equity_drawdown"), 0.0)
    cur_daily = _safe_float(ctx.get("daily_loss"), 0.0)
    losing_streak = int(_safe_float(ctx.get("losing_streak"), 0))
    max_streak = int(_safe_float(ctx.get("max_losing_streak"), 5))

    enabled = True
    reasons: List[str] = []
    if cur_dd >= max_dd:
        enabled = False
        reasons.append(f"DD制限超過({cur_dd:.2%} >= {max_dd:.0%})")
    if cur_daily >= daily_stop:
        enabled = False
        reasons.append(f"日次損失制限超過({cur_daily:.2%} >= {daily_stop:.0%})")
    if losing_streak >= max_streak:
        enabled = False
        reasons.append(f"連敗停止({losing_streak} >= {max_streak})")
    return {"enabled": enabled, "reasons": reasons, "limits": {"max_dd": max_dd, "daily_stop": daily_stop, "max_streak": max_streak}}

# -------------------------
# Order builder (same as v3, but uses dom state)
# -------------------------
def _build_order(ctx: Dict[str, Any], state_probs: Dict[str, float], expected_R_ev: float) -> Dict[str, Any]:
    price = _safe_float(ctx.get("price"), 0.0)
    atr = _safe_float(ctx.get("atr"), 0.0)
    rh = _safe_float(ctx.get("recent_high20"), price)
    rl = _safe_float(ctx.get("recent_low20"), price)

    dom = max(state_probs.items(), key=lambda kv: kv[1])[0]

    order = {
        "decision": "NO_TRADE",
        "side": None,
        "order_type": None,
        "entry_type": None,
        "entry": None,
        "stop_loss": None,
        "take_profit": None,
        "dominant_state": dom,
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

# -------------------------
# Public API: get_ai_order_strategy
# -------------------------
def get_ai_order_strategy(api_key: str, context_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Backward-compatible output + new fields:
      - overlay_meta
      - dynamic_threshold
      - black_swan
      - governor
      - veto_reasons
    """
    ctx = dict(context_data or {})
    base_threshold = _safe_float(ctx.get("min_expected_R"), 0.07)

    # base probs, then overlays
    probs0 = _state_probs_base(ctx)
    probs, overlay_meta = _apply_overlays(probs0, ctx)

    # state stats + small nudges
    state_stats = _state_stats_ev_base()
    news = _safe_float(ctx.get("news_sentiment"), 0.0)
    rate = _safe_float(ctx.get("rate_diff_change"), 0.0)
    state_stats["trend_up"]["mean_R"] += 0.02 * news
    state_stats["trend_down"]["mean_R"] += 0.02 * (-news)
    state_stats["risk_off"]["mean_R"] += -0.02 * abs(rate)

    expected_R_ev, ev_contribs = _ev_from_probs(probs, state_stats)
    p_win_ev = _pwin_from_probs(probs, state_stats)

    # Dynamic threshold: raise threshold in risky regimes
    macro = _clamp(_safe_float(ctx.get("macro_risk_score"), 0.0), 0.0, 1.0)
    global_risk = _clamp(_safe_float(ctx.get("global_risk_index"), 0.0), 0.0, 1.0)
    war = _clamp(_safe_float(ctx.get("war_probability"), 0.0), 0.0, 1.0)
    fin = _clamp(_safe_float(ctx.get("financial_stress"), 0.0), 0.0, 1.0)

    dynamic_threshold = base_threshold * (1.0 + 0.8*macro + 1.0*global_risk + 0.6*war + 0.6*fin)
    dynamic_threshold = float(_clamp(dynamic_threshold, base_threshold, 0.35))

    veto_reasons: List[str] = []

    # Governor & Black Swan guard
    gov = capital_governor(ctx)
    if not gov["enabled"]:
        veto_reasons.extend(gov["reasons"])

    bs = evaluate_black_swan(ctx)
    if bs["flag"]:
        veto_reasons.append("BlackSwanGuard: " + (" / ".join(bs["reasons"]) if bs["reasons"] else "red"))

    # Build order candidate
    order = _build_order(ctx, probs, expected_R_ev)

    # EV gate
    if expected_R_ev < dynamic_threshold:
        veto_reasons.append(f"EV不足: {expected_R_ev:+.3f} < 動的閾値 {dynamic_threshold:.3f}")

    # Final decision
    decision = "TRADE"
    if veto_reasons or order.get("decision") != "TRADE":
        decision = "NO_TRADE"

    confidence = float(_clamp(abs(expected_R_ev) / max(dynamic_threshold, 1e-9), 0.0, 1.0))

    why = " / ".join(veto_reasons) if veto_reasons else f"EV通過: {expected_R_ev:+.3f} ≥ {dynamic_threshold:.3f}"

    out: Dict[str, Any] = {
        "decision": decision,
        "side": order.get("side"),
        "order_type": order.get("order_type"),
        "entry_type": order.get("entry_type"),
        "entry": order.get("entry"),
        "stop_loss": order.get("stop_loss"),
        "take_profit": order.get("take_profit"),
        "dominant_state": order.get("dominant_state"),
        "confidence": confidence,
        "why": why,
        "state_probs": probs,
        "expected_R_ev": float(expected_R_ev),
        "p_win_ev": float(p_win_ev),
        "ev_contribs": ev_contribs,
        "state_stats_ev": state_stats,
        "overlay_meta": overlay_meta,
        "dynamic_threshold": float(dynamic_threshold),
        "black_swan": bs,
        "governor": gov,
        "veto_reasons": veto_reasons,
    }
    return out
