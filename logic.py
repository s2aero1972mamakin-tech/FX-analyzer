# logic.py
from __future__ import annotations

import os
import math
import time
import json
from typing import Dict, Any, Tuple, Optional

import pandas as pd

try:
    import data_layer  # type: ignore
except Exception:
    data_layer = None  # type: ignore

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
    mx = max(logits.values())
    exps = {k: math.exp(v - mx) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in exps}

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    if d is None or d.empty:
        return {"atr": 0.0, "rsi": 50.0, "sma25": 0.0, "sma75": 0.0, "recent_high20": 0.0, "recent_low20": 0.0}

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in d.columns:
            raise ValueError("price df missing columns")

    d = d[["Open", "High", "Low", "Close"]].dropna()
    d["SMA_25"] = d["Close"].rolling(25).mean()
    d["SMA_75"] = d["Close"].rolling(75).mean()

    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    d["RSI"] = 100 - (100 / (1 + rs))

    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    d["TR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR"] = d["TR"].rolling(14).mean()

    d = d.dropna()
    if d.empty:
        return {"atr": 0.0, "rsi": 50.0, "sma25": 0.0, "sma75": 0.0, "recent_high20": 0.0, "recent_low20": 0.0}

    last = d.iloc[-1]
    recent = d.iloc[-20:] if len(d) >= 20 else d
    return {
        "atr": float(last["ATR"]),
        "rsi": float(last["RSI"]),
        "sma25": float(last["SMA_25"]),
        "sma75": float(last["SMA_75"]),
        "recent_high20": float(recent["High"].max()),
        "recent_low20": float(recent["Low"].min()),
    }

def ensure_external_features(ctx: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    pair_label = str(ctx.get("pair_label") or ctx.get("pair") or "USD/JPY")
    keys = ctx.get("keys") or {}
    if not isinstance(keys, dict):
        keys = {}

    if data_layer is None or not hasattr(data_layer, "fetch_external_features"):
        feats = {
            "news_sentiment": 0.0,
            "cpi_surprise": 0.0,
            "nfp_surprise": 0.0,
            "rate_diff_change": 0.0,
            "cot_leveraged_net_pctoi": 0.0,
            "cot_asset_net_pctoi": 0.0,
        }
        meta = {"ok": False, "error": "data_layer_unavailable"}
        ctx.update(feats)
        ctx["external_meta"] = meta
        return feats, meta

    try:
        feats, meta = data_layer.fetch_external_features(pair_label, keys=keys)  # type: ignore[attr-defined]
    except Exception as e:
        feats = {
            "news_sentiment": 0.0,
            "cpi_surprise": 0.0,
            "nfp_surprise": 0.0,
            "rate_diff_change": 0.0,
            "cot_leveraged_net_pctoi": 0.0,
            "cot_asset_net_pctoi": 0.0,
        }
        meta = {"ok": False, "error": f"fetch_failed:{type(e).__name__}", "detail": str(e)}

    ctx.update(feats)
    ctx["external_meta"] = meta
    return feats, meta

_STATE_NAMES = ("trend_up", "trend_down", "range", "risk_off")

def get_state_probabilities_v1(ctx: Dict[str, Any]) -> Dict[str, float]:
    price = max(_safe_float(ctx.get("price"), 0.0), 1e-9)
    atr = max(_safe_float(ctx.get("atr"), 0.0), 1e-9)
    rsi = _safe_float(ctx.get("rsi"), 50.0)
    sma25 = _safe_float(ctx.get("sma25"), price)
    sma75 = _safe_float(ctx.get("sma75"), price)

    news = _safe_float(ctx.get("news_sentiment"), 0.0)
    cpi = _safe_float(ctx.get("cpi_surprise"), 0.0)
    nfp = _safe_float(ctx.get("nfp_surprise"), 0.0)
    spread_chg = _safe_float(ctx.get("rate_diff_change"), 0.0)
    cot_lev = _safe_float(ctx.get("cot_leveraged_net_pctoi"), 0.0)
    cot_ast = _safe_float(ctx.get("cot_asset_net_pctoi"), 0.0)

    slope = (sma25 - sma75) / price
    trend_strength = abs(sma25 - sma75) / atr
    atr_ratio = atr / price
    macro = _clamp((cpi + nfp) / 10.0, -2.0, 2.0) + _clamp(spread_chg, -2.0, 2.0)

    risk_off = 0.0
    risk_off += _clamp((atr_ratio - 0.010) * 80.0, -2.0, 2.0) * 0.35
    risk_off += _clamp((-news) * 2.0, -2.0, 2.0) * 0.35
    risk_off += _clamp((-cot_lev) * 2.0, -2.0, 2.0) * 0.20
    risk_off += _clamp((-cot_ast) * 2.0, -2.0, 2.0) * 0.10

    range_bias = _clamp((1.3 - trend_strength), -2.0, 2.0) + _clamp((0.010 - atr_ratio) * 50.0, -2.0, 2.0)

    logits = {
        "trend_up":  1.7 * slope + 0.22 * ((rsi - 50.0) / 10.0) + 0.18 * trend_strength + 0.18 * macro + 0.15 * news - 0.55 * risk_off,
        "trend_down": -1.7 * slope + 0.22 * ((50.0 - rsi) / 10.0) + 0.18 * trend_strength - 0.18 * macro - 0.15 * news + 0.55 * risk_off,
        "range": 0.85 * range_bias - 0.40 * trend_strength - 0.15 * abs(macro),
        "risk_off": 0.95 * risk_off + 0.10 * abs(macro),
    }
    probs = _softmax(logits)
    for k in _STATE_NAMES:
        probs.setdefault(k, 0.0)
    s = sum(probs.values()) or 1.0
    return {k: float(probs[k] / s) for k in probs}

_STATE_STATS_MEM: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _stats_key(symbol: str, horizon_days: int) -> str:
    return f"{symbol}::h{int(horizon_days)}"

def _stats_path(symbol: str, horizon_days: int) -> str:
    os.makedirs("cache", exist_ok=True)
    safe = symbol.replace("=", "_").replace("^", "_")
    return os.path.join("cache", f"ev_state_stats_{safe}_h{int(horizon_days)}.json")

def _load_stats(symbol: str, horizon_days: int) -> Optional[Dict[str, Any]]:
    p = _stats_path(symbol, horizon_days)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_stats(symbol: str, horizon_days: int, stats: Dict[str, Any]) -> None:
    p = _stats_path(symbol, horizon_days)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _compute_stats_yfinance(symbol: str, horizon_days: int, period: str = "10y") -> Dict[str, Any]:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval="1d")
    if df is None or df.empty:
        return {st: {"mean_R": 0.0, "p_win": 0.0, "n": 0} for st in _STATE_NAMES}
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    d = d[["Open","High","Low","Close"]].dropna()
    d["SMA_25"] = d["Close"].rolling(25).mean()
    d["SMA_75"] = d["Close"].rolling(75).mean()

    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    d["RSI"] = 100 - (100 / (1 + rs))

    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    d["TR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR"] = d["TR"].rolling(14).mean()

    d = d.dropna()
    d["fwd_ret"] = d["Close"].shift(-horizon_days) / d["Close"] - 1.0
    d["R"] = d["fwd_ret"] / (d["ATR"] / d["Close"]).replace(0, float("nan"))
    d = d.dropna(subset=["R"])

    states=[]
    for _, row in d.iterrows():
        c = {"price": float(row["Close"]), "atr": float(row["ATR"]), "rsi": float(row["RSI"]), "sma25": float(row["SMA_25"]), "sma75": float(row["SMA_75"])}
        pr = get_state_probabilities_v1(c)
        states.append(max(pr, key=pr.get))
    d["state"] = states

    out={}
    for st in _STATE_NAMES:
        sub = d[d["state"] == st]
        if sub.empty:
            out[st] = {"mean_R": 0.0, "p_win": 0.0, "n": 0}
        else:
            out[st] = {"mean_R": float(sub["R"].mean()), "p_win": float((sub["R"] > 0).mean()), "n": int(len(sub))}
    return out

def ensure_state_stats(symbol: str, horizon_days: int, ttl_sec: int = 72 * 3600) -> Dict[str, Any]:
    symbol = str(symbol or "JPY=X")
    horizon_days = int(horizon_days or 5)
    key = _stats_key(symbol, horizon_days)
    now = time.time()

    if key in _STATE_STATS_MEM:
        exp, stats = _STATE_STATS_MEM[key]
        if now <= exp:
            return stats


def apply_state_stats_smoothing(stats: Dict[str, Any], min_n: int = 30) -> Dict[str, Any]:
    """For small sample states, shrink mean_R toward 0 to avoid EV overreacting."""
    if not isinstance(stats, dict):
        return {}
    out: Dict[str, Any] = {}
    for st, s in stats.items():
        if not isinstance(s, dict):
            continue
        n = int(s.get("n", 0) or 0)
        mean_R = float(s.get("mean_R", 0.0) or 0.0)
        if n <= 0:
            mean_adj = 0.0
        else:
            w = min(1.0, max(0.0, n / float(min_n)))
            mean_adj = mean_R * w
        out[st] = {**s, "mean_R_adj": float(mean_adj), "shrink_w": float(min(1.0, max(0.0, n / float(min_n))) if n>0 else 0.0)}
    return out


    stats = _load_stats(symbol, horizon_days)
    if isinstance(stats, dict) and stats:
        _STATE_STATS_MEM[key] = (now + ttl_sec, stats)
        return stats

    stats = _compute_stats_yfinance(symbol, horizon_days, period="10y")
    _STATE_STATS_MEM[key] = (now + ttl_sec, stats)
    _save_stats(symbol, horizon_days, stats)
    return stats

def compute_ev(probs: Dict[str, float], stats: Dict[str, Any]) -> Tuple[float, float]:
    ev = 0.0
    pwin = 0.0
    for st, pr in probs.items():
        s = stats.get(st, {}) if isinstance(stats, dict) else {}
        mean_used = float(s.get("mean_R_adj", s.get("mean_R", 0.0)))
        ev += float(pr) * mean_used
        pwin += float(pr) * float(s.get("p_win", 0.0))
    return float(ev), float(pwin)


def compute_ev_details(probs: Dict[str, float], stats: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
    """Return (EV, p_win, contribs) where contribs[state] = P(state) * mean_R_used."""
    ev = 0.0
    pwin = 0.0
    contribs: Dict[str, float] = {}
    for st, pr in probs.items():
        s = stats.get(st, {}) if isinstance(stats, dict) else {}
        mean_used = float(s.get("mean_R_adj", s.get("mean_R", 0.0)))
        c = float(pr) * mean_used
        contribs[str(st)] = float(c)
        ev += c
        pwin += float(pr) * float(s.get("p_win", 0.0))
    return float(ev), float(pwin), contribs


def build_ev_order_plan(ctx: Dict[str, Any], probs: Dict[str, float], stats: Dict[str, Any], horizon: str = "WEEK") -> Dict[str, Any]:
    price = _safe_float(ctx.get("price"), 0.0)
    atr = max(_safe_float(ctx.get("atr"), 0.0), 1e-9)
    recent_high20 = _safe_float(ctx.get("recent_high20"), price)
    recent_low20 = _safe_float(ctx.get("recent_low20"), price)

    ev, pwin, ev_contribs = compute_ev_details(probs, stats)
    dom = max(probs, key=probs.get) if probs else "range"
    min_ev = _safe_float(ctx.get("min_expected_R"), 0.10)

    base = {"state_probs": probs, "expected_R_ev": ev, "p_win_ev": pwin, "ev_contribs": ev_contribs, "state_stats": stats, "external_meta": ctx.get("external_meta", {})}

    if ev < min_ev:
        return {**base, "decision": "NO_TRADE", "side": "NONE", "entry_type": "NONE", "entry": 0.0, "take_profit": 0.0, "stop_loss": 0.0,
                "horizon": horizon, "confidence": float(max(probs.values()) if probs else 0.0),
                "why": f"EV<{min_ev:.2f} のため見送り（expected_R={ev:.3f}）", "notes": [f"dominant_state={dom}"]}

    decision = "NO_TRADE"; side = "NONE"
    entry = tp = sl = 0.0
    entry_type = "NONE"

    if dom == "trend_up":
        side="BUY"; decision="STOP"; entry_type="BREAKOUT_STOP"
        buf = 0.15 * atr
        entry = max(price, recent_high20 + buf)
        sl = max(recent_low20, entry - 1.2 * atr)
        tp = entry + 2.0 * atr
    elif dom == "trend_down":
        side="SELL"; decision="STOP"; entry_type="BREAKOUT_STOP"
        buf = 0.15 * atr
        entry = min(price, recent_low20 - buf)
        sl = min(recent_high20, entry + 1.2 * atr)
        tp = entry - 2.0 * atr
    elif dom == "range":
        rng = max(recent_high20 - recent_low20, 1e-9)
        if price <= recent_low20 + 0.35 * rng:
            side="BUY"; decision="LIMIT"; entry_type="MEANREV_LIMIT"
            entry = price - 0.10 * atr
            sl = recent_low20 - 0.60 * atr
            tp = min(recent_high20, entry + 1.20 * atr)
        elif price >= recent_high20 - 0.35 * rng:
            side="SELL"; decision="LIMIT"; entry_type="MEANREV_LIMIT"
            entry = price + 0.10 * atr
            sl = recent_high20 + 0.60 * atr
            tp = max(recent_low20, entry - 1.20 * atr)
        else:
            return {**base, "decision": "NO_TRADE", "side": "NONE", "entry_type": "NONE", "entry": 0.0, "take_profit": 0.0, "stop_loss": 0.0,
                    "horizon": horizon, "confidence": float(max(probs.values()) if probs else 0.0),
                    "why": "RANGE優勢だが価格がレンジ端に無い（逆張り優位性が弱い）", "notes": [f"dominant_state={dom}"]}
    else:
        return {**base, "decision": "NO_TRADE", "side": "NONE", "entry_type": "NONE", "entry": 0.0, "take_profit": 0.0, "stop_loss": 0.0,
                "horizon": horizon, "confidence": float(max(probs.values()) if probs else 0.0),
                "why": "risk_off優勢のため見送り（急変動局面を回避）", "notes": [f"dominant_state={dom}"]}

    conf = float(max(probs.values()) if probs else 0.0)
    conf = float(_clamp(conf * (1.0 + _clamp(ev, -1.0, 1.0) * 0.25), 0.0, 0.99))
    return {**base, "decision": decision, "side": side, "entry_type": entry_type, "entry": float(entry), "take_profit": float(tp), "stop_loss": float(sl),
            "horizon": horizon, "confidence": conf, "why": f"EVベース決定: dominant={dom} / expected_R={ev:.3f} / p_win={pwin:.3f}",
            "notes": [f"entry_type={entry_type}"]}

def _llm_explain(api_key: str, plan: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[str]:
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        payload = {
            "pair": ctx.get("pair_label"),
            "price": ctx.get("price"),
            "decision_engine": ctx.get("decision_engine"),
            "external_features": {k: ctx.get(k) for k in ["news_sentiment","cpi_surprise","nfp_surprise","rate_diff_change","cot_leveraged_net_pctoi","cot_asset_net_pctoi"]},
            "state_probs": plan.get("state_probs"),
            "expected_R_ev": plan.get("expected_R_ev"),
            "plan": {k: plan.get(k) for k in ["decision","side","entry","take_profit","stop_loss","confidence"]},
        }
        prompt = "以下の情報を、短く運用者向けに説明してください。数値を変更しないでください。\n" + json.dumps(payload, ensure_ascii=False)
        r = model.generate_content(prompt)
        return getattr(r, "text", None)
    except Exception:
        return None

def get_ai_order_strategy(api_key: str, context_data: Dict[str, Any], generation_policy: str = "AUTO_HIERARCHY", override_mode: str = "AUTO", override_reason: str = "") -> Dict[str, Any]:
    ctx = context_data if isinstance(context_data, dict) else {}
    ensure_external_features(ctx)

    symbol = str(ctx.get("pair_symbol") or ctx.get("symbol") or PAIR_MAP.get(str(ctx.get("pair_label") or "USD/JPY"), "JPY=X"))
    horizon_days = int(ctx.get("horizon_days") or 5)

    probs = get_state_probabilities_v1(ctx)
    stats_raw = ensure_state_stats(symbol, horizon_days)
    stats = apply_state_stats_smoothing(stats_raw, min_n=30)
    plan = build_ev_order_plan(ctx, probs, stats, horizon="WEEK")

    engine = str(ctx.get("decision_engine") or "HYBRID").upper()
    if engine == "EV_V1":
        return plan
    if engine == "HYBRID" and plan.get("decision") == "NO_TRADE":
        return plan

    explanation = _llm_explain(api_key, plan, ctx)
    if explanation:
        plan["llm_explanation"] = explanation
    plan["generation_policy"] = generation_policy
    return plan

def get_daily_order_strategy(api_key: str, context_data: Dict[str, Any], max_risk_pct: float = 8.0) -> Dict[str, Any]:
    ctx = context_data if isinstance(context_data, dict) else {}
    ensure_external_features(ctx)
    symbol = str(ctx.get("pair_symbol") or ctx.get("symbol") or PAIR_MAP.get(str(ctx.get("pair_label") or "USD/JPY"), "JPY=X"))
    horizon_days = int(ctx.get("horizon_days") or 3)
    probs = get_state_probabilities_v1(ctx)
    stats_raw = ensure_state_stats(symbol, horizon_days)
    stats = apply_state_stats_smoothing(stats_raw, min_n=30)
    plan = build_ev_order_plan(ctx, probs, stats, horizon="DAY")

    engine = str(ctx.get("decision_engine") or "HYBRID").upper()
    if engine == "EV_V1":
        return plan

    explanation = _llm_explain(api_key, plan, ctx)
    if explanation:
        plan["llm_explanation"] = explanation
    return plan
