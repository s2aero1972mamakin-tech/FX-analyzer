# main.py (v4 integrated, keeps v3 features + adds global risk overlays)
from __future__ import annotations

import os
import time
import csv
import uuid
import math
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import streamlit as st
import requests
import pandas as pd

# ---- optional deps ----
try:
    import yfinance as yf
except Exception:
    yf = None

# ---- local modules ----
import logic

# Integrated external features
try:
    import data_layer
except Exception:
    data_layer = None

# yfinance rate-limit exception (version dependent)
try:
    from yfinance.exceptions import YFRateLimitError  # type: ignore
except Exception:
    class YFRateLimitError(Exception):
        pass

# =========================
# Utilities
# =========================
PAIR_LIST_DEFAULT = [
    "USD/JPY",
    "EUR/USD",
    "GBP/USD",
    "AUD/USD",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
]


# =========================
# Build / Diagnostics
# =========================
APP_BUILD = "fixed27_20260224"
# ---- EV audit (operator logs) ----
EV_AUDIT_PATH = "logs/ev_audit.csv"

def _ev_audit_append(row: Dict[str, Any], path: str = EV_AUDIT_PATH) -> None:
    """Append one row to ev_audit.csv. Never raises to caller."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            "ts_utc",
            "timeframe_mode",
            "pair",
            "decision_adj",
            "decision_raw",
            "killed_by_adj",
            "ev_raw",
            "ev_adj",
            "dynamic_threshold",
            "dominant_state",
            "confidence",
            "global_risk_index",
            "war_probability",
            "financial_stress",
            "macro_risk_score",
            "risk_off_bump",
        ]
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        # Audit must never break trading UI
        return

def _ev_audit_load(path: str = EV_AUDIT_PATH, max_rows: int = 20000) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                rows.append(row)
                if i >= max_rows:
                    break
        return rows
    except Exception:
        return []

def _ev_audit_summary(rows: List[Dict[str, Any]], days: int = 14) -> Dict[str, Any]:
    """Summarize last N days rows (best-effort; ts_utc ISO8601)."""
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    def parse_ts(s: str):
        try:
            # expecting 2026-02-23T12:34:56Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except Exception:
            return None
    recent = []
    for r in rows:
        ts = parse_ts(str(r.get("ts_utc","")))
        if ts and ts >= cutoff:
            recent.append(r)
    total = len(recent)
    trade = sum(1 for r in recent if str(r.get("decision_adj","")) == "TRADE")
    killed = sum(1 for r in recent if str(r.get("killed_by_adj","")).lower() in ("true","1","yes"))
    return {"days": days, "total": total, "trade": trade, "no_trade": total-trade, "killed_by_adj": killed}


# =========================
# Signal / Trade logs (for real operations)
# =========================
SIGNAL_LOG_PATH = "logs/signals.csv"
TRADE_LOG_PATH = "logs/trades.csv"

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _safe_makedirs_for(path: str) -> None:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass

def _append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> bool:
    """Append row to CSV with header creation. Returns True on success. Never raises."""
    try:
        _safe_makedirs_for(path)
        file_exists = os.path.exists(path)
        # If file exists but header differs, write to a versioned file instead (avoid corrupting).
        if file_exists:
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    first = f.readline().strip()
                if first and (first.split(",") != fieldnames):
                    base, ext = os.path.splitext(path)
                    path = f"{base}_v2{ext}"
                    file_exists = os.path.exists(path)
            except Exception:
                pass
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
        return True
    except Exception:
        return False

def _load_csv_df(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _make_signal_id(pair_symbol: str) -> str:
    ps = re.sub(r"[^A-Z0-9_\-]", "", (pair_symbol or "").upper())
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{ps}_{uuid.uuid4().hex[:6]}"

def _norm_side(x: Any) -> str:
    s = str(x or "").upper().strip()
    if s in ("LONG", "BUY"):
        return "LONG"
    if s in ("SHORT", "SELL"):
        return "SHORT"
    return s or "—"

def _float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def _calc_r_multiple(side: str, entry: float, exit_price: float, stop: float) -> Optional[float]:
    """R-multiple based on stop distance. Returns None if cannot compute."""
    try:
        side = _norm_side(side)
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        if side == "LONG":
            return (exit_price - entry) / risk
        if side == "SHORT":
            return (entry - exit_price) / risk
        return None
    except Exception:
        return None

def _build_signal_row(pair_label: str, ctx: Dict[str, Any], feats: Dict[str, Any], plan: Dict[str, Any],
                      price_meta: Dict[str, Any], ext_meta: Dict[str, Any]) -> Dict[str, Any]:
    sym = str(ctx.get("pair_symbol") or _pair_label_to_symbol(pair_label))
    sid = _make_signal_id(sym)
    return {
        "ts_utc": _now_utc_iso(),
        "signal_id": sid,
        "pair": str(pair_label),
        "symbol": sym,
        "timeframe_mode": str(st.session_state.get("timeframe_mode", "")),
        "style_name": str(ctx.get("style_name", "")),
        "priority_mode": str(st.session_state.get("priority_mode", "")),
        "decision": str(plan.get("decision", "")),
        "gate_mode": str(plan.get("gate_mode", "")),
        "ev_raw": _float_or_none(plan.get("expected_R_ev_raw", plan.get("expected_R_ev"))),
        "ev_adj": _float_or_none(plan.get("expected_R_ev_adj", plan.get("expected_R_ev"))),
        "ev_used": _float_or_none(plan.get("expected_R_ev")),
        "dynamic_threshold": _float_or_none(plan.get("dynamic_threshold")),
        "confidence": _float_or_none(plan.get("confidence")),
        "p_win": _float_or_none(plan.get("p_win_ev")),
        "dominant_state": _dominant_state(plan.get("state_probs", {})),
        "direction": _norm_side(plan.get("direction", "")),
        "entry_hint": _float_or_none(plan.get("entry_price")),
        "sl_hint": _float_or_none(plan.get("stop_loss")),
        "tp_hint": _float_or_none(plan.get("take_profit")),
        "trail_sl_hint": _float_or_none(plan.get("trail_sl")),
        "tp_extend_factor": _float_or_none(plan.get("extend_factor")),
        "global_risk_index": _float_or_none(feats.get("global_risk_index")),
        "war_probability": _float_or_none(feats.get("war_probability")),
        "financial_stress": _float_or_none(feats.get("financial_stress")),
        "macro_risk_score": _float_or_none(feats.get("macro_risk_score")),
        "price_close": _float_or_none(ctx.get("price")),
        "price_meta": (json.dumps(price_meta, ensure_ascii=False)[:2000] if isinstance(price_meta, dict) else ""),
        "external_meta": (json.dumps(ext_meta, ensure_ascii=False)[:2000] if isinstance(ext_meta, dict) else ""),
        "why": str(plan.get("why", ""))[:500],
        "veto_reasons": (json.dumps(plan.get("veto_reasons", []), ensure_ascii=False)[:500] if plan.get("veto_reasons") else ""),
    }

def _append_signal(row: Dict[str, Any]) -> bool:
    fieldnames = [
        "ts_utc","signal_id","pair","symbol","timeframe_mode","style_name","priority_mode",
        "decision","gate_mode","ev_raw","ev_adj","ev_used","dynamic_threshold","confidence","p_win","dominant_state",
        "direction","entry_hint","sl_hint","tp_hint","trail_sl_hint","tp_extend_factor",
        "global_risk_index","war_probability","financial_stress","macro_risk_score","price_close",
        "why","veto_reasons","price_meta","external_meta",
    ]
    return _append_csv_row(SIGNAL_LOG_PATH, fieldnames, row)

def _append_trade(row: Dict[str, Any]) -> bool:
    fieldnames = [
        "ts_record_utc","trade_id","signal_id","pair","symbol","side","ts_open_utc","ts_close_utc",
        "entry","exit","stop","take_profit","r_multiple","comment",
    ]
    return _append_csv_row(TRADE_LOG_PATH, fieldnames, row)

def _compute_trade_metrics(df_trades: pd.DataFrame) -> Dict[str, Any]:
    if df_trades is None or df_trades.empty:
        return {"n": 0}
    d = df_trades.copy()
    # numeric coercions
    for c in ["entry","exit","stop","take_profit","r_multiple"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "r_multiple" not in d.columns:
        return {"n": int(len(d))}
    r = d["r_multiple"].dropna()
    if r.empty:
        return {"n": int(len(d))}
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = (wins.sum() / abs(losses.sum())) if (not losses.empty and abs(losses.sum()) > 1e-9) else (float("inf") if wins.sum() > 0 else None)
    cum = r.cumsum()
    dd = (cum - cum.cummax()).min() if len(cum) else 0.0
    return {
        "n": int(len(r)),
        "expectancy_R": float(r.mean()),
        "median_R": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": (float(pf) if pf is not None and math.isfinite(pf) else pf),
        "max_drawdown_R": float(dd),
        "sum_R": float(r.sum()),
    }

def _render_logging_panel(pair_label: str, plan_ui: Dict[str, Any], ctx: Dict[str, Any], feats: Dict[str, Any],
                          price_meta: Dict[str, Any], ext_meta: Dict[str, Any]):
    """UI: save signal + record trade outcome. Keeps everything optional and non-blocking."""
    st.markdown("### 📝 シグナル/損益ログ（運用用）")
    with st.expander("📝 保存（signals / trades）+ 外部Sink（Webhook/Supabase）", expanded=False):
        row = _build_signal_row(pair_label, ctx, feats, plan_ui, price_meta=price_meta, ext_meta=ext_meta)
        st.caption("運用の第一歩：**『シグナル保存 → 決済後に損益保存 → パフォーマンス自動集計』** の流れを固定します。")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ このシグナルを保存", key=f"save_signal_{row['signal_id']}"):
                ok_local = _append_signal(row)
                st.session_state["last_signal_id"] = row["signal_id"]
                st.session_state["last_signal_row"] = row
                st.session_state["last_plan"] = plan_ui
                st.session_state["last_ctx"] = ctx
                st.session_state["last_feats"] = feats
                # external sinks (best-effort)
                ext = _external_log_event("signal", row)
                if ok_local:
                    st.success(f"保存OK: signal_id={row['signal_id']}")
                else:
                    st.warning("ローカル保存に失敗（外部Sinkの結果を確認してください）")
                st.json(ext)
        with c2:
            st.write("**signal_id**")
            st.code(row["signal_id"])
            st.caption("このIDを決済記録に紐づけると分析が強くなります。")

        st.divider()
        st.subheader("決済後：トレード結果を保存（Rの自動計算）")

        default_sid = str(st.session_state.get("last_signal_id", row["signal_id"]))
        default_side = _norm_side(plan_ui.get("direction", "LONG"))
        default_entry = _float_or_none(plan_ui.get("entry_price")) or _float_or_none(ctx.get("price")) or 0.0
        default_stop = _float_or_none(plan_ui.get("stop_loss")) or 0.0
        default_tp = _float_or_none(plan_ui.get("take_profit")) or 0.0

        form_key = f"trade_close_{pair_label.replace('/','_')}"
        with st.form(form_key, clear_on_submit=False):
            signal_id = st.text_input("signal_id（任意だが推奨）", value=default_sid)
            side = st.selectbox("方向", ["LONG", "SHORT"], index=0 if default_side == "LONG" else 1)
            entry = st.number_input("Entry（約定値）", value=float(default_entry), format="%.5f")
            exit_price = st.number_input("Exit（決済値）", value=float(default_tp or default_entry), format="%.5f")
            stop = st.number_input("Stop（実際のSL）", value=float(default_stop or default_entry), format="%.5f")
            take_profit = st.number_input("TP（実際のTP）", value=float(default_tp or default_entry), format="%.5f")
            ts_open_utc = st.text_input("Open時刻（UTC, ISO）", value=_now_utc_iso())
            ts_close_utc = st.text_input("Close時刻（UTC, ISO）", value=_now_utc_iso())
            comment = st.text_input("メモ（任意）", value="")
            submitted = st.form_submit_button("💾 決済結果を保存")

        if submitted:
            r_mult = _calc_r_multiple(side, float(entry), float(exit_price), float(stop))
            trade_row = {
                "ts_record_utc": _now_utc_iso(),
                "trade_id": uuid.uuid4().hex[:12],
                "signal_id": str(signal_id),
                "pair": str(pair_label),
                "symbol": str(ctx.get("pair_symbol") or _pair_label_to_symbol(pair_label)),
                "side": _norm_side(side),
                "ts_open_utc": str(ts_open_utc),
                "ts_close_utc": str(ts_close_utc),
                "entry": float(entry),
                "exit": float(exit_price),
                "stop": float(stop),
                "take_profit": float(take_profit),
                "r_multiple": (float(r_mult) if r_mult is not None else ""),
                "comment": str(comment)[:500],
            }
            ok_local = _append_trade(trade_row)
            ext = _external_log_event("trade_close", trade_row)
            if ok_local:
                st.success(f"保存OK: trade_id={trade_row['trade_id']} / R={r_mult if r_mult is not None else '—'}")
            else:
                st.warning("ローカル保存に失敗（外部Sinkの結果を確認してください）")
            st.json(ext)

        st.divider()
        st.caption("ローカル保存先: logs/signals.csv / logs/trades.csv（Streamlit Cloudではデプロイで消える場合があるため、外部Sinkを併用推奨）")

# =========================
# Operator-friendly labels
# =========================
STATE_LABELS_JA = {
    "trend_up": "上昇トレンド優勢",
    "trend_down": "下降トレンド優勢",
    "range": "レンジ（往復）優勢",
    "risk_off": "リスクオフ（荒れ/急変）",
}

def _state_label_full(key: str) -> str:
    k = str(key or "")
    ja = STATE_LABELS_JA.get(k, k)
    return f"{ja} ({k})" if k and ja != k else ja

def _bucket_01(v: float) -> str:
    """
    0-1 のリスク値を「低/中/高」に丸める（表示用）。
    ※見た目の赤/黄/緑は“時間軸”と“運用スタイル”で少し動かす（見送りを強制しない）。
    """
    try:
        x = float(v)
    except Exception:
        return "—"
    if x != x:
        return "—"

    # 時間軸（horizon_days）でしきい値を調整
    try:
        hd = int(globals().get("horizon_days", 3))
    except Exception:
        hd = 3

    # base thresholds
    if hd <= 1:         # スキャ
        t1, t2 = 0.30, 0.55
    elif hd <= 4:       # デイトレ
        t1, t2 = 0.33, 0.66
    else:               # スイング
        t1, t2 = 0.40, 0.75

    # スタイル補正（表示だけ）
    style = str(globals().get("style_name", "標準") or "標準")
    if style == "保守":
        t1 -= 0.05
        t2 -= 0.05
    elif style == "攻撃":
        t1 += 0.05
        t2 += 0.05

    t1 = max(0.05, min(0.90, t1))
    t2 = max(t1 + 0.05, min(0.95, t2))

    if x < t1:
        return "低（平常）"
    if x < t2:
        return "中（警戒）"
    return "高（危険）"



def _jp_decision(decision: str) -> str:
    """Decision label for operators (JP)."""
    d = str(decision or "").upper()
    mapping = {
        "TRADE": "エントリー可",
        "NO_TRADE": "見送り",
        "HOLD": "様子見",
        "WAIT": "待機",
        "PAUSE": "一時停止",
    }
    return mapping.get(d, d or "—")


def _lot_multiplier(global_risk_index: Any, alpha: Any, floor: float = 0.2, ceil: float = 1.0) -> float:
    """Recommended lot multiplier (display-only). Safe (never raises NameError)."""
    try:
        r = float(global_risk_index)
    except Exception:
        return 1.0
    try:
        a = float(alpha)
    except Exception:
        a = 0.0

    # sanitize
    if r != r or r is None:
        return 1.0
    if a != a or a is None:
        a = 0.0
    r = max(0.0, min(1.0, r))
    a = max(0.0, min(2.0, a))

    x = 1.0 - a * r
    if x < floor:
        x = floor
    if x > ceil:
        x = ceil
    return float(x)

def _fmt_price(x: Any, decimals: int = 3) -> str:
    """Format price for UI: max `decimals` places, trim trailing zeros. Internal calc stays full precision."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s


def _action_hint(global_risk: float, war: float, fin: float, macro: float, bs_flag: bool, gov_enabled: bool) -> str:
    """
    運用者向けの“日本語の次の行動”だけを返す（強制停止はしない）。
    しきい値は時間軸（horizon_days）とスタイル（style_name）で少し動かす。
    """
    if bs_flag or (not gov_enabled):
        return "🛑 新規エントリー停止（強制ガード発動）"

    g = float(global_risk or 0.0)
    w = float(war or 0.0)
    f = float(fin or 0.0)
    m = float(macro or 0.0)

    # ベース：デイトレ想定
    hi_g, hi_w, hi_f, hi_m = 0.80, 0.60, 0.80, 0.80
    mid_g, mid_w, mid_f, mid_m = 0.55, 0.35, 0.55, 0.55

    # 時間軸補正（スキャは敏感、スイングは鈍感）
    try:
        hd = int(globals().get("horizon_days", 3))
    except Exception:
        hd = 3

    if hd <= 1:  # スキャ
        hi_g, hi_w, hi_f, hi_m = 0.70, 0.55, 0.70, 0.70
        mid_g, mid_w, mid_f, mid_m = 0.45, 0.30, 0.45, 0.45
    elif hd >= 7:  # スイング
        hi_g, hi_w, hi_f, hi_m = 0.85, 0.65, 0.85, 0.85
        mid_g, mid_w, mid_f, mid_m = 0.60, 0.40, 0.60, 0.60

    # スタイル補正（保守は厳しめ、攻撃は緩め）
    style = str(globals().get("style_name", "標準") or "標準")
    delta = -0.05 if style == "保守" else (0.05 if style == "攻撃" else 0.0)
    hi_g, hi_f, hi_m = hi_g + delta, hi_f + delta, hi_m + delta
    mid_g, mid_f, mid_m = mid_g + delta, mid_f + delta, mid_m + delta
    # war は過敏になりやすいので控えめに
    hi_w = hi_w + (delta * 0.5)
    mid_w = mid_w + (delta * 0.5)

    if (g >= hi_g) or (w >= hi_w) or (f >= hi_f) or (m >= hi_m):
        return "🔴 高リスク：見送り推奨（入るならロット最小・短期・監視必須）"
    if (g >= mid_g) or (w >= mid_w) or (f >= mid_f) or (m >= mid_m):
        return "🟡 警戒：ロット縮小/回数制限（見送り増は正常）"
    return "🟢 平常：通常運用（ただし重要指標/要人発言/週末は別途警戒）"


def _normalize_pair_label(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) == 6:
        s = s[:3] + "/" + s[3:]
    return s

def _load_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return os.getenv(name, default) or default

def _pair_label_to_symbol(pair_label: str) -> str:
    pl = _normalize_pair_label(pair_label)
    mp = getattr(logic, "PAIR_MAP", None)
    if isinstance(mp, dict) and pl in mp:
        return mp[pl]
    fallback = {
        "USD/JPY": "JPY=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "AUD/USD": "AUDUSD=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "AUD/JPY": "AUDJPY=X",
    }
    return fallback.get(pl, "JPY=X")

def _pair_label_to_stooq_symbol(pair_label: str) -> Optional[str]:
    pl = _normalize_pair_label(pair_label)
    mapping = {
        "USD/JPY": "usdjpy",
        "EUR/USD": "eurusd",
        "GBP/USD": "gbpusd",
        "AUD/USD": "audusd",
        "EUR/JPY": "eurjpy",
        "GBP/JPY": "gbpjpy",
        "AUD/JPY": "audjpy",
    }
    return mapping.get(pl)

def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    need = ["Open", "High", "Low", "Close"]
    for c in need:
        if c not in d.columns:
            return pd.DataFrame()
    d = d[need].dropna()
    return d

def _fetch_from_stooq(pair_label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source": "stooq", "ok": False, "error": None, "interval_used": "1d"}
    sym = _pair_label_to_stooq_symbol(pair_label)
    if not sym:
        meta["error"] = "unsupported_pair_for_stooq"
        return pd.DataFrame(), meta
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        d = pd.read_csv(url)
        if "Date" not in d.columns:
            meta["error"] = "bad_csv"
            return pd.DataFrame(), meta
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").sort_index()
        d = _coerce_ohlc(d)
        if d.empty:
            meta["error"] = "empty_after_parse"
            return pd.DataFrame(), meta
        meta["ok"] = True
        return d, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 60)  # 1 hour
def fetch_price_history(pair_label: str, symbol: str, period: str, interval: str, prefer_stooq: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust price fetch:
      - prefer stooq for daily (reduces yfinance rate-limit on Streamlit Cloud)
      - else try yfinance, then fallback to stooq daily
    """
    meta: Dict[str, Any] = {"source": "yfinance", "ok": False, "error": None, "fallback": None, "interval_used": interval}

    # Prefer stooq for daily / multi-scan
    if prefer_stooq or interval == "1d":
        df_s, m_s = _fetch_from_stooq(pair_label)
        if not df_s.empty and m_s.get("ok"):
            meta.update({"source": "stooq", "ok": True, "fallback": None, "interval_used": "1d"})
            return df_s, meta
        meta["fallback"] = m_s

    if yf is not None:
        last_err = None
        for attempt in range(2):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                df = _coerce_ohlc(df)
                if df.empty:
                    last_err = "empty_df"
                    raise RuntimeError("empty_df")
                meta["ok"] = True
                return df, meta
            except YFRateLimitError:
                last_err = "YFRateLimitError"
                break
            except Exception as e:
                last_err = f"{type(e).__name__}:{e}"
                time.sleep(0.6 * (attempt + 1))
        meta["error"] = last_err
    else:
        meta["error"] = "yfinance_not_installed"

    # fallback stooq
    df2, m2 = _fetch_from_stooq(pair_label)
    meta["fallback"] = m2
    if not df2.empty and m2.get("ok"):
        meta["source"] = "stooq"
        meta["ok"] = True
        meta["interval_used"] = "1d"
        return df2, meta

    return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 20)
def fetch_external(pair_label: str, keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Integrated external features. Never crashes.
    """
    base = {
        "news_sentiment": 0.0,
        "cpi_surprise": 0.0,
        "nfp_surprise": 0.0,
        "rate_diff_change": 0.0,
        "macro_risk_score": 0.0,
        "global_risk_index": 0.0,
        "war_probability": 0.0,
        "financial_stress": 0.0,
        "gdelt_war_count_1d": 0.0,
        "gdelt_finance_count_1d": 0.0,
        "vix": float("nan"),
        "dxy": float("nan"),
        "us10y": float("nan"),
        "jp10y": float("nan"),
        "av_inflation": float("nan"),
        "av_unemployment": float("nan"),
        "av_fed_funds_rate": float("nan"),
        "av_treasury_10y": float("nan"),
        "av_macro_risk": 0.0,
    }
    if data_layer is None:
        return base, {"ok": False, "error": "data_layer_import_failed"}
    if not hasattr(data_layer, "fetch_external_features"):
        return base, {"ok": False, "error": "data_layer_missing_fetch_external_features", "file": getattr(data_layer, "__file__", "unknown")}
    try:
        feats, meta = data_layer.fetch_external_features(pair_label, keys=keys)  # type: ignore
        out = base.copy()
        out.update({k: float(v) for k, v in (feats or {}).items() if k in out and v is not None})
        return out, meta if isinstance(meta, dict) else {"ok": True}
    except Exception as e:
        return base, {"ok": False, "error": f"fetch_external_failed:{type(e).__name__}", "detail": str(e)}


def _parts_status_table(meta: Dict[str, Any]) -> pd.DataFrame:
    parts = (meta or {}).get("parts", {}) if isinstance(meta, dict) else {}
    rows: List[Dict[str, Any]] = []

    def _summarize_detail(detail: Any) -> Tuple[Optional[bool], str, Optional[str]]:
        """Return (ok_override, detail_str, err_override)"""
        if isinstance(detail, dict) and any(isinstance(v, dict) and ('ok' in v) for v in detail.values()):
            nested_ok_bits = []
            nested_errs = []
            all_ok = True
            any_ok_field = False
            for k, v in detail.items():
                if not isinstance(v, dict):
                    continue
                vok = v.get("ok", None)
                if vok is not None:
                    any_ok_field = True
                    all_ok = all_ok and bool(vok)
                nested_ok_bits.append(f"{k}:{'ok' if vok else 'ng'}")
                if v.get("error"):
                    nested_errs.append(f"{k}:{v.get('error')}")
            ok_override = all_ok if any_ok_field else None
            d = ", ".join(nested_ok_bits)[:120]
            e = "; ".join(nested_errs)[:160] if nested_errs else None
            return ok_override, d, e

        if isinstance(detail, dict):
            # compact key:val view
            bits = []
            for k, v in list(detail.items())[:12]:
                if isinstance(v, dict):
                    # keys row etc.
                    if "present" in v:
                        mark = "✓" if v.get("present") else "×"
                        used = v.get("used", None)
                        if used:
                            u = str(used)
                            tag = {"keys": "ui", "secrets": "sec", "env": "env"}.get(u, u)
                            bits.append(f"{k}:{mark}({tag})")
                        else:
                            bits.append(f"{k}:{mark}")
                    elif "ok" in v:
                        bits.append(f"{k}:{'ok' if v.get('ok') else 'ng'}")
                    else:
                        bits.append(f"{k}:…")
                else:
                    bits.append(f"{k}:{v}")
            return None, ", ".join(bits)[:120], None

        if isinstance(detail, str):
            return None, detail[:120], None

        return None, "", None

    if isinstance(parts, dict):
        for name, p in parts.items():
            ok: Optional[bool] = None
            err: Optional[str] = None
            extra = ""

            if isinstance(p, dict):
                ok = p.get("ok")
                err = p.get("error")

                # Prefer p['detail'] for summary
                detail = p.get("detail", None)

                ok2, extra2, err2 = _summarize_detail(detail)
                if extra2:
                    extra = extra2
                # If nested says ng but ok was True, override to False (conservative)
                if ok2 is not None:
                    ok = ok2 if ok is None or ok is True else ok
                if err2 and not err:
                    err = err2

                # Fallback: summarize nested dicts directly in p if detail empty
                if not extra:
                    nested = {k: v for k, v in p.items() if isinstance(v, dict)}
                    ok3, extra3, err3 = _summarize_detail(nested)
                    if extra3:
                        extra = extra3
                    if ok3 is not None:
                        ok = ok3 if ok is None or ok is True else ok
                    if err3 and not err:
                        err = err3

                n = p.get("n", None)
                if n is not None:
                    extra = (extra + f" n={n}").strip()
            rows.append({"source": name, "ok": ok, "error": err, "detail": extra})

    if not rows:
        rows = [{"source": "external", "ok": False, "error": "no_meta_parts", "detail": ""}]
    return pd.DataFrame(rows)





def _recommend_min_expected_R_from_audit(rows: List[Dict[str, Any]], target_trade_rate: float = 0.20) -> Dict[str, Any]:
    """Recommend base min_expected_R using ev_audit rows.

    dynamic_threshold = base_threshold * mult
    mult = 1 + 0.8*macro + 1.0*global + 0.6*war + 0.6*fin

    For EV_RAW gate, TRADE if:
      ev_raw >= base_threshold * mult
    => base_threshold <= ev_raw / mult

    To target trade rate r, choose base_threshold as the (1-r) quantile
    of (ev_raw/mult) over recent rows.
    """
    try:
        r = float(target_trade_rate)
        r = max(0.01, min(0.80, r))
    except Exception:
        r = 0.20

    vals: List[float] = []
    for row in rows:
        try:
            ev_raw = float(row.get("ev_raw", ""))
            macro = float(row.get("macro_risk_score", 0.0) or 0.0)
            global_risk = float(row.get("global_risk_index", 0.0) or 0.0)
            war = float(row.get("war_probability", 0.0) or 0.0)
            fin = float(row.get("financial_stress", 0.0) or 0.0)
            mult = 1.0 + 0.8*macro + 1.0*global_risk + 0.6*war + 0.6*fin
            mult = max(1.0, float(mult))
            vals.append(ev_raw / mult)
        except Exception:
            continue

    if len(vals) < 5:
        return {"ok": False, "n": len(vals), "recommended": None, "reason": "ログ件数が不足（5件以上で暫定推奨）"}

    vals_sorted = sorted(vals)
    q = 1.0 - r
    k = int(round(q * (len(vals_sorted) - 1)))
    k = max(0, min(len(vals_sorted) - 1, k))
    rec = float(vals_sorted[k])

    # Guard rails for swing
    rec = float(max(0.01, min(0.12, rec)))

    return {"ok": True, "n": len(vals), "recommended": rec, "target_trade_rate": r}

def _style_defaults(style_name: str) -> Dict[str, Any]:
    # Presets: avoid manual tuning
    if style_name == "保守":
        return {"min_expected_R": 0.12, "horizon_days": 7}
    if style_name == "攻撃":
        return {"min_expected_R": 0.03, "horizon_days": 5}
    return {"min_expected_R": 0.07, "horizon_days": 7}  # 標準

def _build_ctx(pair_label: str, df: pd.DataFrame, feats: Dict[str, float], horizon_days: int, min_expected_R: float, style_name: str,
               governor_cfg: Dict[str, Any]) -> Dict[str, Any]:
    indicators = logic.compute_indicators(df)
    ctx: Dict[str, Any] = {}
    ctx.update(indicators)
    ctx.update(feats)
    ctx["pair_label"] = pair_label
    ctx["pair_symbol"] = _pair_label_to_symbol(pair_label)
    ctx["price"] = float(df["Close"].iloc[-1])
    ctx["horizon_days"] = int(horizon_days)
    ctx["min_expected_R"] = float(min_expected_R)
    ctx["style_name"] = style_name
    # Capital Governor inputs (user provided / optional)
    ctx.update(governor_cfg)
    return ctx

def _dominant_state(state_probs: Dict[str, Any]) -> str:
    if not isinstance(state_probs, dict) or not state_probs:
        return "—"
    try:
        return max(state_probs.items(), key=lambda kv: float(kv[1]))[0]
    except Exception:
        return "—"


def _render_top_trade_panel(pair_label: str, plan: Dict[str, Any], current_price: float):
    """
    運用者が「いま実行すべきか」を迷わないための最上段パネル。
    - NO_TRADE は「エントリー禁止」ではなく「この条件では期待値が薄い/危険寄りなので見送り推奨」
    - ロット係数は“推奨値（表示）”。実際の発注に反映するかは運用者が決める。
    """
    decision = str(plan.get("decision", "NO_TRADE"))
    decision_jp = _jp_decision(decision)
    expected_R_ev = float(plan.get("expected_R_ev") or 0.0)
    p_win_ev = float(plan.get("p_win_ev") or 0.0)
    confidence = float(plan.get("confidence") or 0.0)
    dyn_th = float(plan.get("dynamic_threshold") or 0.0)
    lot_mult = float(plan.get("_lot_multiplier_reco") or 1.0)

    # override info (manual kill switch / outage)
    orig = plan.get("_decision_original")
    ovr = plan.get("_decision_override_reason")

    # モバイル前提：6列だと潰れるため、2列×3段で表示
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("ペア", pair_label)
    r1c2.metric("最終判断", decision_jp)

    r2c1, r2c2 = st.columns(2)
    r2c1.metric("期待値EV (R)", f"{expected_R_ev:+.3f}")
    r2c2.metric("動的閾値", f"{dyn_th:.3f}")

    # 判定に使ったEVが「生EV」か「リスク調整後EV」かを明示（検証用）
    ev_raw = plan.get("expected_R_ev_raw", None)
    ev_adj = plan.get("expected_R_ev_adj", None)
    gate_mode = plan.get("gate_mode", None)
    if ev_raw is not None and ev_adj is not None and gate_mode:
        st.caption(f"判定モード: {gate_mode} / EV(生)={ev_raw:+.3f} / EV(調整後)={ev_adj:+.3f}")

    r3c1, r3c2 = st.columns(2)
    r3c1.metric("信頼度", f"{confidence:.2f}")
    r3c2.metric("推奨ロット係数", f"{lot_mult:.2f}")

    if orig is not None and ovr:
        st.warning(f"判断は上書きされています：{_jp_decision(str(orig))} → {decision_jp}（理由：{ovr}）")

    st.caption(
        "EV (R) は『損切り幅=1R』基準の期待値です。"
        "動的閾値は危険時に上がります（見送りが増えるのは仕様）。"
        "推奨ロット係数は“連続補正”で、急に半減などはしません。"
    )

    if decision != "NO_TRADE":
        side = plan.get("side", "—")
        order_type = plan.get("order_type", "—")
        entry_type = plan.get("entry_type", "") or "—"
        entry = plan.get("entry", None)
        sl = plan.get("stop_loss", None)
        tp = plan.get("take_profit", None)

        st.success("✅ エントリー候補（このアプリは発注しません。発注は運用者が実行）")
        st.markdown(f"""
- **売買**: {side} / **注文**: {order_type} / **種別**: {entry_type}
- **Entry**: {_fmt_price(entry)}
- **SL**: {_fmt_price(sl)}
- **TP**: {_fmt_price(tp)}
""")
        st.caption(f"参考：勝率推定 p_win={p_win_ev:.2f}（あくまでモデル推定）。")
    else:
        st.warning("⏸ 見送り（NO_TRADE）")

        # 表示は「主因」を1行に統一（重複を避ける）
        why = str(plan.get("why", "") or "").strip()
        veto = plan.get("veto_reasons", None)

        primary = ""
        if why:
            primary = why
        elif isinstance(veto, (list, tuple)) and len(veto) > 0:
            primary = str(veto[0])
        else:
            primary = "—"

        st.markdown(f"**主因**: {primary}")
        st.caption(f"判定値：EV {expected_R_ev:+.3f}  <  動的閾値 {dyn_th:.3f}")

        with st.expander("詳細（見送り理由）", expanded=False):
            if why:
                st.markdown(f"- **理由（why）**: {why}")
            if isinstance(veto, (list, tuple)) and len(veto) > 0:
                st.markdown("- **veto内訳**:")
                st.write(list(veto))



def _render_risk_dashboard(plan: Dict[str, Any], feats: Dict[str, float], ext_meta: Optional[Dict[str, Any]] = None):
    """
    運用者が「危険度」と「データ品質」を見て、実行/見送り/ロット縮小を判断できるパネル。
    数字だけで終わらず、日本語の“意味”と“次の行動”をセットで出す。
    さらに、外部APIの取得状態（401/403/429/timeout等）を同じ場所で確認できるようにする。
    """
    bs = plan.get("black_swan", {}) or {}
    gov = plan.get("governor", {}) or {}
    overlay = plan.get("overlay_meta", {}) or {}
    ext_meta = ext_meta or {}

    global_risk = float(feats.get("global_risk_index", 0.0) or 0.0)
    war = float(feats.get("war_probability", 0.0) or 0.0)
    fin = float(feats.get("financial_stress", 0.0) or 0.0)
    macro = float(feats.get("macro_risk_score", 0.0) or 0.0)
    news = float(feats.get("news_sentiment", 0.0) or 0.0)

    bs_flag = bool(bs.get("flag", False))
    bs_level = str(bs.get("level", "") or "")
    gov_enabled = bool(gov.get("enabled", True))

    # data quality
    q_level = ""
    q_reasons: List[str] = []
    try:
        parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
        q = parts.get("quality", {}) if isinstance(parts, dict) else {}
        qd = (q.get("detail", {}) or {}) if isinstance(q, dict) else {}
        q_level = str(qd.get("level", "") or "")
        q_reasons = [str(x) for x in (qd.get("reasons", []) or [])]
    except Exception:
        q_level = ""

    st.markdown("### リスク/ガード（運用判断）")

    # Quality banner
    if q_level == "OUTAGE":
        st.error("🚨 外部データ品質：OUTAGE（主要ソースが取れていない可能性）")
        if q_reasons:
            st.caption("理由: " + " / ".join(q_reasons[:6]))
    elif q_level == "DEGRADED":
        st.warning("⚠️ 外部データ品質：DEGRADED（一部ソース欠け）")
        if q_reasons:
            st.caption("理由: " + " / ".join(q_reasons[:6]))
    else:
        st.success("✅ 外部データ品質：OK（主要ソースが揃っています）")

    # Main risk meters
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("相場全体リスク", f"{global_risk:.2f}", help="0〜1。高いほど荒れやすい")
    c2.metric("地政学リスク", f"{war:.2f}", help="0〜1。戦争/紛争の悪化確率（推定）")
    c3.metric("金融ストレス", f"{fin:.2f}", help="0〜1。信用/金融不安の強さ（推定）")
    c4.metric("マクロ不安", f"{macro:.2f}", help="0〜1。VIX/DXY/金利等からの合成")
    c5.metric("ニュースムード", f"{news:.2f}", help="0〜1。高いほどネガ寄り（定義は実装依存）")

    st.caption(
        f"判定：相場全体={_bucket_01(global_risk)} / 地政学={_bucket_01(war)} / 金融={_bucket_01(fin)} / マクロ={_bucket_01(macro)}"
    )

    # Black swan / governor
    if bs_flag:
        st.error(f"🟥 Black Swan Guard: ON（{bs_level}）")
        rs = bs.get("reasons", []) or []
        if rs:
            st.write(rs)
    else:
        st.info("🟩 Black Swan Guard: OFF（通常）")

    if not gov_enabled:
        st.error("🛑 Capital Governor: 停止（DD/損失/連敗条件に抵触）")
        rs = gov.get("reasons", []) or []
        if rs:
            st.write(rs)
    else:
        st.info("✅ Capital Governor: OK（停止条件に非該当）")

    # Next action hint
    hint = _action_hint(global_risk, war, fin, macro, bs_flag, gov_enabled)
    st.markdown(f"#### 次のアクション（提案）\n- {hint}")
    st.caption(f"ガード設定: {str(guard_apply)} / 推奨ロット係数は最上段に表示。")

    # Overlay notes (debug-level)
    if isinstance(overlay, dict) and overlay:
        adj = overlay.get("risk_adjustment", {})
        if isinstance(adj, dict) and adj:
            st.caption(
                f"（内部補正）risk_adjustment: global={adj.get('global_risk')} war={adj.get('war')} fin={adj.get('fin')} macro={adj.get('macro')}"
            )

    # ---- status table (root-cause) ----
    try:
        show_diag_default = (str(q_level) in ("OUTAGE","DEGRADED"))
        with st.expander("🧪 外部データ取得ステータス（診断・原因究明）", expanded=show_diag_default or bool(show_debug)):
            st.markdown("#### 外部データ取得ステータス（0固定/異常が出たら最優先でここ）")
            st.caption("OKでも中身が空/一部失敗があり得ます。error / detail を必ず見ます。")
            df = _parts_status_table(ext_meta)

            # Runtime fingerprint: proof of what is actually running
            runtime_line = ""
            try:
                dl_file = getattr(data_layer, "__file__", "IMPORT_FAILED")
                dl_build = getattr(data_layer, "DATA_LAYER_BUILD", "?")
                sha12 = "unknown"
                try:
                    import hashlib as _hashlib
                    with open(dl_file, "rb") as _f:
                        sha12 = _hashlib.sha256(_f.read()).hexdigest()[:12]
                except Exception:
                    pass
                runtime_line = f"main={APP_BUILD}, data_layer={dl_build}, sha12={sha12}, file={dl_file}"
                rows = [{"source": "runtime", "ok": True, "error": None, "detail": runtime_line}]
                df = pd.concat([pd.DataFrame(rows), df], ignore_index=True)
            except Exception:
                pass

            if runtime_line:
                st.text_area("実行中コード指紋（コピー用）", value=runtime_line, height=70)

            meaning = {
                "runtime": "実行中のコード指紋（build/sha）",
                "keys": "キー検出状況（secrets/ui/env）",
                "fred": "VIX/DXY/金利（マクロ系）",
                "te": "経済指標カレンダー（CPI/NFP等）",
                "gdelt": "紛争/金融ニュース量（無料）",
                "newsapi": "記事見出しセンチメント",
                "openai": "LLMによる地政学/危機推定（JSON）",
                "alpha_vantage": "マクロ補助（Alpha Vantage）",
                "risk_values": "リスク値（最終：運用判断の基準）",
                "quality": "外部データ品質（OK/DEGRADED/OUTAGE）",
                "build": "data_layer の build 文字列",
            }
            if "source" in df.columns:
                df["意味"] = df["source"].map(meaning).fillna("")

            st.dataframe(df, use_container_width=True)

            try:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 ステータスCSVをダウンロード（検証用）",
                    data=csv,
                    file_name="risk_status_export.csv",
                    mime="text/csv",
                )
            except Exception:
                pass


        with st.expander("📒 EV監査ログ（直近2週間：rawは良いのにadjで潰れる/閾値で潰れるを可視化）", expanded=False):
            rows = _ev_audit_load()

            # --- 自動最適化（min_expected_R 推奨） ---
            target_rate = 0.10 if st.session_state.get('priority_mode') == '勝率優先（見送り増）' else 0.20
            rec = _recommend_min_expected_R_from_audit(rows, target_trade_rate=target_rate) if rows else {"ok": False, "n": 0, "recommended": None, "reason": "ログなし"}
            colA, colB = st.columns([2, 1])
            with colA:
                if rec.get("ok"):
                    st.success(f"自動推奨 min_expected_R（目標TRADE率 {int(rec['target_trade_rate']*100)}%）: **{rec['recommended']:.3f}R**（n={rec['n']}）")
                else:
                    st.info(f"自動推奨 min_expected_R: まだ算出できません（{rec.get('reason','')} / n={rec.get('n',0)}）")
            with colB:
                if rec.get("ok"):
                    if st.button("この推奨値を適用", use_container_width=True):
                        st.session_state["min_expected_R_override"] = float(rec["recommended"])
                        st.toast(f"min_expected_R を {rec['recommended']:.3f} に固定しました（自動最適化）")
                if st.button("自動最適化を解除", use_container_width=True):
                    st.session_state["min_expected_R_override"] = None
                    st.toast("min_expected_R の自動最適化を解除しました（プリセットに戻ります）")
            st.caption("※自動最適化は『直近ログのEV_raw分布』から、目標TRADE率になるよう min_expected_R を推奨します。外部リスクでEVを削らず、閾値だけ調整します。")
            # --- /自動最適化 ---
            summ = _ev_audit_summary(rows, days=14)
            st.write(f"直近{summ['days']}日：判定 {summ['total']}件 / TRADE {summ['trade']}件 / NO_TRADE {summ['no_trade']}件")
            st.write(f"直近{summ['days']}日：**『EV_rawならTRADEだが EV_adjならNO_TRADE』= {summ['killed_by_adj']}件**（二重ブレーキ確認用）")

            # ---- Optimization hint (data-driven) ----
            if rows and summ["total"] >= 20:
                try:
                    import pandas as _pd
                    df_opt = _pd.DataFrame(rows)
                    # coerce
                    for c in ["ev_raw","ev_adj","dynamic_threshold","macro_risk_score","confidence"]:
                        if c in df_opt.columns:
                            df_opt[c] = _pd.to_numeric(df_opt[c], errors="coerce")
                    # focus on recent 14d already counted by summary: we re-filter by ts_utc
                    df_opt = df_opt.dropna(subset=["ev_raw","dynamic_threshold"])
                    # eligible: confident trend states (avoid pure risk_off)
                    if "dominant_state" in df_opt.columns:
                        df_opt = df_opt[df_opt["dominant_state"].astype(str).isin(["trend_up","trend_down","range","risk_off"])]

                    # Estimate base_threshold used by logic (swing-normal assumes: dyn = base*(1+0.20*macro))
                    if "macro_risk_score" in df_opt.columns and df_opt["macro_risk_score"].notna().any():
                        macro = df_opt["macro_risk_score"].fillna(0.0).clip(lower=0.0)
                        base_est = (df_opt["dynamic_threshold"] / (1.0 + 0.20*macro)).clip(lower=0.0)
                    else:
                        base_est = df_opt["dynamic_threshold"]

                    base_est_med = float(base_est.dropna().median()) if base_est.notna().any() else None

                    st.markdown("#### 🎯 しきい値（min_expected_R）の最適化提案（直近ログから推定）")
                    if base_est_med is not None:
                        st.caption(f"現在の推定 base_threshold（中央値）: {base_est_med:.3f}")

                    # Target: at least 4 TRADE / 14d (≈2/週) as a sane starting point
                    target_trades = st.number_input("目標TRADE件数（直近14日）", min_value=0, max_value=500, value=4, step=1)
                    # For each row, trade if base <= ev_raw / (1+0.20*macro)
                    if "macro_risk_score" in df_opt.columns and df_opt["macro_risk_score"].notna().any():
                        macro = df_opt["macro_risk_score"].fillna(0.0).clip(lower=0.0)
                        ratios = (df_opt["ev_raw"] / (1.0 + 0.20*macro)).replace([_pd.NA, _pd.NaT], _pd.NA)
                    else:
                        ratios = df_opt["ev_raw"]

                    ratios = _pd.to_numeric(ratios, errors="coerce").dropna()
                    ratios = ratios[ratios > 0]  # only meaningful positive EV_raw

                    if len(ratios) >= 10 and target_trades > 0:
                        ratios_sorted = ratios.sort_values(ascending=False).reset_index(drop=True)
                        k = int(min(target_trades, len(ratios_sorted)))
                        reco_base = float(ratios_sorted.iloc[k-1])
                        # simulate
                        sim_trades = int((ratios >= reco_base).sum())
                        st.write(f"推奨 min_expected_R（ベース）: **{reco_base:.3f}** （この値なら直近ログ上の予測TRADE: {sim_trades}件/14日）")
                        st.caption("※これは『過去ログに対するシミュレーション』です。市場が変われば変動します。")
                        if st.button("この推奨値を『min_expected_R』に適用（診断用）"):
                            st.session_state["min_expected_R"] = float(reco_base)
                            st.success("min_expected_R を推奨値に更新しました。画面を再計算してください。")
                    else:
                        st.caption("ログがまだ少ないため、推奨値の推定ができません（目安：10件以上）。")
                except Exception:
                    st.caption("最適化提案の計算に失敗しました（ログ形式の違いの可能性）。")
            # ---- /Optimization hint ----
            if rows:
                st.caption("※直近200件を表示。必要ならCSVをダウンロードして集計してください。")
                # show latest first
                import pandas as _pd
                df_a = _pd.DataFrame(list(reversed(rows))[:200])
                st.dataframe(df_a, use_container_width=True)
                try:
                    csv_a = _pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 EV監査ログCSVをダウンロード（ev_audit.csv）",
                        data=csv_a,
                        file_name="ev_audit.csv",
                        mime="text/csv",
                    )
                except Exception:
                    pass
            else:
                st.info("監査ログはまだありません。判定を数回実行すると蓄積されます（デプロイし直すと消える場合があります）。")

    
    except Exception:
        # status table is best-effort; never break trading UI
        pass

# =========================
# UI
# =========================
st.set_page_config(page_title="FX EV Auto v4 Integrated", layout="centered", initial_sidebar_state="collapsed")
st.title("FX 自動AI判断ツール（EV最大化） v4 Integrated")

with st.sidebar:
    st.header("運用操作（見る順）")
    st.caption("普段は上から順に。『安全/診断/詳細』は折りたたんであります。")

    mode = st.selectbox("モード", ["相場全体から最適ペアを自動抽出（推奨）", "単一ペア最適化（徹底）"], index=0)
    trade_axis = st.selectbox("時間軸（保有期間）", ["スイング（1週〜1ヶ月）", "中長期（1〜3ヶ月）"], index=0)
    style_name = st.selectbox("運用スタイル（見送りライン）", ["標準", "保守", "攻撃"], index=0)
    priority = st.selectbox("優先度", ["バランス（推奨）", "勝率優先（見送り増）"], index=0)
    st.session_state['priority_mode'] = priority

    # 時間軸プリセット（詳細設定で上書き可）
    # ※「ポジショントレード」はUI名ではなく“保有期間”の呼び方です。1ヶ月寄りなら「中長期」や interval=1wk を推奨。
    if "中長期" in trade_axis:
        period = "max"
        interval = "1wk"
        horizon_mode = "月（推奨）"
        horizon_days = 30
    else:  # スイング（1週〜1ヶ月）
        period = "10y"
        interval = "1d"
        horizon_mode = "週〜月（推奨）"
        horizon_days = 14

    preset = _style_defaults(style_name)
    min_expected_R = float(preset["min_expected_R"])

    # Optional override (from EV監査ログの自動最適化)
    if "min_expected_R_override" in st.session_state and st.session_state["min_expected_R_override"] is not None:
        try:
            min_expected_R = float(st.session_state["min_expected_R_override"])
        except Exception:
            pass

    st.caption(
        f"見送りライン（min_expected_R）: {min_expected_R:.3f}R / 想定期間: {horizon_days}日 / 価格: {period}・{interval}"
        + ("（自動最適化適用）" if "min_expected_R_override" in st.session_state and st.session_state["min_expected_R_override"] is not None else "")
    )

    with st.expander("🛡️ 安全/ガード（非常時だけ）", expanded=False):
        outage_policy = st.selectbox("外部データ全滅時の扱い", ["表示のみ（推奨：機会を殺さない）", "強制見送り（安全優先）"], index=0)
        guard_apply = st.selectbox(
            "ガードの反映（UIだけ）",
            ["表示のみ（推奨）", "推奨ロット係数を表示（自分で調整）", "品質OUTAGE時のみ見送り（安全）"],
            index=0,
        )
        lot_risk_alpha = st.slider("推奨ロット係数の強さ（α）", 0.0, 1.0, 0.35, 0.05, help="lot_mult = clamp(1 - α*global_risk_index, 0.2, 1.0)")

        force_no_trade_env = (os.getenv("FORCE_NO_TRADE", "") or "").strip().lower() in ("1","true","yes","on")
        force_no_trade = st.checkbox("🛑 手動緊急停止（最終判断を全てNO_TRADE）", value=force_no_trade_env)

    with st.expander("🔑 APIキー（任意・入れた分だけ強くなる）", expanded=False):
        openai_key = st.text_input("OPENAI_API_KEY（地政学LLM）", value=_load_secret("OPENAI_API_KEY", ""), type="password")
        news_key = st.text_input("NEWSAPI_KEY（記事取得）", value=_load_secret("NEWSAPI_KEY", ""), type="password")
        te_key = st.text_input("TRADING_ECONOMICS_KEY（経済指標）", value=_load_secret("TRADING_ECONOMICS_KEY", ""), type="password")
        fred_key = st.text_input("FRED_API_KEY（金利/VIX/DXY）", value=_load_secret("FRED_API_KEY", ""), type="password")
        av_key = st.text_input("ALPHAVANTAGE_API_KEY（マクロ補助/予備）", value=_load_secret("ALPHAVANTAGE_API_KEY", ""), type="password")
        st.caption("※ChatGPT利用とOpenAI APIは別物です。OpenAIは課金/権限が無いと401になります。")

    with st.expander("Capital Governor（本気運用の安全装置）", expanded=False):
        max_dd = st.slider("最大DD（停止）", 0.05, 0.30, 0.15, 0.01)
        daily_stop = st.slider("日次損失（停止）", 0.01, 0.10, 0.03, 0.01)
        max_streak = st.slider("連敗停止", 2, 12, 5, 1)
        equity_dd = st.number_input("現在DD（運用者入力）", value=0.0, step=0.01, help="0.10=10%DD")
        daily_loss = st.number_input("本日損失率（運用者入力）", value=0.0, step=0.01)
        losing_streak = st.number_input("連敗数（運用者入力）", value=0, step=1)

    with st.expander("🔧 詳細/診断（普段は不要）", expanded=False):
        # プリセットの上書き
        period = st.selectbox("価格期間（上書き）", ["1y", "2y", "5y", "10y"], index=["1y","2y","5y","10y"].index(period))
        interval = st.selectbox("価格間隔（上書き）", ["1d", "1wk", "1h"], index=["1d","1wk","1h"].index(interval))
        show_meta = st.checkbox("取得メタ表示（検証用）", value=False)
        show_debug = st.checkbox("デバッグ表示（検証用）", value=False)
        allow_override = st.checkbox("EV閾値/想定期間を手動上書き", value=False)
        if allow_override:
            min_expected_R = st.slider("min_expected_R", 0.0, 0.30, float(min_expected_R), 0.01)
            horizon_days = st.slider("horizon_days", 1, 30, int(horizon_days), 1)
        pair_custom = st.multiselect("スキャン対象（任意）", PAIR_LIST_DEFAULT, default=PAIR_LIST_DEFAULT)

        if st.button("🔄 キャッシュクリアして再取得"):
            st.cache_data.clear()
            st.rerun()
period = locals().get("period", "10y")
interval = locals().get("interval", "1d")
show_meta = locals().get("show_meta", False)
show_debug = locals().get("show_debug", False)
pair_custom = locals().get("pair_custom", PAIR_LIST_DEFAULT)


guard_apply = locals().get("guard_apply", "表示のみ（推奨）")
lot_risk_alpha = float(locals().get("lot_risk_alpha", 0.35))
force_no_trade = bool(locals().get("force_no_trade", False))

keys = {
    "OPENAI_API_KEY": (locals().get("openai_key","") or "").strip(),
    "NEWSAPI_KEY": (locals().get("news_key","") or "").strip(),
    "TRADING_ECONOMICS_KEY": (locals().get("te_key","") or "").strip(),
    "FRED_API_KEY": (locals().get("fred_key","") or "").strip(),
    "ALPHAVANTAGE_API_KEY": (locals().get("av_key","") or "").strip(),
}

governor_cfg = {
    "max_drawdown_limit": float(locals().get("max_dd", 0.15)),
    "daily_loss_limit": float(locals().get("daily_stop", 0.03)),
    "max_losing_streak": int(locals().get("max_streak", 5)),
    "equity_drawdown": float(locals().get("equity_dd", 0.0)),
    "daily_loss": float(locals().get("daily_loss", 0.0)),
    "losing_streak": int(locals().get("losing_streak", 0)),
}

tabs = st.tabs(["🟢 AUTO判断", "🧪 バックテスト（WFA）", "📊 パフォーマンス", "📘 使い方"])

# =========================
# Tab 1: AUTO
# =========================
with tabs[0]:
    st.subheader("最終判断（ここだけ見ればOK）")

    if "相場全体" in mode:
        pairs = [_normalize_pair_label(p) for p in (pair_custom or PAIR_LIST_DEFAULT)]
        pairs = [p for p in pairs if p]
        if not pairs:
            st.error("スキャン対象が空です。")
            st.stop()

        st.caption("複数ペアを同じロジックで評価し、EV最大のペアを自動選択します（日足はStooq優先で安定化）。")

        rows: List[Dict[str, Any]] = []
        # 外部リスクはグローバル（ペア依存しない）なので、マルチペアでも1回だけ取得
        feats_global, ext_meta_global = fetch_external("GLOBAL", keys=keys)
        for p in pairs:
            sym = _pair_label_to_symbol(p)
            df, price_meta = fetch_price_history(p, sym, period=period, interval=interval, prefer_stooq=(str(interval)=="1d"))
            if df.empty:
                rows.append({"pair": p, "EV": None, "decision": "NO_DATA", "confidence": None, "dom_state": None})
                continue

            feats, ext_meta = feats_global, ext_meta_global
            ctx = _build_ctx(p, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
            plan = logic.get_ai_order_strategy(api_key=keys.get("OPENAI_API_KEY",""), context_data=ctx)


            
            # --- Win-rate focus soft gate (display/decision layer) ---
            # Does not change internal calculations; only tightens final permission when user selects 勝率優先.
            try:
                if str(st.session_state.get('priority_mode','')) == '勝率優先（見送り増）':
                    # 勝率優先 = 「p_winをハード閾値で切り捨て」ではなく、
                    # 低p_winを"強いEVでのみ許可"する階段ゲートにする（ゼロ化を防ぐ）
                    p_win = float(plan.get("p_win_ev", 0.0) or 0.0)
                    conf = float(plan.get("confidence", 0.0) or 0.0)
                    ev_raw = float(plan.get("expected_R_ev_raw", plan.get("expected_R_ev", 0.0)) or 0.0)
                    thr = float(plan.get("dynamic_threshold", 0.0) or 0.0)

                    # ゲート設定（固定・後方互換）
                    conf_min = 0.55
                    hard_pwin_min = 0.50
                    soft_pwin_min = 0.54
                    ev_margin_if_soft = 0.02  # p_winが弱いときは、EVで補強されている場合のみ許可

                    veto = None
                    if conf < conf_min:
                        veto = f"勝率優先フィルタ: conf={conf:.2f} < {conf_min:.2f}"
                    elif p_win < hard_pwin_min:
                        veto = f"勝率優先フィルタ: p_win={p_win:.2f} < {hard_pwin_min:.2f}"
                    elif p_win < soft_pwin_min and ev_raw < (thr + ev_margin_if_soft):
                        veto = f"勝率優先フィルタ: p_win={p_win:.2f}（弱）→ EV補強不足 {ev_raw:+.3f} < {thr + ev_margin_if_soft:.3f}"

                    if str(plan.get("decision","")) == "TRADE" and veto:
                        plan["decision"] = "NO_TRADE"
                        reasons = list(plan.get("veto_reasons", []) or [])
                        reasons.append(veto)
                        plan["veto_reasons"] = reasons
                        plan["why"] = reasons[-1]
            except Exception:
                pass
# ---- EV audit row (for 2-week verification) ----
            try:
                overlay = plan.get("overlay_meta", {}) or {}
                ev_raw = plan.get("expected_R_ev_raw", plan.get("expected_R_ev", 0.0))
                ev_adj = plan.get("expected_R_ev_adj", plan.get("expected_R_ev", 0.0))
                thr = plan.get("dynamic_threshold", 0.0)
                # "killed" means: would TRADE on raw, but would be NO_TRADE on adj (for analysis)
                decision_raw = "TRADE" if float(ev_raw) >= float(thr) else "NO_TRADE"
                decision_adj = "TRADE" if float(ev_adj) >= float(thr) else "NO_TRADE"
                killed_by_adj = (decision_raw == "TRADE" and decision_adj != "TRADE")
                _ev_audit_append({
                    "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "timeframe_mode": str(st.session_state.get("timeframe_mode", "")),
                    "pair": str(p),
                    "decision_adj": str(plan.get("decision","")),
                    "decision_raw": decision_raw,
                    "killed_by_adj": str(bool(killed_by_adj)),
                    "ev_raw": float(ev_raw),
                    "ev_adj": float(ev_adj),
                    "dynamic_threshold": float(thr),
                    "dominant_state": str(plan.get("dominant_state","")),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "global_risk_index": float(overlay.get("global_risk_index", 0.0) or 0.0),
                    "war_probability": float(overlay.get("war_probability", 0.0) or 0.0),
                    "financial_stress": float(overlay.get("financial_stress", 0.0) or 0.0),
                    "macro_risk_score": float(overlay.get("macro_risk_score", 0.0) or 0.0),
                    "risk_off_bump": float(overlay.get("risk_off_bump", 0.0) or 0.0),
                })
            except Exception:
                pass
            # ---- /EV audit row ----
            # ---- operator guard (UI-level; default display-only) ----

            lot_mult = _lot_multiplier(feats.get("global_risk_index", 0.0), lot_risk_alpha)

            decision_override = None

            override_reason = ""

            # 手動緊急停止は最優先

            if force_no_trade:

                decision_override = "NO_TRADE"

                override_reason = "手動緊急停止"

            # 品質OUTAGE時のみ見送り（安全）

            try:

                parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}

                level = str((((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or ""))

            except Exception:

                level = ""

            if decision_override is None and ("品質OUTAGE時のみ見送り" in str(guard_apply)) and level == "OUTAGE":

                decision_override = "NO_TRADE"

                override_reason = "外部データ品質OUTAGE"

            if decision_override is not None:

                plan = dict(plan or {})

                plan["_decision_original"] = plan.get("decision")

                plan["decision"] = decision_override

                plan["_decision_override_reason"] = override_reason

            plan = dict(plan or {})

            plan["_lot_multiplier_reco"] = float(lot_mult)

            plan_ui = plan
            try:
                parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
                level = str(((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or "")
                if "強制見送り" in str(locals().get("outage_policy","")) and level == "OUTAGE":
                    # UI only: do not change logic.py; just stop recommending entries when blind
                    plan_ui = dict(plan or {})
                    plan_ui["decision"] = "NO_TRADE"
                    vr = list(plan_ui.get("veto_reasons") or [])
                    if "DATA_OUTAGE" not in vr:
                        vr.append("DATA_OUTAGE（外部データ全滅）")
                    plan_ui["veto_reasons"] = vr
            except Exception:
                plan_ui = plan

            ev = float(plan.get("expected_R_ev") or 0.0)
            decision = str(plan.get("decision") or "NO_TRADE")
            conf = float(plan.get("confidence") or 0.0)
            dom = _dominant_state(plan.get("state_probs", {}))

            rows.append({
                "pair": p,
                "EV": ev,
                "decision": decision,
                "confidence": conf,
                "dom_state": dom,
                "_plan": plan,
                "_plan_ui": plan_ui,
                "_ctx": ctx,
                "_feats": feats,
                "_price_meta": price_meta,
                "_ext_meta": ext_meta,
            })

        ranked = [r for r in rows if isinstance(r.get("EV"), (int, float))]
        ranked.sort(key=lambda r: float(r["EV"]), reverse=True)
        if not ranked:
            st.error("有効なペアがありません（全てNO_DATA）。")
            st.dataframe(pd.DataFrame(rows)[["pair", "decision"]], use_container_width=True)
            st.stop()

        best = ranked[0]
        plan = best["_plan"]
        plan_ui_best = best.get("_plan_ui", plan)
        feats = best["_feats"]
        price = float(best["_ctx"].get("price", 0.0))

        # Top panel must show entry format + price (user request)
        _render_top_trade_panel(best["pair"], plan_ui_best, price)

        # Risk dashboard (new)
        _render_risk_dashboard(plan_ui_best, feats, ext_meta=best.get("_ext_meta", {}))

        # Logging (optional)
        _render_logging_panel(best["pair"], plan_ui_best, best.get("_ctx", {}), feats, best.get("_price_meta", {}), best.get("_ext_meta", {}))

        st.markdown("### EVランキング（代替案ペアはここ）")
        view = [{
            "pair": r["pair"],
            "EV": float(r["EV"]),
            "decision": r["decision"],
            "confidence": float(r["confidence"]),
            "dominant_state": _state_label_full(r["dom_state"]),
            "global_risk": float(r["_feats"].get("global_risk_index", 0.0)),
            "war": float(r["_feats"].get("war_probability", 0.0)),
        } for r in ranked]
        st.dataframe(pd.DataFrame(view), use_container_width=True)

        st.markdown("### EV内訳（最適ペア）")

        # モバイル前提：棒グラフだけだと「0の棒が見えない」ため、
        # まず表で「確率(%) / 寄与EV(R)」を明示し、図は折りたたみにします。
        ev_contribs = (plan.get("ev_contribs", {}) or {})
        state_probs = (plan.get("state_probs", {}) or {})
        _states = ["trend_up", "trend_down", "range", "risk_off"]

        rows_ev = []
        for st_name in _states:
            c = float((ev_contribs or {}).get(st_name, 0.0) or 0.0)
            p = float((state_probs or {}).get(st_name, 0.0) or 0.0)
            rows_ev.append({
                "状態": _state_label_full(st_name),
                "確率(%)": round(p * 100.0, 1),
                "寄与EV(R)": round(c, 4),
            })

        cdf = pd.DataFrame(rows_ev)
        if not cdf.empty:
            st.dataframe(cdf, use_container_width=True)
            st.caption("※ 棒が見えない＝寄与が0付近です。未計算ではありません。")

            with st.expander("棒グラフ（参考）", expanded=False):
                st.bar_chart(cdf.set_index("状態")[["寄与EV(R)"]])
        else:
            st.info("EV内訳が空です。")

        with st.expander("詳細（最適ペア）", expanded=False):
            st.json({"plan": plan})
            if show_debug:
                st.json({"ctx": best["_ctx"], "feats": feats})
            if show_meta:
                st.json({"price_meta": best.get("_price_meta", {}), "external_meta": best.get("_ext_meta", {})})

    else:
        pair_label = _normalize_pair_label(st.text_input("通貨ペア（単一最適化）", value="USD/JPY"))
        symbol = _pair_label_to_symbol(pair_label)

        df, price_meta = fetch_price_history(pair_label, symbol, period=period, interval=interval, prefer_stooq=(str(interval)=="1d"))
        if df.empty:
            st.error("価格データ取得に失敗しました。")
            st.json(price_meta)
            st.stop()

        feats, ext_meta = fetch_external(pair_label, keys=keys)
        ctx = _build_ctx(pair_label, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
        plan = logic.get_ai_order_strategy(api_key=keys.get("OPENAI_API_KEY",""), context_data=ctx)

        # ---- operator guard (UI-level; default display-only) ----

        lot_mult = _lot_multiplier(feats.get("global_risk_index", 0.0), lot_risk_alpha)

        decision_override = None

        override_reason = ""

        # 手動緊急停止は最優先

        if force_no_trade:

            decision_override = "NO_TRADE"

            override_reason = "手動緊急停止"

        # 品質OUTAGE時のみ見送り（安全）

        try:

            parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}

            level = str((((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or ""))

        except Exception:

            level = ""

        if decision_override is None and ("品質OUTAGE時のみ見送り" in str(guard_apply)) and level == "OUTAGE":

            decision_override = "NO_TRADE"

            override_reason = "外部データ品質OUTAGE"

        if decision_override is not None:

            plan = dict(plan or {})

            plan["_decision_original"] = plan.get("decision")

            plan["decision"] = decision_override

            plan["_decision_override_reason"] = override_reason

        plan = dict(plan or {})

        plan["_lot_multiplier_reco"] = float(lot_mult)

        plan_ui = plan
        try:
            parts = (ext_meta or {}).get("parts", {}) if isinstance(ext_meta, dict) else {}
            level = str(((parts.get("quality", {}) or {}).get("detail", {}) or {}).get("level", "") or "")
            if "強制見送り" in str(locals().get("outage_policy","")) and level == "OUTAGE":
                plan_ui = dict(plan or {})
                plan_ui["decision"] = "NO_TRADE"
                vr = list(plan_ui.get("veto_reasons") or [])
                if "DATA_OUTAGE" not in vr:
                    vr.append("DATA_OUTAGE（外部データ全滅）")
                plan_ui["veto_reasons"] = vr
        except Exception:
            plan_ui = plan

        price = float(ctx.get("price", 0.0))
        _render_top_trade_panel(pair_label, plan_ui, price)
        _render_risk_dashboard(plan_ui, feats, ext_meta=ext_meta)

        _render_logging_panel(pair_label, plan_ui, ctx, feats, price_meta, ext_meta)

        st.markdown("### EV内訳（何がEVを潰しているか）")
        ev_contribs = plan.get("ev_contribs", {}) or {}
        if isinstance(ev_contribs, dict) and ev_contribs:
            cdf = pd.DataFrame([{"state": k, "contrib_R": float(v)} for k, v in ev_contribs.items()]).sort_values("contrib_R")
            cdf["state_label"] = cdf["state"].apply(_state_label_full)
            st.bar_chart(cdf.set_index("state_label")[["contrib_R"]])
        else:
            st.info("EV内訳が空です。")

        with st.expander("詳細", expanded=False):
            st.json(plan.get("state_probs", {}))
            if show_debug:
                st.json({"ctx": ctx, "feats": feats})
            if show_meta:
                st.json({"price_meta": price_meta, "external_meta": ext_meta})

# =========================
# Tab 2: Backtest (keep existing)
# =========================
with tabs[1]:
    st.subheader("ウォークフォワード（WFA）バックテスト")
    st.caption("方向性確認用（コスト・スリップ未反映）。バックテストは“残す”方針。")

    colA, colB, colC = st.columns(3)
    with colA:
        bt_pair = st.selectbox("バックテスト対象ペア", PAIR_LIST_DEFAULT, index=0)
        bt_period = st.selectbox("BT期間", ["5y", "10y"], index=1)
        train_years = st.number_input("train_years", min_value=1, max_value=8, value=3, step=1)
    with colB:
        test_months = st.number_input("test_months", min_value=1, max_value=24, value=6, step=1)
        bt_horizon = st.number_input("horizon_days", min_value=1, max_value=14, value=int(horizon_days), step=1)
    with colC:
        bt_min_ev = st.slider("min_expected_R", 0.0, 0.3, float(min_expected_R), 0.01)

    run = st.button("バックテスト実行", type="primary")
    if run:
        try:
            import backtest_ev_v1
            sym = _pair_label_to_symbol(bt_pair)
            wf_df, summ = backtest_ev_v1.run_backtest(
                pair_symbol=sym,
                period=bt_period,
                horizon_days=int(bt_horizon),
                train_years=int(train_years),
                test_months=int(test_months),
                min_expected_R=float(bt_min_ev),
            )
            st.markdown("### サマリー")
            st.json(summ)
            st.markdown("### WFA結果")
            st.dataframe(wf_df, use_container_width=True)

            csv = wf_df.to_csv(index=False).encode("utf-8")
            st.download_button("CSVダウンロード", data=csv, file_name=f"ev_wfa_{bt_pair.replace('/','_')}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"バックテストでエラー: {type(e).__name__}: {e}")


# =========================
# Tab 3: Performance
# =========================
with tabs[2]:
    st.subheader("パフォーマンス（損益ログから自動集計）")
    st.caption("signals / trades を保存していれば、ここで期待値・勝率・ドローダウンを自動で見えます。")

    df_s = _load_csv_df(SIGNAL_LOG_PATH)
    df_t = _load_csv_df(TRADE_LOG_PATH)

    m = _compute_trade_metrics(df_t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("トレード数", f"{m.get('n',0)}")
    if m.get("n",0) > 0:
        c2.metric("期待値（平均R）", f"{m.get('expectancy_R',0.0):+.3f}")
        c3.metric("勝率", f"{m.get('win_rate',0.0)*100:.1f}%")
        pf = m.get("profit_factor")
        c4.metric("PF", ("∞" if pf == float("inf") else (f"{pf:.2f}" if isinstance(pf,(int,float)) else "—")))
    else:
        c2.metric("期待値（平均R）", "—")
        c3.metric("勝率", "—")
        c4.metric("PF", "—")

    if m.get("n",0) > 0 and "r_multiple" in df_t.columns:
        d = df_t.copy()
        d["r_multiple"] = pd.to_numeric(d["r_multiple"], errors="coerce")
        d = d.dropna(subset=["r_multiple"])
        d["cum_R"] = d["r_multiple"].cumsum()
        st.line_chart(d.set_index(pd.RangeIndex(len(d)))["cum_R"])
        st.caption(f"最大DD（R）: {m.get('max_drawdown_R',0.0):.3f} / 総R: {m.get('sum_R',0.0):+.3f}")

    st.markdown("### 直近のトレード（trades）")
    st.dataframe(df_t.tail(200), use_container_width=True)

    st.markdown("### 直近のシグナル（signals）")
    st.dataframe(df_s.tail(200), use_container_width=True)

    # downloads
    try:
        if not df_t.empty:
            st.download_button("trades.csv をダウンロード", data=df_t.to_csv(index=False).encode("utf-8"),
                               file_name="trades.csv", mime="text/csv")
        if not df_s.empty:
            st.download_button("signals.csv をダウンロード", data=df_s.to_csv(index=False).encode("utf-8"),
                               file_name="signals.csv", mime="text/csv")
    except Exception:
        pass


# =========================
# Tab 3: Guide
# =========================
with tabs[3]:
    st.markdown("""
# 📘 運用者向け・画面の見方（メイン/サイドバー）

このツールは **「期待値（EV）を最大化しつつ、外部リスクで“止める/弱める”」** ための運用パネルです。  
**迷ったら** → AUTO判断タブの **「最終判断」→「リスクダッシュボード」→「外部データ取得ステータス」** の順に見てください。

---

## 1) サイドバー（左）の機能
### モード
- **相場全体から最適ペアを自動抽出（推奨）**：複数ペアを走査し、EVが最大のペアを出します（運用向け）
- **単一ペア最適化（徹底）**：指定ペアだけを深く見る（検証/研究向け）

### 運用スタイル（標準/保守/攻撃）
- **保守**：見送りラインが高く、厳選（資金大きい/イベント多い時）
- **標準**：バランス
- **攻撃**：見送りラインが低く、回転（検証や小さめ資金向け）

### 想定期間（週/日）
- **週（推奨）**：ノイズに強く、判断が安定
- **日**：短期トレード寄り（シグナルは速いがブレやすい）

### APIキー（任意：入れた分だけ“外部リスク”が精密）
- **FRED**：VIX/DXY/金利（マクロ・不安定度）
- **NewsAPI**：記事見出しセンチメント
- **TradingEconomics**：CPI/NFPなど（ただし無料キーは国制限で403になりがち）
- **OpenAI**：LLMが地政学/危機確率を推定（JSON）→ **GlobalRisk/WarProb に反映**

### Capital Governor（本気運用の安全装置）
- 最大DD/日次損失/連敗が閾値を超えると **強制停止**します（運用者が入力）

---

## 2) メインパネル（AUTO判断タブ）の見方
### 最終判断（ここだけ見ればOK）
- **TRADE**：推奨エントリー（Entry/SL/TP）が出ます
- **NO_TRADE**：見送り。理由（veto）が出ます（EV不足/リスク過多/ガバナー停止など）

### 期待値EV (R) / 動的閾値 / 信頼度
- **EV (R)**：1R（＝損切り幅）を基準にした「1回あたりの期待値」  
  例）EV=+0.07 → 1回の取引で **平均 +0.07R** を狙う設計
- **動的閾値**：相場が危険になるほど上がる “見送りライン”  
  → 危険時に見送りが増えるのは **仕様**
- **信頼度**：モデルの確信度（0〜1）。低いほど慎重に。

### 🛡️ リスクダッシュボード（運用の心臓部）
- **総合リスク / 戦争・地政学 / 金融ストレス / マクロ不確実性** を 0〜1 で表示  
  0=平常 / 1=危機。**低/中/高** と **推奨アクション** が併記されます。

### 外部データ取得ステータス（原因究明）
- 0固定や異常値の原因は、ここに **http_401/403/429/timeout** として出ます。
- **keys 行**：キーがどこから読めたか（sec/ui/env）を表示します。

### EV内訳（棒グラフ）
- 相場タイプ（上昇/下降/レンジ/リスクオフ）の **どれがEVを押し上げ/押し下げ**しているかの内訳です。  
  「リスクオフが大きくマイナス」なら、見送りになりやすいのは正常です。

---

## 3) よくあるトラブルと対処
- **OpenAI 401**：APIキー/課金/権限（ChatGPT契約とは別）
- **TradingEconomics 403**：無料キー国制限（仕様寄り）
- **GDELT timeout/429**：ネットワーク到達性 or 間隔制御不足（キャッシュ/リトライで緩和）
""")

def _profit_max_reco(plan: dict) -> dict:
    """Compute an extended take-profit and a simple trail-stop suggestion (advisory)."""
    try:
        side = (plan.get("side") or "").upper()
        entry = float(plan.get("entry") or 0.0)
        sl = float(plan.get("stop_loss") or 0.0)
        tp = float(plan.get("take_profit") or 0.0)
        conf = float(plan.get("confidence") or 0.0)
        dom = str(plan.get("dominant_state") or "")
        overlay = plan.get("overlay_meta") or {}
        gri = float(overlay.get("global_risk_index") or 0.0)
        war = float(overlay.get("war_probability") or 0.0)
        fin = float(overlay.get("financial_stress") or 0.0)

        risk = max(0.0, min(1.0, 0.5*gri + 0.3*war + 0.2*fin))
        extend = 1.0 + 0.9 * max(0.0, conf - 0.5) * (1.0 - risk)
        if "risk_off" in dom:
            extend = 1.0
        extend = max(1.0, min(1.8, extend))

        dist_tp = abs(tp - entry)
        if dist_tp <= 0:
            tp_ext = tp
        else:
            tp_ext = entry + dist_tp * extend if side == "BUY" else (entry - dist_tp * extend if side == "SELL" else tp)

        dist_sl = abs(entry - sl)
        if dist_sl <= 0:
            trail_sl = sl
        else:
            trail_sl = entry + 0.5*dist_sl if side == "BUY" else (entry - 0.5*dist_sl if side == "SELL" else sl)

        return {"tp_ext": tp_ext, "trail_sl": trail_sl, "extend_factor": extend}
    except Exception:
        return {"tp_ext": plan.get("take_profit"), "trail_sl": plan.get("stop_loss"), "extend_factor": 1.0}



def _try_post_json(url: str, payload: dict, timeout_s: int = 6):
    """POST JSON and return (ok: bool, status_code: int|None, response_text: str, error: str)."""
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        return (200 <= getattr(r, "status_code", 0) < 300), getattr(r, "status_code", None), (r.text or ""), ""
    except Exception as e:
        return False, None, "", str(e)

def _post_json_webhook(payload: dict):
    """Optional external sink: POST JSON to LOG_WEBHOOK_URL if provided in Streamlit Secrets.
    Returns (ok, status_code, response_text, error). Never raises."""
    try:
        url = st.secrets.get("LOG_WEBHOOK_URL", "")
        if not url:
            return False, None, "", "LOG_WEBHOOK_URL not set"
        return _try_post_json(url, payload, timeout_s=6)
    except Exception as e:
        return False, None, "", str(e)

def _supabase_insert(table: str, row: dict):
    """Optional Supabase sink (REST): requires SUPABASE_URL and SUPABASE_ANON_KEY in secrets.
    Returns (ok, status_code, response_text, error). Never raises."""
    try:
        sb_url = st.secrets.get("SUPABASE_URL", "")
        sb_key = st.secrets.get("SUPABASE_ANON_KEY", "")
        if not sb_url or not sb_key or not table:
            return False, None, "", "Supabase secrets/table not set"
        endpoint = sb_url.rstrip("/") + f"/rest/v1/{table}"
        headers = {
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        r = requests.post(endpoint, headers=headers, json=[row], timeout=6)
        ok = 200 <= getattr(r, "status_code", 0) < 300
        return ok, getattr(r, "status_code", None), (r.text or ""), ""
    except Exception as e:
        return False, None, "", str(e)

def _external_log_event(kind: str, row: dict):
    """Send an event to external sinks (webhook and/or Supabase) if configured.
    Returns a dict with per-sink results."""
    payload = {"kind": kind, **row}
    results = {"webhook": None, "supabase": None}

    ok, sc, txt, err = _post_json_webhook(payload)
    results["webhook"] = {"ok": ok, "status_code": sc, "response": (txt[:500] if txt else ""), "error": err}

    try:
        table = st.secrets.get("SUPABASE_LOG_TABLE", "")
    except Exception:
        table = ""
    if table:
        ok2, sc2, txt2, err2 = _supabase_insert(table, payload)
        results["supabase"] = {"ok": ok2, "status_code": sc2, "response": (txt2[:500] if txt2 else ""), "error": err2}

    return results



# --- Webhook Diagnostics (fixed27) ---
with st.expander("🔧 Webhook診断（送信テスト/失敗理由の表示）", expanded=False):
    url = ""
    try:
        url = st.secrets.get("LOG_WEBHOOK_URL", "")
    except Exception:
        url = ""

    if url:
        masked = url[:32] + "..." + url[-12:] if len(url) > 48 else url
        st.write(f"LOG_WEBHOOK_URL: `{masked}`")
    else:
        st.warning("LOG_WEBHOOK_URL が Secrets に設定されていません。B1(Webhook) は無効です。")

    test_payload = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pair": st.session_state.get("selected_pair", ""),
        "decision": "TEST",
        "ev_raw": 0.123,
        "ev_adj": 0.045,
        "dynamic_threshold": 0.010,
        "gate_mode": "EV_RAW",
    }

    colA, colB = st.columns(2)
    with colA:
        if st.button("Webhookへテスト送信", use_container_width=True):
            if not url:
                st.error("LOG_WEBHOOK_URL が未設定です。")
            else:
                res = _external_log_event("debug_test", test_payload)
                st.session_state["last_webhook_result"] = res
                wh = (res or {}).get("webhook") or {}
                if wh.get("ok"):
                    st.success(f"Webhook送信OK (HTTP {wh.get('status_code')})")
                else:
                    st.error(f"Webhook送信NG: {wh.get('error') or 'unknown'} (HTTP {wh.get('status_code')})")
                sb = (res or {}).get("supabase")
                if sb is not None:
                    if sb.get("ok"):
                        st.success(f"Supabase INSERT OK (HTTP {sb.get('status_code')})")
                    else:
                        st.error(f"Supabase INSERT NG: {sb.get('error') or 'unknown'} (HTTP {sb.get('status_code')})")
    with colB:
        st.caption("送信payload（確認用）")
        st.json({"kind": "debug_test", **test_payload})

    if "last_webhook_result" in st.session_state:
        st.caption("直近の送信結果（デバッグ）")
        st.json(st.session_state["last_webhook_result"])
# --- /Webhook Diagnostics ---
