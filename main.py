# main.py (v4 integrated, keeps v3 features + adds global risk overlays)
from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple, Optional, List

import streamlit as st
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
    decision = str(plan.get("decision", "NO_TRADE"))
    expected_R_ev = float(plan.get("expected_R_ev") or 0.0)
    p_win_ev = float(plan.get("p_win_ev") or 0.0)
    confidence = float(plan.get("confidence") or 0.0)
    dyn_th = float(plan.get("dynamic_threshold") or 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ペア", pair_label)
    c2.metric("現在値", f"{current_price:.5f}" if current_price else "—")
    c3.metric("期待値EV (R)", f"{expected_R_ev:+.3f}")
    c4.metric("動的閾値", f"{dyn_th:.3f}")
    c5.metric("信頼度", f"{confidence:.2f}")

    if decision != "NO_TRADE":
        side = plan.get("side", "—")
        order_type = plan.get("order_type", "—")
        entry_type = plan.get("entry_type", "") or "—"
        entry = plan.get("entry", None)
        sl = plan.get("stop_loss", None)
        tp = plan.get("take_profit", None)

        st.success("✅ エントリー候補")
        st.markdown(f"""
- **売買**: {side} / **注文**: {order_type} / **種別**: {entry_type}
- **Entry**: {entry if entry is not None else '—'}
- **SL**: {sl if sl is not None else '—'}
- **TP**: {tp if tp is not None else '—'}
""")
    else:
        st.warning("⏸ 見送り（NO_TRADE）")
        why = str(plan.get("why","") or "")
        st.markdown(f"**理由**: {why if why else '—'}")
        veto = plan.get("veto_reasons", None)
        if isinstance(veto, (list, tuple)) and len(veto) > 0:
            st.markdown("**見送り理由（veto）内訳**")
            for r in veto:
                st.write(f"- {r}")

def _render_risk_dashboard(plan: Dict[str, Any], feats: Dict[str, float]):
    bs = plan.get("black_swan", {}) or {}
    gov = plan.get("governor", {}) or {}
    overlay = plan.get("overlay_meta", {}) or {}

    st.markdown("### リスクダッシュボード（最重要）")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GlobalRisk", f"{float(feats.get('global_risk_index',0.0)):.2f}")
    col2.metric("WarProb", f"{float(feats.get('war_probability',0.0)):.2f}")
    col3.metric("FinStress", f"{float(feats.get('financial_stress',0.0)):.2f}")
    col4.metric("MacroRisk", f"{float(feats.get('macro_risk_score',0.0)):.2f}")

    colA, colB, colC, colD = st.columns(4)
    vix = feats.get("vix", float("nan"))
    dxy = feats.get("dxy", float("nan"))
    us10y = feats.get("us10y", float("nan"))
    jp10y = feats.get("jp10y", float("nan"))
    colA.metric("VIX", f"{vix:.1f}" if pd.notna(vix) else "—")
    colB.metric("DXY(代替)", f"{dxy:.1f}" if pd.notna(dxy) else "—")
    colC.metric("US10Y", f"{us10y:.2f}" if pd.notna(us10y) else "—")
    colD.metric("JP10Y", f"{jp10y:.2f}" if pd.notna(jp10y) else "—")

    level = str(bs.get("level", "green"))
    if bs.get("flag"):
        st.error(f"🛑 Black Swan Guard: {level} — 新規エントリー禁止")
    elif level == "yellow":
        st.warning("⚠ リスク上昇（yellow）— 閾値が上がり見送りやすくなります")
    else:
        st.info("✅ リスク通常（green）")

    if isinstance(bs.get("reasons"), list) and bs["reasons"]:
        st.write("**検知理由**")
        for r in bs["reasons"]:
            st.write(f"- {r}")

    if isinstance(gov, dict):
        enabled = bool(gov.get("enabled", True))
        st.write(f"**Capital Governor**: {'ON（取引可）' if enabled else 'OFF（取引停止）'}")
        if not enabled:
            for r in (gov.get("reasons") or []):
                st.write(f"- {r}")

    with st.expander("詳細（overlay/metrics）", expanded=False):
        st.json({"overlay_meta": overlay, "black_swan": bs, "governor": gov})

# =========================
# UI
# =========================
st.set_page_config(page_title="FX EV Auto v4 Integrated", layout="wide")
st.title("FX 自動AI判断ツール（EV最大化） v4 Integrated")

with st.sidebar:
    st.header("AUTO運用（最小設定）")
    mode = st.selectbox("モード", ["相場全体から最適ペアを自動抽出（推奨）", "単一ペア最適化（徹底）"], index=0)
    style_name = st.selectbox("運用スタイル", ["標準", "保守", "攻撃"], index=0)
    horizon_mode = st.selectbox("想定期間", ["週（推奨）", "日"], index=0)

    preset = _style_defaults(style_name)
    horizon_days = 7 if "週" in horizon_mode else 3
    min_expected_R = float(preset["min_expected_R"])

    st.divider()
    with st.expander("APIキー（任意・入れた分だけ強くなる）", expanded=False):
        gemini_key = st.text_input("GEMINI_API_KEY（地政学LLM・任意）", value=_load_secret("GEMINI_API_KEY", ""), type="password")
        news_key = st.text_input("NEWSAPI_KEY（記事取得・任意）", value=_load_secret("NEWSAPI_KEY", ""), type="password")
        te_key = st.text_input("TRADING_ECONOMICS_KEY（経済指標・任意）", value=_load_secret("TRADING_ECONOMICS_KEY", ""), type="password")
        fred_key = st.text_input("FRED_API_KEY（金利/VIX/DXY・任意）", value=_load_secret("FRED_API_KEY", ""), type="password")

    with st.expander("Capital Governor（本気運用用）", expanded=False):
        max_dd = st.slider("最大DD（停止）", 0.05, 0.30, 0.15, 0.01)
        daily_stop = st.slider("日次損失（停止）", 0.01, 0.10, 0.03, 0.01)
        max_streak = st.slider("連敗停止", 2, 12, 5, 1)
        equity_dd = st.number_input("現在DD（運用者入力）", value=0.0, step=0.01, help="0.10=10%DD")
        daily_loss = st.number_input("本日損失率（運用者入力）", value=0.0, step=0.01)
        losing_streak = st.number_input("連敗数（運用者入力）", value=0, step=1)

    st.divider()
    with st.expander("詳細設定（上級者用）", expanded=False):
        period = st.selectbox("価格期間", ["1y", "2y", "5y", "10y"], index=3)
        interval = st.selectbox("価格間隔", ["1d", "1h"], index=0)
        show_meta = st.checkbox("取得メタ表示", value=False)
        show_debug = st.checkbox("デバッグ表示", value=False)
        allow_override = st.checkbox("EV閾値を手動上書き", value=False)
        if allow_override:
            min_expected_R = st.slider("min_expected_R", 0.0, 0.3, float(min_expected_R), 0.01)
            horizon_days = st.slider("horizon_days", 1, 14, int(horizon_days), 1)
        pair_custom = st.multiselect("スキャン対象（任意）", PAIR_LIST_DEFAULT, default=PAIR_LIST_DEFAULT)

    if st.button("🔄 キャッシュクリアして再取得"):
        st.cache_data.clear()
        st.rerun()

# defaults
period = locals().get("period", "10y")
interval = locals().get("interval", "1d")
show_meta = locals().get("show_meta", False)
show_debug = locals().get("show_debug", False)
pair_custom = locals().get("pair_custom", PAIR_LIST_DEFAULT)

keys = {
    "GEMINI_API_KEY": (locals().get("gemini_key","") or "").strip(),
    "NEWSAPI_KEY": (locals().get("news_key","") or "").strip(),
    "TRADING_ECONOMICS_KEY": (locals().get("te_key","") or "").strip(),
    "FRED_API_KEY": (locals().get("fred_key","") or "").strip(),
}

governor_cfg = {
    "max_drawdown_limit": float(locals().get("max_dd", 0.15)),
    "daily_loss_limit": float(locals().get("daily_stop", 0.03)),
    "max_losing_streak": int(locals().get("max_streak", 5)),
    "equity_drawdown": float(locals().get("equity_dd", 0.0)),
    "daily_loss": float(locals().get("daily_loss", 0.0)),
    "losing_streak": int(locals().get("losing_streak", 0)),
}

tabs = st.tabs(["🟢 AUTO判断", "🧪 バックテスト（WFA）", "📘 使い方"])

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
        for p in pairs:
            sym = _pair_label_to_symbol(p)
            df, price_meta = fetch_price_history(p, sym, period=period, interval="1d", prefer_stooq=True)
            if df.empty:
                rows.append({"pair": p, "EV": None, "decision": "NO_DATA", "confidence": None, "dom_state": None})
                continue

            feats, ext_meta = fetch_external(p, keys=keys)
            ctx = _build_ctx(p, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
            plan = logic.get_ai_order_strategy(api_key=keys.get("GEMINI_API_KEY",""), context_data=ctx)

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
        feats = best["_feats"]
        price = float(best["_ctx"].get("price", 0.0))

        # Top panel must show entry format + price (user request)
        _render_top_trade_panel(best["pair"], plan, price)

        # Risk dashboard (new)
        _render_risk_dashboard(plan, feats)

        st.markdown("### EVランキング（代替案ペアはここ）")
        view = [{
            "pair": r["pair"],
            "EV": float(r["EV"]),
            "decision": r["decision"],
            "confidence": float(r["confidence"]),
            "dominant_state": r["dom_state"],
            "global_risk": float(r["_feats"].get("global_risk_index", 0.0)),
            "war": float(r["_feats"].get("war_probability", 0.0)),
        } for r in ranked]
        st.dataframe(pd.DataFrame(view), use_container_width=True)

        st.markdown("### EV内訳（最適ペア）")
        ev_contribs = (plan.get("ev_contribs", {}) or {})
        if isinstance(ev_contribs, dict) and ev_contribs:
            cdf = pd.DataFrame([{"state": k, "contrib_R": float(v)} for k, v in ev_contribs.items()]).sort_values("contrib_R")
            st.bar_chart(cdf.set_index("state"))
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

        df, price_meta = fetch_price_history(pair_label, symbol, period=period, interval=interval, prefer_stooq=False)
        if df.empty:
            st.error("価格データ取得に失敗しました。")
            st.json(price_meta)
            st.stop()

        feats, ext_meta = fetch_external(pair_label, keys=keys)
        ctx = _build_ctx(pair_label, df, feats, horizon_days=int(horizon_days), min_expected_R=float(min_expected_R), style_name=style_name, governor_cfg=governor_cfg)
        plan = logic.get_ai_order_strategy(api_key=keys.get("GEMINI_API_KEY",""), context_data=ctx)

        price = float(ctx.get("price", 0.0))
        _render_top_trade_panel(pair_label, plan, price)
        _render_risk_dashboard(plan, feats)

        st.markdown("### EV内訳（何がEVを潰しているか）")
        ev_contribs = plan.get("ev_contribs", {}) or {}
        if isinstance(ev_contribs, dict) and ev_contribs:
            cdf = pd.DataFrame([{"state": k, "contrib_R": float(v)} for k, v in ev_contribs.items()]).sort_values("contrib_R")
            st.bar_chart(cdf.set_index("state"))
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
# Tab 3: Guide
# =========================
with tabs[2]:
    st.markdown("""
## 使い方（最短）
### ① まず「相場全体から最適ペアを自動抽出」
- 最上段に **最適ペア + 現在値 + 注文形式（Entry/SL/TP）** が出ます（TRADE時）
- 見送り時は **理由（veto）** が出ます（EV不足 / BlackSwan / Governor など）

### ② “代替案ペア” はどこ？
- **AUTO判断タブの「EVランキング」**が代替案一覧です（2位/3位が次候補）。

### ③ 世界情勢（地政学）を使うには
- キー無しでも **GDELT（無料）**で“ニュース量異常”は入ります
- `FRED_API_KEY` を入れると **VIX/DXY/金利** が入ります
- `TRADING_ECONOMICS_KEY` を入れると **CPI/NFPサプライズ** が入ります
- `NEWSAPI_KEY` を入れると **記事見出しセンチメント** が入ります
- `GEMINI_API_KEY` を入れると **LLMが地政学/危機確率をJSONで返し**、GlobalRiskに反映されます

### ④ 見送りが増える理由
- v4では **危機が近いほど動的閾値が上がる**（本気資金向け）
- さらに **Black Swan Guard** や **Capital Governor** が止めます
""")
