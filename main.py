# main.py
from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd

import yfinance as yf

import logic

# data_layer is optional; app will run without it (features become 0)
try:
    import data_layer  # type: ignore
except Exception:
    data_layer = None  # type: ignore

# yfinance rate limit exception (version-dependent)
try:
    from yfinance.exceptions import YFRateLimitError  # type: ignore
except Exception:
    class YFRateLimitError(Exception):
        pass


# -------------------------
# Helpers
# -------------------------
def _normalize_pair_label(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) == 6:
        s = s[:3] + "/" + s[3:]
    return s

def _pair_label_to_symbol(pair_label: str) -> str:
    pl = _normalize_pair_label(pair_label)
    if hasattr(logic, "PAIR_MAP") and isinstance(getattr(logic, "PAIR_MAP"), dict):
        sym = logic.PAIR_MAP.get(pl)
        if sym:
            return sym
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
    """
    Stooq symbols (common):
      usdjpy, eurusd, gbpusd, audusd, eurjpy, gbpjpy, audjpy
    Endpoint:
      https://stooq.com/q/d/l/?s=usdjpy&i=d
    """
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

def _load_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return os.getenv(name, default) or default


# -------------------------
# Price data fetch (robust)
# -------------------------

def _coerce_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]
    # Ensure OHLC exists
    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in d.columns:
            return pd.DataFrame()
    d = d[needed].dropna()
    return d

def _fetch_from_stooq(pair_label: str, interval: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stooq is mainly good for daily data. If interval != 1d, we still return daily to keep app alive.
    """
    meta = {"source": "stooq", "ok": False, "error": None, "interval_used": "1d"}
    stq = _pair_label_to_stooq_symbol(pair_label)
    if not stq:
        meta["error"] = "unsupported_pair_for_stooq"
        return pd.DataFrame(), meta

    # Stooq daily CSV
    url = f"https://stooq.com/q/d/l/?s={stq}&i=d"
    try:
        d = pd.read_csv(url)
        # columns: Date, Open, High, Low, Close, Volume
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
        # interval downgrade notice (handled in UI)
        if interval != "1d":
            meta["interval_used"] = "1d"
        return d, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return pd.DataFrame(), meta

@st.cache_data(ttl=60 * 60)  # 1 hour cache to reduce rate limit
def _fetch_price_history_robust(pair_label: str, symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tries yfinance first. If rate-limited or fails, falls back to Stooq (daily).
    Returns (df, meta).
    """
    meta: Dict[str, Any] = {"source": "yfinance", "ok": False, "error": None, "fallback": None, "interval_used": interval}

    # yfinance attempt with limited retries (don’t hammer)
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
    # fallback to Stooq (daily)
    df2, m2 = _fetch_from_stooq(pair_label, interval=interval)
    meta["fallback"] = m2
    if not df2.empty and m2.get("ok"):
        meta["source"] = "stooq"
        meta["ok"] = True
        meta["interval_used"] = m2.get("interval_used", "1d")
        return df2, meta

    return pd.DataFrame(), meta


# -------------------------
# External features
# -------------------------
@st.cache_data(ttl=60 * 30)
def _fetch_external(pair_label: str, keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if data_layer is None:
        return {
            "news_sentiment": 0.0,
            "cpi_surprise": 0.0,
            "nfp_surprise": 0.0,
            "rate_diff_change": 0.0,
            "cot_leveraged_net_pctoi": 0.0,
            "cot_asset_net_pctoi": 0.0,
        }, {"ok": False, "error": "data_layer_import_failed"}

    if hasattr(data_layer, "fetch_external_features"):
        try:
            return data_layer.fetch_external_features(pair_label, keys=keys)  # type: ignore[attr-defined]
        except Exception as e:
            return {
                "news_sentiment": 0.0,
                "cpi_surprise": 0.0,
                "nfp_surprise": 0.0,
                "rate_diff_change": 0.0,
                "cot_leveraged_net_pctoi": 0.0,
                "cot_asset_net_pctoi": 0.0,
            }, {"ok": False, "error": f"fetch_external_failed:{type(e).__name__}", "detail": str(e)}
    else:
        return {
            "news_sentiment": 0.0,
            "cpi_surprise": 0.0,
            "nfp_surprise": 0.0,
            "rate_diff_change": 0.0,
            "cot_leveraged_net_pctoi": 0.0,
            "cot_asset_net_pctoi": 0.0,
        }, {
            "ok": False,
            "error": "data_layer_missing_fetch_external_features",
            "data_layer_file": getattr(data_layer, "__file__", "unknown"),
        }


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="FX EV Ver1 (State Probabilities)", layout="wide")
st.title("FX 統合AI 状態確率モデル Ver1（EV最大化）")

with st.sidebar:
    st.markdown("## 設定")

    pair_label = st.text_input("通貨ペア（例: USD/JPY）", value="USD/JPY")
    pair_label = _normalize_pair_label(pair_label)
    symbol = _pair_label_to_symbol(pair_label)

    st.caption(f"primary source: yfinance `{symbol}` (fallback: stooq daily)")

    st.markdown("### 🔑 APIキー（任意：無くても落ちない）")
    gemini_key = st.text_input("GEMINI_API_KEY（HYBRID/LLM_ONLYで使用）", value=_load_secret("GEMINI_API_KEY", ""), type="password")
    te_key = st.text_input("TRADING_ECONOMICS_KEY（経済指標）", value=_load_secret("TRADING_ECONOMICS_KEY", ""), type="password")
    fred_key = st.text_input("FRED_API_KEY（金利差）", value=_load_secret("FRED_API_KEY", ""), type="password")

    st.markdown("### 🧠 意思決定エンジン（Ver1）")
    engine_ui = st.selectbox(
        "モード",
        ["HYBRID（EVゲート＋LLM解説）", "EV_V1（数値のみ）", "LLM_ONLY（従来のみ）"],
        index=0,
    )
    if "HYBRID" in engine_ui:
        decision_engine = "HYBRID"
    elif "EV_V1" in engine_ui:
        decision_engine = "EV_V1"
    else:
        decision_engine = "LLM_ONLY"

    min_expected_R = st.slider("EV閾値（min expected R）", 0.0, 1.0, 0.10, 0.01)
    horizon_days = st.number_input("EV horizon（日数）", min_value=1, max_value=20, value=5, step=1)

    show_meta = st.checkbox("外部データ取得メタを表示", value=False)
    show_debug = st.checkbox("デバッグ情報を表示", value=False)

    st.markdown("### 価格データ")
    period = st.selectbox("期間", ["1y", "2y", "5y", "10y"], index=3)
    interval = st.selectbox("間隔", ["1d", "1h"], index=0)

    # manual refresh (clears cache for price/external)
    if st.button("🔄 データ再取得（キャッシュクリア）"):
        st.cache_data.clear()
        st.toast("キャッシュをクリアしました。再読み込みします。")
        st.rerun()

v1_keys = {"TRADING_ECONOMICS_KEY": (te_key or "").strip(), "FRED_API_KEY": (fred_key or "").strip()}

tabs = st.tabs(["📌 注文戦略（Ver1）", "🧪 EVバックテスト（簡易WFA）", "ℹ️ 使い方・運用"])

with tabs[0]:
    df, price_meta = _fetch_price_history_robust(pair_label, symbol, period=period, interval=interval)

    if df.empty:
        st.error("価格データが取得できませんでした（yfinanceが制限/失敗、stooqも失敗）")
        st.json(price_meta)
        st.stop()

    st.subheader(f"{pair_label} / {symbol}")
    st.caption(f"Price source: {price_meta.get('source')} / interval_used: {price_meta.get('interval_used')}")
    if price_meta.get("source") == "stooq" and interval != "1d":
        st.warning("yfinanceが制限中のため、日足（stooq）に降格して表示しています。")

    st.line_chart(df["Close"])

    ctx: Dict[str, Any] = {}
    latest = df.dropna().iloc[-1]
    ctx["pair_label"] = pair_label
    ctx["pair_symbol"] = symbol
    ctx["price"] = float(latest["Close"])

    ind = logic.compute_indicators(df)
    ctx.update(ind)

    feats, meta = _fetch_external(pair_label, keys=v1_keys)
    ctx.update(feats)
    ctx["external_meta"] = meta

    ctx["decision_engine"] = decision_engine
    ctx["min_expected_R"] = float(min_expected_R)
    ctx["horizon_days"] = int(horizon_days)
    ctx["keys"] = v1_keys

    plan = logic.get_ai_order_strategy(
        api_key=gemini_key,
        context_data=ctx,
        generation_policy="AUTO_HIERARCHY",
        override_mode="AUTO",
        override_reason="",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### ✅ 出力（注文戦略）")
        if isinstance(plan, dict):
            st.json({
                "decision": plan.get("decision"),
                "side": plan.get("side"),
                "entry": plan.get("entry"),
                "take_profit": plan.get("take_profit"),
                "stop_loss": plan.get("stop_loss"),
                "confidence": plan.get("confidence"),
                "why": plan.get("why"),
            })
        else:
            st.write(plan)

    with c2:
        st.markdown("### 📊 状態確率 / EV")
        if isinstance(plan, dict):
            st.write("**state_probs**")
            st.json(plan.get("state_probs", {}))
            st.write("**EV**")
            st.json({
                "expected_R_ev": plan.get("expected_R_ev"),
                "p_win_ev": plan.get("p_win_ev"),
            })

    st.markdown("### 🌐 外部特徴量（Ver1）")
    st.json(feats)

    if show_meta:
        st.markdown("### 🧾 外部データ取得メタ")
        st.json(meta)
        st.markdown("### 🧾 価格データ取得メタ")
        st.json(price_meta)

    if show_debug:
        st.markdown("### 🛠️ ctx（内部）")
        st.json({k: v for k, v in ctx.items() if k not in ("keys",)})

with tabs[1]:
    st.subheader("簡易ウォークフォワード（EVゲート）")
    st.caption("注意：スプレッド/スリップ/指値到達率などの厳密約定は未考慮。Ver1の方向性確認用です。")

    colA, colB, colC = st.columns(3)
    with colA:
        bt_period = st.selectbox("BT期間", ["5y", "10y"], index=1, key="bt_period")
        bt_horizon = st.number_input("horizon_days", min_value=1, max_value=20, value=int(horizon_days), step=1, key="bt_horizon")
    with colB:
        train_years = st.number_input("train_years", min_value=1, max_value=8, value=3, step=1, key="train_years")
        test_months = st.number_input("test_months", min_value=1, max_value=24, value=6, step=1, key="test_months")
    with colC:
        bt_min_ev = st.slider("min_expected_R", 0.0, 1.0, float(min_expected_R), 0.01, key="bt_min_ev")

    run = st.button("バックテスト実行", type="primary")
    if run:
        try:
            import backtest_ev_v1
            wf, summ = backtest_ev_v1.run_backtest(
                pair_symbol=symbol,
                period=bt_period,
                horizon_days=int(bt_horizon),
                train_years=int(train_years),
                test_months=int(test_months),
                min_expected_R=float(bt_min_ev),
            )
            st.markdown("### Summary")
            st.json(summ)
            st.markdown("### Walk-Forward windows")
            st.dataframe(wf, use_container_width=True)

            csv = wf.to_csv(index=False).encode("utf-8")
            st.download_button("CSVダウンロード", data=csv, file_name=f"wfa_{pair_label.replace('/','')}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"バックテストでエラー: {type(e).__name__}: {e}")

with tabs[2]:
    st.markdown("""
## 追加機能（Ver1の“全部入り”）

### A) 価格データ取得の堅牢化（今回の修正ポイント）
- まず yfinance を試す
- **RateLimit（YFRateLimitError）なら自動で Stooq（日足）にフォールバック**
- どっちもダメなら、メタ情報（原因）を表示して停止
- キャッシュTTLを長く（1時間）して、Streamlit Cloudでの連打を回避
- 「データ再取得（キャッシュクリア）」ボタンで手動更新

### B) 外部データ取得（ニュース/経済指標/金利差/COT）
- data_layer.py が提供する features を ctx に合流
- 失敗時は 0 フォールバック（落ちない）
- 「外部データ取得メタ表示」で取得状況を確認

### C) EVゲート（期待値最大化の核）
- 状態確率（4状態）→ 状態別期待R → EV を計算
- EVが閾値未満なら NO_TRADE（無駄撃ちを抑制）

---

## 運用方法（Streamlit Cloudで安定させるコツ）
1. **intervalは基本 1d**
   - 1h は yfinance 依存が強いので RateLimitの時に落ちやすい。
2. 「データ再取得」ボタンは多用しない（キャッシュが効かなくなる）
3. どうしても1hが必要なら、Ver2で **TwelveData / AlphaVantage / Polygon** 等のAPIに切替（キー必須）
""")
