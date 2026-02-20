# main.py
from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import streamlit as st
import pandas as pd
import yfinance as yf

import logic

# data_layer is optional; the app will run without it (features become 0)
try:
    import data_layer  # type: ignore
except Exception:
    data_layer = None  # type: ignore


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

@st.cache_data(ttl=60 * 30)
def _fetch_price_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=60 * 30)
def _fetch_external(pair_label: str, keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    # Hard guard: never crash even if module is missing / wrong
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

def _load_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return os.getenv(name, default) or default


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

    st.caption(f"yfinance symbol: `{symbol}`")

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

v1_keys = {"TRADING_ECONOMICS_KEY": (te_key or "").strip(), "FRED_API_KEY": (fred_key or "").strip()}

tabs = st.tabs(["📌 注文戦略（Ver1）", "🧪 EVバックテスト（簡易WFA）", "ℹ️ 使い方・運用"])

with tabs[0]:
    df = _fetch_price_history(symbol, period=period, interval=interval)
    if df.empty:
        st.error("価格データが取得できませんでした（yfinance）")
    else:
        st.subheader(f"{pair_label} / {symbol}")
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
## 追加機能（Ver1で“全部入れ”した内容）

### 1) 外部データ取得層（0フォールバック＋キャッシュ）
- **ニュース（GDELT）**：APIキー不要。直近数日分のトーン（tone）を -1〜+1 に正規化し `news_sentiment` として利用。
- **経済指標（TradingEconomics）**：CPI/NFPの **Actual - Forecast** を「サプライズ指標」として `cpi_surprise / nfp_surprise` に格納。
- **金利差（FRED）**：米10年(DGS10)と日本10年(IRLTLT01JPM156N)のスプレッド変化を `rate_diff_change` に格納。
- **COT（CFTC）**：FinFutWk.txt からレバレッジ勢/アセット勢のネットポジションを `cot_*_net_pctoi` として格納（-1〜+1）。

> キーが無い・APIが落ちる・制限に当たる → **0** が入るだけで、アプリは落ちません。  
> 取得状況は「外部データ取得メタ表示」で確認できます。

### 2) 状態確率推定（Softmax）
以下の4状態の確率を返します：
- P(trend_up), P(trend_down), P(range), P(risk_off)

価格指標（SMA/RSI/ATR）＋外部特徴量（ニュース/指標/金利差/COT）を使い、ロジット→Softmaxで確率化します。

### 3) EV（期待値）ゲート
- 状態別の期待R（mean_R）を過去データから推定し、
- **EV = Σ(P(state) × mean_R(state))**
- EVが `min_expected_R` 未満なら **NO_TRADE** で見送り（機会損失より“無駄撃ち抑制”を優先）

### 4) 意思決定（STOP/LIMIT/MARKETの自動選択）
支配的状態により発注方式を決めます（Ver1の簡易版）：
- trend_up / trend_down → STOP（ブレイク想定）
- range → LIMIT（端での逆張り想定）
- risk_off → 原則 NO_TRADE（急変動回避）

### 5) エンジン切替（運用モード）
- **EV_V1**：数値のみ（再現性最優先）。まずここで“落ちない＆筋が通る”を確認。
- **HYBRID**：EVゲートで「撃つ/撃たない」を決め、撃つ時だけ LLMに“説明文だけ”生成（注文値は数値ロジック優先）。
- **LLM_ONLY**：従来型（おすすめしません：事故りやすい）。

---

## 運用手順（最短）
1. **EV_V1** で起動 → 外部特徴量が 0 以外になるか確認（キー無しなら一部は0でOK）
2. `min_expected_R` を 0.10〜0.20 から開始（無駄撃ち抑制）
3. 「🧪 EVバックテスト」で **AvgR（平均R）/ MaxDD(R)** を確認し、閾値と horizon を調整
4. HYBRIDに切替（必要なら）→ LLMは“説明”に限定して暴走を防ぐ

---

## 重要な注意
- このVer1は「期待値最大化」の土台です。**利益保証はできません**。
- 実運用には Ver2 で、スプレッド/スリッページ/指値到達率/ロールなどの約定モデルが必須です。
""")
