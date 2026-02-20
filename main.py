# main.py
from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd

# ---- optional deps ----
try:
    import yfinance as yf
except Exception:
    yf = None

# ---- local modules ----
import logic

try:
    import data_layer  # optional
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
def fetch_price_history(pair_label: str, symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust price fetch:
      - try yfinance
      - on rate limit / fail, fall back to stooq daily
    """
    meta: Dict[str, Any] = {"source": "yfinance", "ok": False, "error": None, "fallback": None, "interval_used": interval}

    # yfinance
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

    # fallback stooq daily
    df2, m2 = _fetch_from_stooq(pair_label)
    meta["fallback"] = m2
    if not df2.empty and m2.get("ok"):
        meta["source"] = "stooq"
        meta["ok"] = True
        meta["interval_used"] = m2.get("interval_used", "1d")
        return df2, meta

    return pd.DataFrame(), meta


@st.cache_data(ttl=60 * 30)
def fetch_external(pair_label: str, keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    External features are optional. Always returns a dict with expected keys, never crashes.
    """
    base = {
        "news_sentiment": 0.0,
        "cpi_surprise": 0.0,
        "nfp_surprise": 0.0,
        "rate_diff_change": 0.0,
        "cot_leveraged_net_pctoi": 0.0,
        "cot_asset_net_pctoi": 0.0,
    }
    if data_layer is None:
        return base, {"ok": False, "error": "data_layer_import_failed"}
    if not hasattr(data_layer, "fetch_external_features"):
        return base, {"ok": False, "error": "data_layer_missing_fetch_external_features", "file": getattr(data_layer, "__file__", "unknown")}
    try:
        feats, meta = data_layer.fetch_external_features(pair_label, keys=keys)  # type: ignore
        out = base.copy()
        out.update({k: float(v) for k, v in (feats or {}).items() if k in out})
        return out, meta if isinstance(meta, dict) else {"ok": True}
    except Exception as e:
        return base, {"ok": False, "error": f"fetch_external_failed:{type(e).__name__}", "detail": str(e)}


def _style_defaults(style_name: str) -> Dict[str, Any]:
    # User-friendly presets: no numeric fiddling
    if style_name == "保守":
        return {"min_expected_R": 0.12, "horizon_days": 7}
    if style_name == "攻撃":
        return {"min_expected_R": 0.03, "horizon_days": 5}
    return {"min_expected_R": 0.07, "horizon_days": 7}  # 標準


# =========================
# UI
# =========================
st.set_page_config(page_title="FX EV Auto Ver2", layout="wide")
st.title("FX 自動AI判断ツール（EV最大化）")

# --- Sidebar: only 3 controls in AUTO mode ---
with st.sidebar:
    st.header("AUTO運用（最小設定）")
    pair_label = _normalize_pair_label(st.text_input("通貨ペア", value="USD/JPY"))
    style_name = st.selectbox("運用スタイル", ["標準", "保守", "攻撃"], index=0)
    horizon_mode = st.selectbox("想定期間", ["週（推奨）", "日"], index=0)

    preset = _style_defaults(style_name)
    horizon_days = 7 if "週" in horizon_mode else 3
    min_expected_R = float(preset["min_expected_R"])
    horizon_days = int(horizon_days)

    st.divider()
    st.caption("必要なら下の「詳細設定」で微調整できます。")

    with st.expander("詳細設定（上級者用）", expanded=False):
        period = st.selectbox("価格期間", ["1y", "2y", "5y", "10y"], index=3)
        interval = st.selectbox("価格間隔", ["1d", "1h"], index=0)
        # Optional keys (kept here)
        gemini_key = st.text_input("GEMINI_API_KEY（任意）", value=_load_secret("GEMINI_API_KEY", ""), type="password")
        te_key = st.text_input("TRADING_ECONOMICS_KEY（任意）", value=_load_secret("TRADING_ECONOMICS_KEY", ""), type="password")
        fred_key = st.text_input("FRED_API_KEY（金利差・任意）", value=_load_secret("FRED_API_KEY", ""), type="password")
        show_debug = st.checkbox("デバッグ表示", value=False)
        show_meta = st.checkbox("取得メタ表示", value=False)

        # allow override thresholds if user insists
        allow_override = st.checkbox("AUTO設定を上書きする", value=False)
        if allow_override:
            min_expected_R = st.slider("min_expected_R", 0.0, 0.3, float(min_expected_R), 0.01)
            horizon_days = st.slider("horizon_days", 1, 14, int(horizon_days), 1)

    if st.button("🔄 キャッシュクリアして再取得"):
        st.cache_data.clear()
        st.rerun()

# defaults for non-expanded
period = locals().get("period", "10y")
interval = locals().get("interval", "1d")
gemini_key = locals().get("gemini_key", "")
te_key = locals().get("te_key", "")
fred_key = locals().get("fred_key", "")
show_debug = locals().get("show_debug", False)
show_meta = locals().get("show_meta", False)

symbol = _pair_label_to_symbol(pair_label)
keys = {"TRADING_ECONOMICS_KEY": (te_key or "").strip(), "FRED_API_KEY": (fred_key or "").strip()}

tabs = st.tabs(["🟢 AUTO判断", "🧪 バックテスト（WFA）", "📘 使い方"])

# =========================
# Tab 1: AUTO panel
# =========================
with tabs[0]:
    df, price_meta = fetch_price_history(pair_label, symbol, period=period, interval=interval)
    if df.empty:
        st.error("価格データ取得に失敗しました。")
        st.json(price_meta)
        st.stop()

    feats, ext_meta = fetch_external(pair_label, keys=keys)
    indicators = logic.compute_indicators(df)

    ctx: Dict[str, Any] = {}
    ctx.update(indicators)
    ctx.update(feats)
    ctx["pair_label"] = pair_label
    ctx["pair_symbol"] = symbol
    ctx["price"] = float(df["Close"].iloc[-1])
    ctx["horizon_days"] = int(horizon_days)
    ctx["min_expected_R"] = float(min_expected_R)
    ctx["style_name"] = style_name

    plan = logic.get_ai_order_strategy(api_key=gemini_key, context_data=ctx)

    # --------- TOP AUTO PANEL (single source of truth) ----------
    st.subheader("最終判断（ここだけ見ればOK）")
    if not isinstance(plan, dict):
        st.error("戦略出力が不正です。")
        st.write(plan)
        st.stop()

    decision = plan.get("decision", "NO_TRADE")
    expected_R_ev = float(plan.get("expected_R_ev") or 0.0)
    p_win_ev = float(plan.get("p_win_ev") or 0.0)
    confidence = float(plan.get("confidence") or 0.0)
    why = str(plan.get("why") or "")

    # headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("判定", decision)
    c2.metric("期待値EV (R)", f"{expected_R_ev:+.3f}")
    c3.metric("勝率(参考)", f"{p_win_ev*100:.1f}%")
    c4.metric("信頼度", f"{confidence:.2f}")

    if decision != "NO_TRADE":
        side = plan.get("side", "—")
        order_type = plan.get("order_type", "—")
        entry_type = plan.get("entry_type", "")
        entry = plan.get("entry", None)
        sl = plan.get("stop_loss", None)
        tp = plan.get("take_profit", None)

        st.success("✅ エントリー条件を満たしました（AUTO）")

        entry_type_display = entry_type if entry_type else "—"
        st.markdown(f"""
- **売買**: {side} / **注文**: {order_type} / **種別**: {entry_type_display}
- **Entry**: {entry if entry is not None else '—'}
- **SL**: {sl if sl is not None else '—'}
- **TP**: {tp if tp is not None else '—'}
""")
    else:
        st.warning("⏸ 見送り（NO_TRADE）")
        st.markdown(f"**理由**: {why}")

    # --------- EV breakdown visualization ----------
    st.markdown("### EV内訳（何がEVを潰しているか）")
    ev_contribs = plan.get("ev_contribs", {}) or {}
    if isinstance(ev_contribs, dict) and ev_contribs:
        cdf = pd.DataFrame(
            [{"state": k, "contrib_R": float(v)} for k, v in ev_contribs.items()]
        ).sort_values("contrib_R")
        st.bar_chart(cdf.set_index("state"))
        worst = cdf.iloc[0]
        best = cdf.iloc[-1]
        st.caption(f"EVを最も押し下げている: {worst['state']} ({worst['contrib_R']:+.3f}R) / 押し上げている: {best['state']} ({best['contrib_R']:+.3f}R)")
    else:
        st.info("EV内訳がまだ生成されていません（plan.ev_contribs が空）。")

    # --------- Details (collapsed) ----------
    with st.expander("詳細（診断・データ・内部情報）", expanded=False):
        st.markdown("#### 状態確率")
        st.json(plan.get("state_probs", {}))

        st.markdown("#### 外部特徴量")
        st.json(feats)

        if show_meta:
            st.markdown("#### 取得メタ")
            st.json({"price_meta": price_meta, "external_meta": ext_meta})

        if show_debug:
            st.markdown("#### Indicators / ctx")
            st.json({"indicators": indicators, "ctx": ctx})

# =========================
# Tab 2: Backtest
# =========================
with tabs[1]:
    st.subheader("ウォークフォワード（WFA）バックテスト")
    st.caption("方向性確認用。コスト・スリップは未反映（次段で追加推奨）。")

    colA, colB, colC = st.columns(3)
    with colA:
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
            wf_df, summ = backtest_ev_v1.run_backtest(
                pair_symbol=symbol,
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

            st.markdown("### 判定（自動コメント）")
            if isinstance(wf_df, pd.DataFrame) and not wf_df.empty:
                total_trades = int(wf_df.get("n_trades", pd.Series([0])).sum())
                sum_R = float(wf_df.get("sum_R", pd.Series([0.0])).sum())
                max_dd = float(wf_df.get("max_dd_R", pd.Series([0.0])).max())
                avg_R_per_trade = (sum_R / total_trades) if total_trades > 0 else 0.0

                wf2 = wf_df.copy()
                if "test_end" in wf2.columns:
                    wf2["test_end"] = pd.to_datetime(wf2["test_end"], errors="coerce")
                    wf2 = wf2.sort_values("test_end")
                last = wf2.iloc[-1]
                last_sum = float(last.get("sum_R", 0.0))
                last_trades = int(last.get("n_trades", 0))

                verdict = "注意"
                if total_trades == 0:
                    verdict = "データ不足/厳しすぎ"
                elif last_sum < 0:
                    verdict = "停止推奨（直近悪化）"
                elif sum_R > 0 and max_dd <= 20:
                    verdict = "合格（小ロット運用可）"
                elif sum_R > 0:
                    verdict = "注意（DD大）"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("判定", verdict)
                c2.metric("合計R", f"{sum_R:+.2f}R")
                c3.metric("平均R/回", f"{avg_R_per_trade:+.3f}R")
                c4.metric("最大DD", f"{max_dd:.1f}R")

                tips = []
                if last_sum < 0:
                    tips.append("直近6ヶ月がマイナスです。AUTO運用は停止し、閾値(min_expected_R)を上げる/レジーム判定を強化が必要。")
                if max_dd > 20:
                    tips.append("最大DDが大きいです。ロット縮小・自動停止ルール導入を推奨。")
                if last_trades == 0 and total_trades > 0:
                    tips.append("直近で取引が出ていません。閾値が高い/条件が厳しい可能性があります。")
                if not tips:
                    tips.append("成績は概ね安定。次はコスト（スプレッド/スリップ）を入れて現実寄せする段階です。")
                st.write("**運用コメント**")
                for t in tips:
                    st.write(f"- {t}")

                if "test_end" in wf2.columns and "sum_R" in wf2.columns:
                    st.line_chart(wf2.set_index("test_end")["sum_R"])
                if "test_end" in wf2.columns and "max_dd_R" in wf2.columns:
                    st.line_chart(wf2.set_index("test_end")["max_dd_R"])

            csv = wf_df.to_csv(index=False).encode("utf-8")
            st.download_button("CSVダウンロード", data=csv, file_name=f"ev_wfa_{pair_label.replace('/','_')}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"バックテストでエラー: {type(e).__name__}: {e}")

# =========================
# Tab 3: Guide
# =========================
with tabs[2]:
    st.markdown("""
## ここだけ読めば使えます

### ① いちばん上の「最終判断」だけ見てください
- **判定が NO_TRADE** → 見送り（無駄撃ち回避）
- **判定が TRADE（STOP/LIMIT/MARKET）** → Entry/SL/TP が表示されます

### ② 見送りのときは「EV内訳」を見る
棒グラフで **どれがEVを潰しているか** が分かります。  
例：risk_off が大きくマイナス → そのリスクが残っているので見送り。

### ③ スタイルは3つだけ（数値を触らない）
- 保守：厳選して回数少なめ
- 標準：バランス
- 攻撃：回数多め（その分リスク増）

### ④ バックテストは「直近窓」が最重要
- 直近6ヶ月がマイナスなら停止推奨（相場が変わった可能性が高い）
""")
