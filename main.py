
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import base64
import math
import os
from datetime import datetime, timedelta
import pytz
import logic  # ← logic.pyが必要

# --- 起動時セルフチェック（logic.pyの差し替えミスを即検知） ---
_REQUIRED_LOGIC = [
    "get_market_data", "calculate_indicators", "judge_condition",
    "get_latest_quote", "get_ai_range", "get_ai_analysis", "get_ai_order_strategy",
    "get_ai_portfolio", "get_currency_strength",
    "suggest_alternative_pair_if_usdjpy_stay", "violates_currency_concentration", "can_open_under_weekly_cap",
]
_missing = [name for name in _REQUIRED_LOGIC if not hasattr(logic, name)]
if _missing:
    st.error("❌ logic.py に必要な関数が見つかりません（差し替えミスの可能性大）。不足: " + ", ".join(_missing))
    st.error("👉 対処: 私が渡した修正版 logic_fixed_final.py を logic.py にリネームして差し替えてください。")
    st.stop()


# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer 2026")
st.title("🤖 AI連携型 USD/JPY 戦略分析ツール (SBI仕様)")

TOKYO = pytz.timezone("Asia/Tokyo")

# --- Backtest / as-of date (paper trading) ---
BACKTEST_AS_OF_DATE = None  # date
BACKTEST_AS_OF_TS = None    # pandas Timestamp (end of day JST)
BACKTEST_FETCH_PERIOD = None

def _today_tokyo_str():
    return datetime.now(TOKYO).strftime("%Y%m%d")

def _choose_period_for_asof(as_of_date):
    """Pick a yfinance-compatible period string that surely covers as_of_date."""
    try:
        today = datetime.now(TOKYO).date()
        if as_of_date is None:
            return "1y"
        delta = (today - as_of_date).days
        if delta <= 370:
            return "2y"
        if delta <= 365 * 5 + 30:
            return "5y"
        if delta <= 365 * 10 + 30:
            return "10y"
        return "max"
    except Exception:
        return "5y"

def _write_jsonl(path: str, rec: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _shadow_diff(a: dict, b: dict):
    def _f(d, k, default=0.0):
        try:
            return float((d or {}).get(k, default))
        except Exception:
            return default
    def _s(d, k, default=""):
        v = (d or {}).get(k, default)
        return str(v) if v is not None else default
    return {
        "decision_diff": _s(a, "decision") != _s(b, "decision"),
        "side_diff": _s(a, "side") != _s(b, "side"),
        "entry_delta": _f(a, "entry") - _f(b, "entry"),
        "tp_delta": _f(a, "take_profit") - _f(b, "take_profit"),
        "sl_delta": _f(a, "stop_loss") - _f(b, "stop_loss"),
        "p_win_delta": _f(a, "p_win") - _f(b, "p_win"),
        "expected_R_delta": _f(a, "expected_R") - _f(b, "expected_R"),
        "confidence_delta": _f(a, "confidence") - _f(b, "confidence"),
    }


def _parse_bool(v, default: bool = False) -> bool:
    """Parse booleans from Streamlit secrets/UI which may deliver strings like 'false'."""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    return default

try:
    DEV_MODE = _parse_bool(st.secrets.get("DEV_MODE", False), False)
except Exception:
    DEV_MODE = False


# --- SBI必要証拠金（1万通貨あたり / JPY） ---
# ユーザー提示の固定値を優先して「最大発注可能数（枚）」を計算します。
# ※SBI側の改定があり得るので、数値は必要に応じて更新してください。
SBI_MARGIN_10K_JPY = {
    "USD/JPY (ドル円)": 63000,
    "EUR/USD (ユーロドル)": 75000,
    "GBP/USD (ポンドドル)": 86000,
    "AUD/USD (豪ドル米ドル)": 45000,
    "EUR/JPY (ユーロ円)": 75000,
    "GBP/JPY (ポンド円)": 86000,
    "AUD/JPY (豪ドル円)": 45000,
}

# --- Pair-context builder for alternative pairs (prevents hallucination / wrong indicators) ---
def _normalize_pair_label(label: str) -> str:
    """Try to map AI-returned label to an existing PAIR_MAP key."""
    try:
        if hasattr(logic, "PAIR_MAP") and label in logic.PAIR_MAP:
            return label
    except Exception:
        pass
    head = (label or "").split()[0]
    try:
        if hasattr(logic, "PAIR_MAP"):
            for k in logic.PAIR_MAP.keys():
                if (k or "").split()[0] == head:
                    return k
    except Exception:
        pass
    return label

def _build_ctx_for_pair(pair_label: str, base_ctx: dict, us10y_raw):
    """Build context_data (price/ATR/RSI/SMA_DIFF) for a specific FX pair label."""
    pair_label = _normalize_pair_label(pair_label)
    ctx2 = dict(base_ctx or {})
    ctx2["pair_label"] = pair_label

    sym = None
    try:
        if hasattr(logic, "PAIR_MAP"):
            sym = logic.PAIR_MAP.get(pair_label)
    except Exception:
        sym = None

    if not sym:
        try:
            if hasattr(logic, "_pair_label_to_symbol"):
                sym = logic._pair_label_to_symbol(pair_label)
        except Exception:
            sym = None

    if sym:
        ctx2["ticker"] = sym
        try:
            raw = None
            if hasattr(logic, "_fetch_ohlc"):
                raw = logic._fetch_ohlc(sym, period="1y", interval="1d")
            elif hasattr(logic, "_yahoo_chart"):
                raw = logic._yahoo_chart(sym, rng="1y", interval="1d")

            df2 = logic.calculate_indicators(raw, us10y_raw) if raw is not None else None
            if df2 is not None and not df2.empty:
                try:
                    df2.index = pd.to_datetime(df2.index)
                    if BACKTEST_AS_OF_TS is not None:
                        df2 = df2.loc[df2.index <= BACKTEST_AS_OF_TS]
                except Exception:
                    pass
                if df2 is None or df2.empty:
                    ctx2["_pair_ctx_ok"] = False
                    return ctx2
                lr = df2.iloc[-1]
                def _get(col, default):
                    try:
                        v = lr[col]
                        return float(v) if pd.notna(v) else float(default)
                    except Exception:
                        return float(default)

                ctx2["price"] = _get("Close", ctx2.get("price", 0.0))
                ctx2["atr"] = _get("ATR", ctx2.get("atr", 0.0))
                ctx2["rsi"] = _get("RSI", ctx2.get("rsi", 50.0))
                ctx2["sma_diff"] = _get("SMA_DIFF", ctx2.get("sma_diff", 0.0))
                ctx2["sma25"] = _get("SMA_25", ctx2.get("sma25", ctx2.get("price", 0.0)))
                ctx2["sma75"] = _get("SMA_75", ctx2.get("sma75", ctx2.get("price", 0.0)))
                try:
                    ctx2["atr_avg60"] = float(df2["ATR"].tail(60).mean()) if ("ATR" in df2.columns and df2["ATR"].tail(60).notna().any()) else ctx2.get("atr", 0.0)
                except Exception:
                    ctx2["atr_avg60"] = ctx2.get("atr", 0.0)
                ctx2["us10y"] = _get("US10Y", ctx2.get("us10y", 0.0))
                ctx2["_pair_ctx_ok"] = True
                return ctx2
        except Exception:
            pass

    ctx2["_pair_ctx_ok"] = False
    return ctx2





def _get_df_for_pair(pair_label: str, us10y_raw):
    """
    チャート表示用に、指定ペアのOHLCを取得して指標計算したDataFrameを返す。
    - USD/JPY以外の代替ペアでも「グラフ1」を切り替えられるようにするため。
    - 失敗時は None を返す。
    """
    pair_label = _normalize_pair_label(pair_label)
    sym = None
    try:
        sym = getattr(logic, "PAIR_MAP", {}).get(pair_label)
    except Exception:
        sym = None
    if not sym:
        try:
            if hasattr(logic, "_pair_label_to_symbol"):
                sym = logic._pair_label_to_symbol(pair_label)
        except Exception:
            sym = None
    if not sym:
        return None

    try:
        raw = None
        if hasattr(logic, "_fetch_ohlc"):
            raw = logic._fetch_ohlc(sym, period="1y", interval="1d")
        elif hasattr(logic, "_yahoo_chart"):
            raw = logic._yahoo_chart(sym, rng="1y", interval="1d")
        df2 = logic.calculate_indicators(raw, us10y_raw) if raw is not None else None
        if df2 is None or df2.empty:
            return None
        df2.index = pd.to_datetime(df2.index)
        if BACKTEST_AS_OF_TS is not None:
            try:
                df2 = df2.loc[df2.index <= BACKTEST_AS_OF_TS]
            except Exception:
                pass
        if df2 is None or df2.empty:
            return None
        return df2
    except Exception:
        return None


def _strategy_to_overlay(pair_label: str, strategy: dict):
    """注文戦略dictから、チャートに重ねるEntry/TP/SLライン情報を抽出してsessionに保持する。"""
    if not isinstance(strategy, dict):
        return None
    if strategy.get("decision") != "TRADE":
        return None
    try:
        entry = float(strategy.get("entry", 0) or 0)
        tp = float(strategy.get("take_profit", 0) or 0)
        sl = float(strategy.get("stop_loss", 0) or 0)
    except Exception:
        return None
    if entry <= 0 or tp <= 0 or sl <= 0:
        return None
    return {"pair_label": _normalize_pair_label(pair_label), "entry": entry, "tp": tp, "sl": sl}


# --- 表示用: JSONキーを日本語化（注文命令書・代替提案の表示専用）---
_KEY_JP = {
    # 注文命令書
    "decision": "判定",
    "side": "売買方向",
    "entry": "エントリー価格",
    "take_profit": "利確（TP）",
    "stop_loss": "損切（SL）",
    "horizon": "想定期間",
    "confidence": "確信度",
    "why": "理由",
    "notes": "注記",
    "market_regime": "相場モード",
    "regime_why": "モード理由",

    # 代替ペア提案
    "best_pair_name": "推奨ペア",
    "reason": "理由",
    "blocked": "ブロック",
    "blocked_by": "ブロック理由",
    "candidates": "候補",
    "pair": "ペア",
    "status": "状態",
    "rejected_by": "落選理由",
    "source": "出典",

    # 参考（ctx / ポートフォリオ表示などで使う可能性）
    "pair_label": "ペア",
    "ticker": "ティッカー",
    "direction": "方向",
    "risk_percent": "リスク（%）",
    "entry_price": "建値",
    "entry_time": "建玉時刻",
    "current_time": "現在時刻",
    "is_gotobi": "五十日",
    "capital": "資金（JPY）",
    "us10y": "米10年債利回り",
    "atr": "ATR",
    "atr_avg60": "ATR平均（60日）",
    "rsi": "RSI",
    "sma_diff": "MA乖離",
    "sma25": "SMA25",
    "sma75": "SMA75",
    "panel_short": "短期パネル",
    "panel_mid": "中期パネル",
    "last_report": "前回レポート",
    # 週末判断（JSON）
    "action": "週末アクション",
    "levels": "水準",
    "trail": "トレール",
    "month_hold_line": "1か月保有ライン",
    "structure_ok": "構造OK",
    "structure_detail": "構造詳細",
    "higher_high": "週足高値更新",
    "lower_low": "週足安値更新",
    "close_confirm": "週足終値確認",
    "cur_high": "今週高値",
    "cur_low": "今週安値",
    "cur_close": "今週終値",
    "prior_high_max": "過去高値(窓)",
    "prior_low_min": "過去安値(窓)",

}

_DECISION_JP = {
    "TRADE": "取引",
    "NO_TRADE": "見送り",
    "BUY": "買い",
    "SELL": "売り",
    "HOLD_WEEK": "週で確定",
    "HOLD_MONTH": "1か月保有",
    "STAY": "見送り",
    "TAKE_PROFIT": "利確",
    "CUT_LOSS": "損切",
    "NO_POSITION": "ノーポジ",

}
_SIDE_JP = {"LONG": "買い", "SHORT": "売り", "NONE": "なし"}
_HORIZON_JP = {"DAY": "1日", "WEEK": "1週間", "MONTH": "1か月"}
_REGIME_JP = {"DEFENSIVE": "守備", "OFFENSIVE": "攻勢", "NEUTRAL": "中立", "RANGE": "レンジ", "TREND": "トレンド"}

def _jpize_value(key: str, val):
    try:
        if isinstance(val, bool):
            return "はい" if val else "いいえ"
        if key == "action" and isinstance(val, str):
            return _DECISION_JP.get(val, val)
        if key == "decision" and isinstance(val, str):
            return _DECISION_JP.get(val, val)
        if key == "side" and isinstance(val, str):
            return _SIDE_JP.get(val, val)
        if key == "horizon" and isinstance(val, str):
            return _HORIZON_JP.get(val, val)
        if key == "market_regime" and isinstance(val, str):
            return _REGIME_JP.get(val, val)
    except Exception:
        pass
    return val

def jpize_json(obj):
    """辞書キーを日本語化したコピーを返す（表示専用）。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            jk = _KEY_JP.get(k, k)
            out[jk] = jpize_json(_jpize_value(k, v))
        return out
    if isinstance(obj, list):
        return [jpize_json(x) for x in obj]
    return obj


# --- シンプル表示ヘルパー（注文書/代替提案の見やすさ改善） ---
def _dget(d: dict, *keys, default=""):
    for k in keys:
        try:
            v = d.get(k)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return default

def render_order_summary(order: dict, pair_name: str = "", title: str = "📌 注文サマリー"):
    """注文命令書(dict)を、エントリー判断に必要な項目だけに絞って表示する。"""
    if not isinstance(order, dict):
        st.markdown(order)
        return

    decision = _dget(order, "判定", "decision", default="")
    side = _dget(order, "売買方向", "side", default="")
    entry = _dget(order, "エントリー価格", "entry", default=0)
    tp = _dget(order, "利確（TP）", "take_profit", "tp", default=0)
    sl = _dget(order, "損切（SL）", "stop_loss", "sl", default=0)
    horizon = _dget(order, "想定期間", "horizon", default="")
    conf = _dget(order, "確信度", "confidence", default="")
    method = _dget(order, "bundle_hint_jp", "order_bundle", "entry_price_kind_jp", default="")
    rr = _dget(order, "rr_ratio", default="")


    gen = _dget(order, "生成経路", "generator_path", default="")


    gen_map = {
            "ai_strict": "AI(1回)",
            "ai": "AI",
            "ai_retry": "AI再生成",
            "ai_retry_failed": "AI再生成(失敗)",
            "numeric_fallback": "数値フォールバック",
            "numeric_fallback_failed": "数値フォールバック(失敗)",
            "numeric_fallback_blocked": "数値フォールバック(ブロック)",
            "error": "エラー",
    }
    gen_disp = gen_map.get(str(gen), str(gen)) if gen else ""

    why = _dget(order, "理由", "why", default="")
    regime = _dget(order, "相場モード", "market_regime", default="")
    regime_why = _dget(order, "モード理由", "regime_why", default="")

    head = f"{title}"
    if pair_name:
        head += f"（{pair_name}）"
    st.subheader(head)

    if str(decision) in ["取引", "TRADE"]:
        st.success(f"✅ 判定: {decision} / 方向: {side} / 期間: {horizon} / 確信度: {conf}" + (f" / 生成: {gen_disp}" if gen_disp else ""))
    else:
        st.warning(f"⛔ 判定: {decision} / 方向: {side} / 期間: {horizon} / 確信度: {conf}" + (f" / 生成: {gen_disp}" if gen_disp else ""))

    try:
        entry_f = float(entry)
        tp_f = float(tp)
        sl_f = float(sl)
        rr_f = float(rr) if rr not in ("", None) else None
        line = f"**エントリー**: {entry_f:.3f} / **利確TP**: {tp_f:.3f} / **損切SL**: {sl_f:.3f}  \\n**注文方式**: {method}"
        if rr_f is not None:
            line += f" / **RR**: {rr_f:.2f}"
        st.markdown(line)
    except Exception:
        st.markdown(f"**エントリー**: {entry} / **TP**: {tp} / **SL**: {sl}  \\n**注文方式**: {method}")

    if why:
        w = str(why).strip()
        if len(w) > 220:
            w = w[:220] + " …"
        st.caption(f"理由: {w}")

    if regime or regime_why:
        with st.expander("相場モード（参考）"):
            if regime:
                st.write(f"相場モード: {regime}")
            if regime_why:
                st.write(regime_why)

def render_alt_summary(alt: dict, title: str = "🔁 代替ペア提案サマリー"):
    if not isinstance(alt, dict):
        st.markdown(alt)
        return
    pair = _dget(alt, "推奨ペア", "best_pair_name", default="")
    conf = _dget(alt, "確信度", "confidence", default="")
    blocked = _dget(alt, "ブロック", "blocked", default="")
    reason = _dget(alt, "理由", "reason", default="")
    st.subheader(title)
    if pair:
        st.info(f"候補: **{pair}** / 確信度: **{conf}** / ブロック: **{blocked}**")
    else:
        st.warning(f"候補なし / ブロック: {blocked}")
    if reason:
        r = str(reason).strip()
        if len(r) > 240:
            r = r[:240] + " …"
        st.caption(f"理由: {r}")


    # ✅ 候補（最大3）と「落選理由」を表示（学習＋監査＝事故防止）
    cand = alt.get("候補") if isinstance(alt.get("候補"), list) else alt.get("candidates")
    if isinstance(cand, list) and cand:
        st.markdown("**候補（最大3）**")
        for i, c in enumerate(cand[:3], start=1):
            if not isinstance(c, dict):
                continue
            p = _dget(c, "ペア", "pair", default="")
            conf2 = _dget(c, "確信度", "confidence", default="")
            stt = _dget(c, "状態", "status", default="")
            rej = _dget(c, "落選理由", "rejected_by", default=[])
            if isinstance(rej, list):
                rej_txt = ", ".join([str(x) for x in rej if str(x).strip()])
            else:
                rej_txt = str(rej).strip()
            # ステータスの日本語化
            if stt == "SELECTED":
                stt_jp = "採用"
            elif stt == "REJECTED":
                stt_jp = "落選"
            elif stt == "CANDIDATE":
                stt_jp = "候補"
            else:
                stt_jp = str(stt) if stt else "候補"

            line = f"{i}. {p}（{stt_jp} / 確信度:{conf2}）"
            if rej_txt:
                line += f" / 落選理由: {rej_txt}"
            st.caption(line)

# --- 状態保持の初期化 ---
if "ai_range" not in st.session_state:
    st.session_state.ai_range = None
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state:
    st.session_state.last_ai_report = ""

# ✅【追加】シャドー比較結果（Gemini vs GPT）
if "last_shadow_base" not in st.session_state:
    st.session_state.last_shadow_base = None
if "last_shadow_openai" not in st.session_state:
    st.session_state.last_shadow_openai = None
if "last_shadow_diff" not in st.session_state:
    st.session_state.last_shadow_diff = None


# ✅【追加】注文命令書/代替ペアの状態保持（Streamlitのボタン再実行対策）
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = None
if "last_alt" not in st.session_state:
    st.session_state.last_alt = None
if "last_alt_strategy" not in st.session_state:
    st.session_state.last_alt_strategy = None

# ✅【追加】ロット計算機の“対象ペア”を自動追従させる（USD/JPY or 代替ペア）
if "calc_pair_label" not in st.session_state:
    st.session_state.calc_pair_label = "USD/JPY (ドル円)"
if "calc_ctx" not in st.session_state:
    st.session_state.calc_ctx = None
if "calc_strategy" not in st.session_state:
    st.session_state.calc_strategy = None

# ✅【追加】週末判断（JSON）状態保持
if "last_weekend" not in st.session_state:
    st.session_state.last_weekend = None

# ✅【追加】ポートフォリオ（複数ポジション）状態
if "portfolio_positions" not in st.session_state:
    # 各要素: {"pair": str, "direction": "LONG/SHORT", "risk_percent": float, "entry_price": float, "entry_time": iso}
    st.session_state.portfolio_positions = []

# ✅【追加】チャート表示の対象ペア（USD/JPY or 代替ペア）
if "chart_pair_label" not in st.session_state:
    st.session_state.chart_pair_label = "USD/JPY (ドル円)"
# ✅【追加】チャート重ね表示ライン（entry/tp/sl）
if "chart_overlay" not in st.session_state:
    st.session_state.chart_overlay = None

# ✅【追加】代替候補の評価（最大3）を表示するための保持
if "last_alt" not in st.session_state:
    st.session_state.last_alt = None
if "last_alt_strategy" not in st.session_state:
    st.session_state.last_alt_strategy = None


# --- APIキー取得 ---
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# --- シャドー比較（Gemini vs GPT） ---
try:
    _default_openai_key = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    _default_openai_key = ""
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 シャドー比較（Gemini vs GPT）")
shadow_enabled = st.sidebar.checkbox(
    "シャドー比較を有効化（ログ保存）",
    value=False,
    help="Geminiの実運用判断はそのまま。OpenAIを追加で呼んで比較し、ログ（logs/shadow_compare_YYYYMMDD.jsonl）へ保存します。"
)
openai_api_key_shadow = st.sidebar.text_input(
    "OpenAI API Key (shadow)",
    value=_default_openai_key,
    type="password",
    help="未入力ならシャドー比較は実行されません。"
)
openai_model_shadow = st.sidebar.selectbox(
    "シャドーGPTモデル",
    options=["gpt-5-mini", "gpt-4o-mini", "gpt-5", "gpt-4o", "gpt-4.1"],
    index=0,
    help="まずは gpt-5-mini 推奨（コスパ良）。"
)


# --- サイドバー設定 (資金管理機能追加) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI FX 資金管理")

# 1. 資金管理入力
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider(
    "1トレード許容損失 (%)", 1.0, 10.0, 2.0,
    help="負けた時に資金の何%を失う覚悟があるか。プロは2%推奨。"
)
# ✅ ここはあなたの新機能で参照しているので、UI側でも定義しておく（削除ではなく追加）
weekly_dd_cap_percent = st.sidebar.slider(
    "週単位DDキャップ (%)", 0.5, 5.0, 2.0, 0.1,
    help="週単位で許容する損失上限（全ポジ合計リスク%）。"
)
max_positions_per_currency = st.sidebar.number_input(
    "同一通貨の最大保有数（通貨集中フィルタ）", min_value=1, max_value=5, value=1, step=1
)


# ✅【追加】固定1建（SBI最小1枚）現実対応
fixed_1lot_mode = st.sidebar.checkbox(
    "固定1建運用モード（SBI最小1枚前提）",
    value=True,
    help="最小1枚の制約で2%に収まらない場合でも、下の『許容最大リスク%（上限）』以内なら1枚で許可します。"
)
max_risk_percent_cap = st.sidebar.slider(
    "許容最大リスク%（上限）",
    2.0, 12.0, 6.0, 0.5,
    help="2%は目標。固定1枚で2%を超える局面は上限以内なら取引可、上限超はNO_TRADE。"
)
prefer_pullback_limit = st.sidebar.checkbox(
    "リスク過大時は押し目LIMIT案を優先（3案生成）",
    value=True,
    help="AI案が遠い/損切幅が広い場合に、押し目LIMIT/確認後成行の代替案を自動生成して採用します。"
)

# ✅【追加】過去検証（紙トレ/回数見積もり）
st.sidebar.markdown("---")
st.sidebar.subheader("🕰 過去検証（紙トレ/回数見積もり）")
backtest_enabled = st.sidebar.checkbox(
    "過去日付で評価（データをその日までに固定）",
    value=False,
    help="実ポジを取らずに『その日時点でツールがどう判断したか』を再現します。クオート更新は無効になります。"
)
_default_asof = (datetime.now(TOKYO) - timedelta(days=7)).date()
as_of_date = st.sidebar.date_input(
    "評価日（JST）",
    value=_default_asof,
    disabled=not backtest_enabled,
    help="この日付までのデータで固定して、同じボタン（詳細レポート/注文命令書）を実行できます。"
)
# グローバルへ反映（関数群が参照）
if backtest_enabled:
    BACKTEST_AS_OF_DATE = as_of_date
    BACKTEST_AS_OF_TS = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    BACKTEST_FETCH_PERIOD = _choose_period_for_asof(as_of_date)
else:
    BACKTEST_AS_OF_DATE = None
    BACKTEST_AS_OF_TS = None
    BACKTEST_FETCH_PERIOD = None


# ✅【追加】過去N週一括：TRADE回数見積もり（2/4/6%）
st.sidebar.markdown("")
st.sidebar.markdown("**📊 過去N週を一括で回して『TRADE回数』を集計（2%/4%/6%）**")
_batch_weekday_label = st.sidebar.selectbox(
    "集計の基準曜日（JST）",
    options=["水曜（週中判断）", "月曜（週初判断）"],
    index=0,
    help="その曜日の終値までのデータで『その時点の判断』を再現します。"
)
_batch_weekday = 2 if _batch_weekday_label.startswith("水曜") else 0  # Mon=0, Wed=2
_batch_n_weeks = st.sidebar.number_input(
    "過去N週",
    min_value=1, max_value=156, value=12, step=1,
    help="例：12なら直近12回（週ごと）を集計します。"
)
_batch_scan_pairs = st.sidebar.checkbox(
    "USD/JPYがNO_TRADEでも代替ペアをスキャン（FAST・AIなし）",
    value=True,
    help="PAIR_MAP（7通貨ペア）を数値ゲートでスキャンし、最も強いペアで判定します（AIは呼びません）。"
)
_batch_use_ai = st.sidebar.checkbox(
    "AIも呼んで厳密に再現（高コスト/遅い）",
    value=False,
    help="各週ごとにAI呼び出しが発生します。まずはOFF推奨（数値フォールバックで十分に回数傾向を見れます）。"
)
if st.sidebar.button("📊 一括集計を実行", use_container_width=True):
    st.session_state["_batch_run_flag"] = True
    st.session_state["_batch_params"] = {
        "n_weeks": int(_batch_n_weeks),
        "weekday": int(_batch_weekday),
        "scan_pairs": bool(_batch_scan_pairs),
        "use_ai": bool(_batch_use_ai),
        "capital_jpy": float(capital),
        "risk_percent_target": float(risk_percent),
        "fixed_1lot_mode": bool(fixed_1lot_mode),
        "max_positions_per_currency": int(max_positions_per_currency),
        "leverage": 25,  # fixed (avoid NameError before leverage is defined)
        "api_key": str(api_key or ""),
    }

# ✅【追加】デバッグ（テスト用）: Secretsの DEV_MODE=true のときだけ表示
if DEV_MODE:
    st.sidebar.subheader("🧪 デバッグ")
    force_no_trade_debug = st.sidebar.checkbox(
        "NO_TRADE分岐を強制表示（テスト用）",
        value=False,
        help="代替ペアの動線テスト用。実運用ではOFF。",
    )
else:
    force_no_trade_debug = False


leverage = 25  # 固定

# 2. ポジション情報 (AI連動 & チャート表示用)
st.sidebar.markdown("---")
st.sidebar.subheader("📦 ポートフォリオ（複数）")

# --- ポートフォリオ概要（このツール内の管理用） ---
def _pair_head(_label: str) -> str:
    try:
        return (_label or "").split()[0].strip()
    except Exception:
        return ""

def _pair_to_ccy(_label: str):
    head = _pair_head(_label)
    if "/" in head and len(head) >= 7:
        base, quote = head.split("/")[:2]
        return base.strip()[:3], quote.strip()[:3]
    if "/" in (_label or ""):
        base, quote = (_label or "").split("/")[:2]
        return base.strip()[:3], quote.strip()[:3]
    return "UNK", "UNK"

def _portfolio_summary(active_positions: list):
    total_risk = 0.0
    counts = {}
    for p in active_positions or []:
        try:
            total_risk += float(p.get("risk_percent", p.get("risk", 0.0)) or 0.0)
        except Exception:
            pass
        pair = p.get("pair") or p.get("pair_label") or p.get("pair_name") or ""
        b, q = _pair_to_ccy(pair)
        counts[b] = counts.get(b, 0) + 1
        counts[q] = counts.get(q, 0) + 1
    return float(total_risk), counts

# --- 余力（証拠金）計算 & 推奨lots算出ヘルパー ---
_ONE_LOT_UNITS = 10000  # 1枚=1万通貨

def _infer_quote_ccy_from_label(pair_label: str) -> str:
    try:
        head = (pair_label or "").split()[0]
        if "/" in head:
            return head.split("/")[1].strip()[:3].upper()
    except Exception:
        pass
    return "JPY"

def _jpy_conversion_factor(quote_ccy: str, usd_jpy: float) -> float:
    q = (quote_ccy or "").upper()
    if q == "JPY":
        return 1.0
    if q == "USD":
        try:
            return float(usd_jpy) if float(usd_jpy) > 0 else 1.0
        except Exception:
            return 1.0
    # 想定外（例: EUR/GBPなど）は概算扱い
    return 1.0

def _required_margin_per_lot_jpy(pair_label: str, pair_price: float, usd_jpy: float, leverage: int = 25) -> float:
    """1枚（1万通貨）あたりの必要証拠金(JPY)。SBI固定値を優先、なければ概算。"""
    try:
        fixed = SBI_MARGIN_10K_JPY.get(pair_label)
        if fixed is not None and float(fixed) > 0:
            return float(fixed)
    except Exception:
        pass

    quote_ccy = _infer_quote_ccy_from_label(pair_label)
    conv = _jpy_conversion_factor(quote_ccy, usd_jpy)
    try:
        price = float(pair_price)
    except Exception:
        price = 0.0
    notional_jpy = price * _ONE_LOT_UNITS * conv
    try:
        lev = int(leverage) if int(leverage) > 0 else 25
    except Exception:
        lev = 25
    return notional_jpy / float(lev) if notional_jpy > 0 else 0.0

def _portfolio_margin_used_jpy(active_positions: list, usd_jpy: float, leverage: int = 25) -> float:
    total = 0.0
    for p in active_positions or []:
        try:
            pair = p.get("pair") or p.get("pair_label") or p.get("pair_name") or ""
            lots = float(p.get("lots", 0.0) or 0.0)
            if lots <= 0:
                continue
            price = float(p.get("entry_price", 0.0) or 0.0)
            m = _required_margin_per_lot_jpy(pair, price if price > 0 else usd_jpy, usd_jpy, leverage=leverage)
            if m > 0:
                total += m * lots
        except Exception:
            continue
    return float(total)

def _recommend_lots_int_and_risk(
    pair_label: str,
    entry: float,
    stop_loss: float,
    capital_jpy: float,
    risk_percent_target: float,
    usd_jpy: float,
    remaining_margin_jpy: float,
    leverage: int = 25,
):
    """2%ルールに沿って『実行可能な整数lots』と実質リスク%を返す。"""
    try:
        cap = float(capital_jpy)
    except Exception:
        cap = 0.0
    try:
        rp = float(risk_percent_target)
    except Exception:
        rp = 0.0
    try:
        e = float(entry)
        sl = float(stop_loss)
    except Exception:
        return 0, 0.0, 0.0, 0.0, 0.0, _infer_quote_ccy_from_label(pair_label)

    stop_w = abs(e - sl)
    quote_ccy = _infer_quote_ccy_from_label(pair_label)
    conv = _jpy_conversion_factor(quote_ccy, usd_jpy)
    loss_per_lot_jpy = stop_w * _ONE_LOT_UNITS * conv

    if cap <= 0 or rp <= 0 or loss_per_lot_jpy <= 0:
        return 0, 0.0, 0.0, float(loss_per_lot_jpy), float(stop_w), quote_ccy

    risk_amount = cap * (rp / 100.0)
    safe_lots_float = risk_amount / loss_per_lot_jpy if loss_per_lot_jpy > 0 else 0.0
    lots_int = int(math.floor(safe_lots_float + 1e-9))

    # 証拠金での上限（余力）
    req_margin_per_lot = _required_margin_per_lot_jpy(pair_label, e if e > 0 else usd_jpy, usd_jpy, leverage=leverage)
    if req_margin_per_lot > 0:
        try:
            rem = float(remaining_margin_jpy)
        except Exception:
            rem = 0.0
        max_lots_by_margin = int(math.floor(rem / req_margin_per_lot + 1e-9)) if rem > 0 else 0
        lots_int = min(lots_int, max_lots_by_margin)

    actual_risk_pct = (lots_int * loss_per_lot_jpy / cap * 100.0) if (cap > 0 and lots_int > 0) else 0.0
    return int(lots_int), float(actual_risk_pct), float(req_margin_per_lot), float(loss_per_lot_jpy), float(stop_w), quote_ccy

# =============================
# 固定1建（SBI最小1枚）前提の実損失・実質リスク計算ユーティリティ
#  - 2%は「目標」。固定1枚で2%に収まらない局面は上限（max_risk_percent_cap）で制御
#  - SL幅/最大損失/実質リスク%/必要資金（逆算）を必ず出す
# =============================

def _pip_size_from_quote(quote_ccy: str) -> float:
    q = (quote_ccy or "").upper()
    return 0.01 if q == "JPY" else 0.0001

def _stop_width_to_pips(stop_w: float, quote_ccy: str) -> float:
    ps = _pip_size_from_quote(quote_ccy)
    try:
        return float(stop_w) / ps if ps > 0 else 0.0
    except Exception:
        return 0.0

def _needed_capital_for_target(loss_per_lot_jpy: float, risk_percent_target: float) -> float:
    try:
        rp = float(risk_percent_target)
        l = float(loss_per_lot_jpy)
    except Exception:
        return 0.0
    if rp <= 0:
        return 0.0
    return l / (rp / 100.0)

def _select_lots_with_fixed_mode(
    pair_label: str,
    entry: float,
    stop_loss: float,
    capital_jpy: float,
    risk_percent_target: float,
    max_risk_percent_cap: float,
    fixed_1lot_mode: bool,
    usd_jpy: float,
    remaining_margin_jpy: float,
    leverage: int = 25,
):
    """
    返り値:
      {
        "lots": int,
        "risk_actual_pct": float,      # 選択lotsでの実質リスク%
        "risk_1lot_pct": float,        # 1枚での実質リスク%
        "loss_per_lot_jpy": float,     # 1枚の最大損失（JPY）
        "stop_w": float,               # SL幅（価格差）
        "sl_pips": float,              # SL幅（pips換算）
        "required_capital_for_target_1lot": float,  # 1枚を目標risk%に収めるのに必要な資金
        "req_margin_per_lot": float,
        "quote_ccy": str,
        "blocked": bool,
        "blocked_reason": str,
      }
    """
    lots_int, risk_pct_floor, req_margin_per_lot, loss_per_lot_jpy, stop_w, quote_ccy = _recommend_lots_int_and_risk(
        pair_label, entry, stop_loss, capital_jpy, risk_percent_target, usd_jpy, remaining_margin_jpy, leverage=leverage
    )

    cap = float(capital_jpy) if capital_jpy else 0.0
    risk_1lot_pct = (float(loss_per_lot_jpy) / cap * 100.0) if (cap > 0 and loss_per_lot_jpy > 0) else 0.0
    sl_pips = _stop_width_to_pips(stop_w, quote_ccy)
    required_cap = _needed_capital_for_target(loss_per_lot_jpy, risk_percent_target)

    # 証拠金上限
    try:
        rem_m = float(remaining_margin_jpy)
    except Exception:
        rem_m = 0.0
    max_lots_by_margin = int(math.floor(rem_m / req_margin_per_lot + 1e-9)) if (req_margin_per_lot > 0 and rem_m > 0) else 0

    # 実質リスク上限
    try:
        max_rp = float(max_risk_percent_cap)
    except Exception:
        max_rp = 0.0
    max_lots_by_cap = int(math.floor((cap * (max_rp / 100.0)) / loss_per_lot_jpy + 1e-9)) if (cap > 0 and loss_per_lot_jpy > 0 and max_rp > 0) else 0

    # 目標risk%に沿った推奨lots（整数/切り捨て）
    target_lots = int(lots_int)

    lots_sel = 0
    blocked_reason = ""

    if fixed_1lot_mode:
        # 1枚すら上限/余力で無理なら不可
        if max_lots_by_margin < 1:
            blocked_reason = "margin"
            lots_sel = 0
        elif max_lots_by_cap < 1:
            blocked_reason = "risk_cap"
            lots_sel = 0
        else:
            lots_sel = max(1, target_lots)  # targetが0でも1枚を検討
            lots_sel = min(lots_sel, max_lots_by_margin, max_lots_by_cap)
            lots_sel = max(1, lots_sel)
    else:
        lots_sel = target_lots
        lots_sel = min(lots_sel, max_lots_by_margin) if max_lots_by_margin > 0 else lots_sel
        lots_sel = min(lots_sel, max_lots_by_cap) if max_lots_by_cap > 0 else lots_sel
        if lots_sel < 1:
            blocked_reason = "target_lots_zero"
            lots_sel = 0

    risk_actual_pct = (lots_sel * loss_per_lot_jpy / cap * 100.0) if (cap > 0 and lots_sel > 0) else 0.0

    return {
        "lots": int(lots_sel),
        "risk_actual_pct": float(risk_actual_pct),
        "risk_1lot_pct": float(risk_1lot_pct),
        "loss_per_lot_jpy": float(loss_per_lot_jpy),
        "stop_w": float(stop_w),
        "sl_pips": float(sl_pips),
        "required_capital_for_target_1lot": float(required_cap),
        "req_margin_per_lot": float(req_margin_per_lot),
        "quote_ccy": str(quote_ccy),
        "blocked": (lots_sel < 1),
        "blocked_reason": blocked_reason,
        "max_lots_by_margin": int(max_lots_by_margin),
        "max_lots_by_cap": int(max_lots_by_cap),
    }

def _calc_rr(side: str, entry: float, take_profit: float, stop_loss: float) -> float:
    try:
        e = float(entry); tp = float(take_profit); sl = float(stop_loss)
    except Exception:
        return 0.0
    sw = abs(e - sl)
    if sw <= 0:
        return 0.0
    s = (side or "").upper()
    if s == "SHORT":
        return (e - tp) / sw
    return (tp - e) / sw

def _next_weekday_jst(now_dt: datetime, weekday: int) -> datetime:
    """
    weekday: Mon=0 ... Sun=6
    """
    d = now_dt
    days_ahead = (weekday - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)

def _derive_pullback_limit_candidate(pair_label: str, base_side: str, ctx: dict, df_src: pd.DataFrame):
    """
    押し目/戻りのLIMIT案（コード側で数値決定）
    """
    try:
        price = float(ctx.get("price", 0.0) or 0.0)
        atr = float(ctx.get("atr", 0.0) or 0.0)
        sma25 = float(ctx.get("sma25", price) or price)
        atr_avg60 = float(ctx.get("atr_avg60", atr) or atr)
    except Exception:
        return None
    if price <= 0 or atr <= 0:
        return None

    side = "SHORT" if str(base_side).upper() == "SHORT" else "LONG"
    # 直近構造（20日高安）
    try:
        recent_high20 = float(df_src["High"].tail(20).max())
        recent_low20 = float(df_src["Low"].tail(20).min())
    except Exception:
        recent_high20 = price
        recent_low20 = price

    compressed = (atr_avg60 > 0 and atr <= atr_avg60 * 0.95)

    if side == "LONG":
        entry = sma25 + 0.10 * atr if sma25 < price else price - 0.25 * atr
        entry = min(entry, price)  # 上から指すのはNG
        sl_struct = min(recent_low20, entry - 0.80 * atr)
        stop_loss = sl_struct - 0.05 * atr
        stop_w = max(1e-6, entry - stop_loss)
        tp_rr = entry + stop_w * 2.0
        # 壁（直近高値）の少し手前を優先（RRが崩れる場合はRR優先）
        tp_wall = recent_high20 - 0.10 * atr if recent_high20 > entry else tp_rr
        take_profit = tp_rr if _calc_rr("LONG", entry, tp_wall, stop_loss) < 1.2 else min(tp_rr, tp_wall)
    else:
        entry = sma25 - 0.10 * atr if sma25 > price else price + 0.25 * atr
        entry = max(entry, price)
        sl_struct = max(recent_high20, entry + 0.80 * atr)
        stop_loss = sl_struct + 0.05 * atr
        stop_w = max(1e-6, stop_loss - entry)
        tp_rr = entry - stop_w * 2.0
        tp_wall = recent_low20 + 0.10 * atr if recent_low20 < entry else tp_rr
        take_profit = tp_rr if _calc_rr("SHORT", entry, tp_wall, stop_loss) < 1.2 else max(tp_rr, tp_wall)

    rr = _calc_rr(side, entry, take_profit, stop_loss)

    notes = []
    if compressed:
        notes.append("ATR圧縮（浅いSLが成立しやすい局面）")
    else:
        notes.append("ATR圧縮ではない（浅いSLは刈られやすい可能性）")
    notes.append("構造（直近20日高安）を優先してSLを近づけたLIMIT案")

    return {
        "decision": "TRADE",
        "side": side,
        "entry": float(entry),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "horizon": "WEEK",
        "confidence": 0.55,
        "why": "固定1枚の現実に合わせ、押し目/戻りでSL幅を縮めるLIMIT案。",
        "notes": notes,
        "order_bundle": "IFD_OCO",
        "entry_type": "LIMIT",
        "entry_price_kind_jp": "指値",
        "bundle_hint_jp": "SBI: 指値(IFD-OCO)で放置（TP/SL同時）",
        "_rr": float(rr),
        "_candidate_kind": "PULLBACK_LIMIT",
    }

def _derive_hybrid_confirm_market_candidate(pair_label: str, base_side: str, ctx: dict, df_src: pd.DataFrame):
    """
    Hybrid案：ブレイク確認後に成行（ツールは“条件”として提示）
    """
    try:
        price = float(ctx.get("price", 0.0) or 0.0)
        atr = float(ctx.get("atr", 0.0) or 0.0)
        atr_avg60 = float(ctx.get("atr_avg60", atr) or atr)
    except Exception:
        return None
    if price <= 0 or atr <= 0:
        return None

    side = "SHORT" if str(base_side).upper() == "SHORT" else "LONG"
    try:
        recent_high20 = float(df_src["High"].tail(20).max())
        recent_low20 = float(df_src["Low"].tail(20).min())
    except Exception:
        recent_high20 = price
        recent_low20 = price

    # 構造SL（少し広めだが、STOPよりは近くなることが多い）
    if side == "LONG":
        entry = price
        stop_loss = min(recent_low20, price - 0.90 * atr) - 0.05 * atr
        stop_w = max(1e-6, entry - stop_loss)
        take_profit = entry + stop_w * 2.0
    else:
        entry = price
        stop_loss = max(recent_high20, price + 0.90 * atr) + 0.05 * atr
        stop_w = max(1e-6, stop_loss - entry)
        take_profit = entry - stop_w * 2.0

    rr = _calc_rr(side, entry, take_profit, stop_loss)
    compressed = (atr_avg60 > 0 and atr <= atr_avg60 * 0.95)
    notes = [
        "ブレイク確認（終値/足確定）後に成行で建てる“条件付き”案（固定1枚の代替）",
        "未約定リスク0を活かす（条件未達なら見送り）",
    ]
    if compressed:
        notes.append("ATR圧縮")
    return {
        "decision": "TRADE",
        "side": side,
        "entry": float(entry),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "horizon": "WEEK",
        "confidence": 0.50,
        "why": "固定1枚でブレイクの勢い確認後に入るHybrid（条件付き成行）案。",
        "notes": notes,
        "order_bundle": "IFD_OCO",
        "entry_type": "MARKET",
        "entry_price_kind_jp": "成行",
        "bundle_hint_jp": "SBI: 条件成立後に成行→IFD-OCO（手動実行）",
        "_rr": float(rr),
        "_candidate_kind": "HYBRID_CONFIRM_MARKET",
    }

def _decorate_time_rules(candidate: dict):
    """時間ルール（未約定キャンセル/建値化/時間損切）をnotesに追記（目安）。"""
    try:
        now = datetime.now(TOKYO)
    except Exception:
        now = datetime.now()
    wed = _next_weekday_jst(now, 2)  # Wed
    fri = _next_weekday_jst(now, 4)  # Fri
    if not isinstance(candidate, dict):
        return candidate
    notes = list(candidate.get("notes") or [])
    notes.append(f"未約定なら {wed.strftime('%Y-%m-%d')}（水）23:59 でキャンセル目安")
    notes.append("＋1R到達でSLを建値へ（事故回避）")
    notes.append(f"{fri.strftime('%Y-%m-%d')}（金）まで進まないなら時間損切り検討")
    candidate["notes"] = notes
    return candidate

def _evaluate_and_pick_candidates(
    pair_label: str,
    candidates: list,
    capital_jpy: float,
    risk_percent_target: float,
    max_risk_percent_cap: float,
    fixed_1lot_mode: bool,
    usd_jpy: float,
    remaining_margin_jpy: float,
    weekly_dd_cap_percent: float,
    active_positions: list,
    max_positions_per_currency: int,
    leverage: int = 25,
):
    evaluated = []
    for c in (candidates or []):
        if not isinstance(c, dict) or c.get("decision") != "TRADE":
            continue
        side = c.get("side", "LONG")
        e = float(c.get("entry") or 0.0)
        sl = float(c.get("stop_loss") or 0.0)
        tp = float(c.get("take_profit") or 0.0)
        if e <= 0 or sl <= 0 or tp <= 0:
            c["_eval"] = {"ok": False, "reason": "missing_price"}
            evaluated.append(c)
            continue

        rr = _calc_rr(side, e, tp, sl)
        sel = _select_lots_with_fixed_mode(
            pair_label=pair_label,
            entry=e,
            stop_loss=sl,
            capital_jpy=capital_jpy,
            risk_percent_target=risk_percent_target,
            max_risk_percent_cap=max_risk_percent_cap,
            fixed_1lot_mode=fixed_1lot_mode,
            usd_jpy=usd_jpy,
            remaining_margin_jpy=remaining_margin_jpy,
            leverage=leverage,
        )

        ok = True
        reasons = []

        if sel.get("blocked"):
            ok = False
            reasons.append(sel.get("blocked_reason") or "blocked")

        if rr < 1.2:
            ok = False
            reasons.append("rr_too_low")

        # 週DDキャップ
        if ok and not logic.can_open_under_weekly_cap(active_positions, float(sel.get("risk_actual_pct", 0.0)), float(weekly_dd_cap_percent)):
            ok = False
            reasons.append("weekly_dd_cap")

        # 通貨集中
        if ok and logic.violates_currency_concentration(pair_label, active_positions, int(max_positions_per_currency)):
            ok = False
            reasons.append("currency_concentration")

        c["_rr"] = float(rr)
        c["_eval"] = {
            "ok": bool(ok),
            "reasons": reasons,
            **sel,
        }
        evaluated.append(c)

    # pick
    valid = [c for c in evaluated if isinstance(c, dict) and c.get("_eval", {}).get("ok")]
    if valid:
        # リスク最小を優先（同率ならRR大）
        valid.sort(key=lambda x: (x["_eval"].get("risk_actual_pct", 999.0), -float(x.get("_rr", 0.0))))
        return valid[0], evaluated
    # none valid: return lowest risk as reference
    if evaluated:
        evaluated.sort(key=lambda x: (x.get("_eval", {}).get("risk_1lot_pct", 999.0), -float(x.get("_rr", 0.0))))
        return None, evaluated
    return None, evaluated



# =============================
# 📊 過去N週一括：TRADE回数見積もり（2/4/6%）
# =============================
@st.cache_data(show_spinner=False, ttl=1800)
def _batch_cached_indicator_df(pair_label: str, period: str):
    """PAIR_MAP の通貨ペアをまとめて使うための軽量キャッシュ（US10Yは未使用でOK）。"""
    try:
        sym = None
        try:
            sym = getattr(logic, "PAIR_MAP", {}).get(pair_label)
        except Exception:
            sym = None
        if not sym:
            try:
                if hasattr(logic, "_pair_label_to_symbol"):
                    sym = logic._pair_label_to_symbol(pair_label)
            except Exception:
                sym = None
        if not sym:
            return None

        raw = None
        try:
            if hasattr(logic, "_fetch_ohlc"):
                raw = logic._fetch_ohlc(sym, period=period, interval="1d")
            elif hasattr(logic, "_yahoo_chart"):
                raw = logic._yahoo_chart(sym, rng=period, interval="1d", ttl_sec=900)
        except Exception:
            raw = None

        if raw is None or getattr(raw, "empty", True):
            return None

        df_ind = logic.calculate_indicators(raw, None)
        if df_ind is None or getattr(df_ind, "empty", True):
            return None

        try:
            df_ind = df_ind.copy()
            df_ind.index = pd.to_datetime(df_ind.index)
        except Exception:
            pass
        return df_ind
    except Exception:
        return None


def _batch_slice_asof(df_ind: pd.DataFrame, as_of_date) -> pd.DataFrame:
    if df_ind is None or getattr(df_ind, "empty", True):
        return None
    try:
        ts = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        d = df_ind.loc[df_ind.index <= ts]
        if d is None or getattr(d, "empty", True):
            return None
        return d
    except Exception:
        return None


def _batch_ctx_from_df(df_ind_asof: pd.DataFrame) -> dict:
    lr = df_ind_asof.iloc[-1]
    try:
        atr_avg60 = float(df_ind_asof["ATR"].tail(60).mean())
    except Exception:
        atr_avg60 = float(lr.get("ATR", float("nan")))

    ctx = {
        "price": float(lr.get("Close", 0.0) or 0.0),
        "atr": float(lr.get("ATR", 0.0) or 0.0),
        "rsi": float(lr.get("RSI", 0.0) or 0.0),
        "sma25": float(lr.get("SMA_25", 0.0) or 0.0),
        "sma75": float(lr.get("SMA_75", 0.0) or 0.0),
        "atr_avg60": float(atr_avg60 or 0.0),
        # no_trade_gate は panel_* を参照するが、ここでは空でOK（数値条件のみで決まる）
        "panel_short": "",
        "panel_mid": "",
    }
    return ctx


def _batch_prev_weekday(from_date, weekday: int):
    d = from_date
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _batch_estimate_trade_frequency(
    n_weeks: int,
    weekday: int,
    scan_pairs: bool,
    use_ai: bool,
    capital_jpy: float,
    risk_percent_target: float,
    fixed_1lot_mode: bool,
    max_positions_per_currency: int,
    leverage: int,
    api_key: str,
):
    # 日付リスト（指定曜日の“その日終値まで”で判定）
    today = datetime.now(TOKYO).date()
    anchor = _batch_prev_weekday(today, int(weekday))
    dates = [anchor - timedelta(days=7 * i) for i in range(int(n_weeks))]
    earliest = dates[-1] if dates else anchor
    period = _choose_period_for_asof(earliest)

    # 対象ペア（FASTスキャン or USDJPY固定）
    try:
        all_pairs = list(getattr(logic, "PAIR_MAP", {}).keys())
    except Exception:
        all_pairs = ["USD/JPY (ドル円)"]
    if not all_pairs:
        all_pairs = ["USD/JPY (ドル円)"]
    pair_list = all_pairs if bool(scan_pairs) else ["USD/JPY (ドル円)"]

    # 指標DFを先読みキャッシュ
    df_map = {pl: _batch_cached_indicator_df(pl, period) for pl in set(pair_list + ["USD/JPY (ドル円)"])}

    rows = []
    counts = {2: 0, 4: 0, 6: 0}

    for d in dates:
        # USDJPY換算
        usd_jpy_rate = 150.0
        try:
            _usd_df = _batch_slice_asof(df_map.get("USD/JPY (ドル円)"), d)
            if _usd_df is not None and len(_usd_df) > 0:
                usd_jpy_rate = float(_usd_df.iloc[-1].get("Close", 150.0) or 150.0)
        except Exception:
            usd_jpy_rate = 150.0

        best_pair = None
        best_score = -1e18
        best_meta = {}

        # ペア選択（NO_TRADEで弾かれたら次へ）
        for pl in pair_list:
            df_full = df_map.get(pl)
            df_asof = _batch_slice_asof(df_full, d)
            if df_asof is None or len(df_asof) < 90:
                continue

            ctx = _batch_ctx_from_df(df_asof)

            # ゲート（代替ペアの“TRADE-able” と同じ考え方：trend_only + no_trade）
            try:
                nt, regime, nt_reasons = logic.no_trade_gate(ctx, "DEFENSIVE", force_defensive=True)
            except Exception:
                nt, regime, nt_reasons = True, "DEFENSIVE", ["no_trade_gate_error"]

            try:
                ok_trend, side_hint, trend_score, trend_reasons = logic.trend_only_gate(ctx)
            except Exception:
                ok_trend, side_hint, trend_score, trend_reasons = False, "NONE", None, ["trend_only_gate_error"]

            if (not ok_trend) or bool(nt) or (trend_score is None):
                continue

            score = float(trend_score)
            if score > best_score:
                best_score = score
                best_pair = pl
                best_meta = {
                    "date": d,
                    "pair": pl,
                    "side_hint": side_hint,
                    "trend_score": score,
                    "regime": regime,
                    "trend_reasons": trend_reasons,
                    "nt_reasons": nt_reasons,
                    "df_asof": df_asof,
                    "ctx": ctx,
                }

        if not best_pair:
            rows.append({
                "date": str(d),
                "pair": "",
                "side": "NONE",
                "trend_score": 0.0,
                "cap2_trade": False,
                "cap4_trade": False,
                "cap6_trade": False,
                "cap2_kind": "",
                "cap4_kind": "",
                "cap6_kind": "",
                "note": "trend_only_gate / no_trade_gate で全ペア見送り",
            })
            continue

        # ベース注文（AI or 数値フォールバック）
        ctx0 = dict(best_meta.get("ctx") or {})
        ctx0["trend_side_hint"] = best_meta.get("side_hint", "NONE")
        ctx0["trend_score"] = float(best_meta.get("trend_score", 0.0) or 0.0)

        market_regime = str(best_meta.get("regime") or "DEFENSIVE")
        regime_why = "batch_estimate"

        base = None
        if bool(use_ai) and (api_key or ""):
            try:
                # 最低限のctxで呼ぶ（失敗時は数値フォールバックへ）
                _ctx_ai = dict(ctx0)
                _ctx_ai["asof_date_jst"] = str(d)
                base = logic.get_ai_order_strategy(api_key, _ctx_ai, pair_name=best_pair, generation_policy="AUTO_HIERARCHY")
            except Exception:
                base = None

        if not isinstance(base, dict) or base.get("decision") != "TRADE":
            try:
                if hasattr(logic, "_build_numeric_fallback_order"):
                    base = logic._build_numeric_fallback_order(ctx0, market_regime, regime_why, pair_name=best_pair)
                else:
                    base = None
            except Exception:
                base = None

        if not isinstance(base, dict) or base.get("decision") != "TRADE":
            rows.append({
                "date": str(d),
                "pair": best_pair,
                "side": "NONE",
                "trend_score": float(best_meta.get("trend_score", 0.0) or 0.0),
                "cap2_trade": False,
                "cap4_trade": False,
                "cap6_trade": False,
                "cap2_kind": "",
                "cap4_kind": "",
                "cap6_kind": "",
                "note": "ベース注文が作れず（AI/数値とも不成立）",
            })
            continue

        # 3案生成（pullback LIMIT / breakout STOP / hybrid）
        df_asof = best_meta.get("df_asof")
        base_side = str(base.get("side") or best_meta.get("side_hint") or "LONG").upper()
        ctx_for_cands = dict(ctx0)

        cands = []
        try:
            c_break = dict(base)
            c_break["_candidate_kind"] = "BREAKOUT_STOP"
            cands.append(_decorate_time_rules(c_break))
        except Exception:
            pass
        try:
            c_pull = _derive_pullback_limit_candidate(best_pair, base_side, ctx_for_cands, df_asof)
            if isinstance(c_pull, dict):
                c_pull["_candidate_kind"] = "PULLBACK_LIMIT"
                cands.append(_decorate_time_rules(c_pull))
        except Exception:
            pass
        try:
            c_hyb = _derive_hybrid_confirm_market_candidate(best_pair, base_side, ctx_for_cands, df_asof)
            if isinstance(c_hyb, dict):
                c_hyb["_candidate_kind"] = "HYBRID_CONFIRM"
                cands.append(_decorate_time_rules(c_hyb))
        except Exception:
            pass

        def _pick_for_cap(capv: float):
            picked, evaluated = _evaluate_and_pick_candidates(
                pair_label=best_pair,
                candidates=cands,
                capital_jpy=float(capital_jpy),
                risk_percent_target=float(risk_percent_target),
                max_risk_percent_cap=float(capv),
                fixed_1lot_mode=bool(fixed_1lot_mode),
                usd_jpy=float(usd_jpy_rate),
                remaining_margin_jpy=float(capital_jpy),  # ノーポジ前提（余力=資金）
                weekly_dd_cap_percent=float(capv),        # 週DDキャップも同値で揃える
                active_positions=[],
                max_positions_per_currency=int(max_positions_per_currency),
                leverage=int(leverage),
            )
            if isinstance(picked, dict):
                return True, str(picked.get("_candidate_kind") or "")
            return False, ""

        t2, k2 = _pick_for_cap(2.0)
        t4, k4 = _pick_for_cap(4.0)
        t6, k6 = _pick_for_cap(6.0)

        if t2: counts[2] += 1
        if t4: counts[4] += 1
        if t6: counts[6] += 1

        rows.append({
            "date": str(d),
            "pair": best_pair,
            "side": base_side,
            "trend_score": float(best_meta.get("trend_score", 0.0) or 0.0),
            "cap2_trade": bool(t2),
            "cap4_trade": bool(t4),
            "cap6_trade": bool(t6),
            "cap2_kind": k2,
            "cap4_kind": k4,
            "cap6_kind": k6,
            "note": "",
        })

    df_out = pd.DataFrame(rows)
    summary = {
        "n_weeks": int(n_weeks),
        "weekday": int(weekday),
        "anchor": str(anchor),
        "period": str(period),
        "counts": counts,
    }
    return df_out, summary


# 実行トリガ（サイドバーのボタンでフラグが立つ）
if st.session_state.get("_batch_run_flag"):
    _p = st.session_state.get("_batch_params") or {}
    try:
        with st.spinner("過去N週を一括集計中…（数値ゲート＋3案生成）"):
            _df_b, _sum_b = _batch_estimate_trade_frequency(**_p)
        st.session_state["batch_freq_df"] = _df_b
        st.session_state["batch_freq_summary"] = _sum_b
        st.sidebar.success("✅ 一括集計が完了しました（下に結果が出ます）")
    except Exception as e:
        st.session_state["batch_freq_df"] = None
        st.session_state["batch_freq_summary"] = None
        st.sidebar.error(f"❌ 一括集計でエラー: {type(e).__name__}: {e}")
    finally:
        st.session_state["_batch_run_flag"] = False


# 結果表示
if st.session_state.get("batch_freq_df") is not None:
    _sum = st.session_state.get("batch_freq_summary") or {}
    _cnt = (_sum.get("counts") or {})
    with st.expander("📊 回数見積もり結果（過去N週 / 2%・4%・6%）", expanded=True):
        st.markdown(
            f"**対象:** 過去{_sum.get('n_weeks','?')}週 / 基準曜日={('水曜' if int(_sum.get('weekday',2))==2 else '月曜')}（anchor={_sum.get('anchor','')}）  \n"
            f"**集計（TRADEになった週数）:** 2%={_cnt.get(2,0)} / 4%={_cnt.get(4,0)} / 6%={_cnt.get(6,0)}  \n"
            f"（取得period={_sum.get('period','')}）"
        )
        st.dataframe(st.session_state.get("batch_freq_df"), use_container_width=True)
        st.caption("※ FASTスキャンはAIを呼ばず、trend_only_gate + no_trade_gate + 数値フォールバックで『TRADE可能か』の傾向を素早く見積もります。")

total_risk_pct, ccy_counts = _portfolio_summary(st.session_state.portfolio_positions)
remain_risk_pct = float(weekly_dd_cap_percent) - float(total_risk_pct)

# ✅ 余力（必要証拠金）: いま持っているポジションの合計必要証拠金と、口座余力の概算
try:
    _usd_jpy_est = float((st.session_state.get("quote") or (None, None))[0] or 0.0)
except Exception:
    _usd_jpy_est = 0.0
if _usd_jpy_est <= 0:
    _usd_jpy_est = 150.0  # クオート未取得時の保険（/USD換算を使う場合のみ）

used_margin_jpy = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, _usd_jpy_est, leverage=leverage)
remain_margin_jpy = float(capital) - float(used_margin_jpy)

st.sidebar.markdown(
    f"**現在の保有数:** {len(st.session_state.portfolio_positions)}  \n"
    f"**合計リスク%:** {total_risk_pct:.2f}%  \n"
    f"**残り枠:** {remain_risk_pct:.2f}%  \n"
    f"**総必要証拠金（概算）:** ¥{used_margin_jpy:,.0f}  \n"
    f"**余力（概算）:** ¥{remain_margin_jpy:,.0f}"
)

if remain_margin_jpy < 0:
    st.sidebar.error("❌ 余力がマイナスです（このツール内の概算）。ポジション登録内容（枚数/証拠金）を見直してください。")

# 通貨偏りの簡易表示
if ccy_counts:
    ccy_line = " / ".join([f"{k}:{v}" for k, v in sorted(ccy_counts.items(), key=lambda x: (-x[1], x[0]))])
    st.sidebar.caption("通貨露出（本ツール内）: " + ccy_line)


# --- 💾 状態保存 / 復元（端末側JSON） ---
# Streamlit Cloud はブラウザを閉じる・再接続すると st.session_state が初期化されます。
# そのため「ポートフォリオ（保有ポジション）」は端末側にJSONとして保存→必要時に復元する方式を用意します。
with st.sidebar.expander("💾 状態保存 / 復元（ポートフォリオ）", expanded=False):
    # 保存データ（最小：ポートフォリオのみ）
    _now_jst = datetime.now(TOKYO).strftime("%Y-%m-%d_%H%M%S")
    _state = {
        "schema": "AI-FX Analyzer 2026::portfolio_state_v1",
        "created_at_jst": _now_jst,
        "portfolio_positions": st.session_state.get("portfolio_positions", []) or [],
    }
    _bytes = json.dumps(_state, ensure_ascii=False, indent=2).encode("utf-8")
    _fname = f"ai_fx_portfolio_{_now_jst}.json"

    # Safari 対策（Secretsの FORCE_SAFARI_DOWNLOAD=true でリンク方式に切替）
    try:
        _force_safari = _parse_bool(st.secrets.get("FORCE_SAFARI_DOWNLOAD", False), False)
    except Exception:
        _force_safari = False

    if _force_safari:
        _b64 = base64.b64encode(_bytes).decode("ascii")
        _href = f"data:application/json;charset=utf-8;base64,{_b64}"
        st.markdown(f'🔗 <a href="{_href}" target="_blank">Safari: タップして保存（JSONを開く）</a>', unsafe_allow_html=True)
        st.caption("※開いたら共有→「ファイルに保存」でiCloud等へ保存してください（Safariはdownloadが失敗しやすいため）。")
    else:
        st.download_button("📥 ポートフォリオを保存（JSON）", data=_bytes, file_name=_fname, mime="application/json")

    st.markdown("---")
    _up = st.file_uploader("📤 復元用JSONを選択", type=["json"], accept_multiple_files=False)
    _restore = st.button("✅ このJSONで復元する", use_container_width=True)
    if _up is not None and _restore:
        try:
            _loaded = json.loads(_up.getvalue().decode("utf-8"))
            if _loaded.get("schema") != "AI-FX Analyzer 2026::portfolio_state_v1":
                st.error("このJSONは本ツールのポートフォリオ保存形式ではありません（schema不一致）。")
            else:
                _positions = _loaded.get("portfolio_positions", []) or []
                if not isinstance(_positions, list):
                    raise ValueError("portfolio_positions がlistではありません。")
                # 最小限のバリデーション
                _clean = []
                for p in _positions:
                    if not isinstance(p, dict):
                        continue
                    _clean.append(p)
                st.session_state.portfolio_positions = _clean
                st.success(f"復元しました（{len(_clean)}件）。")
                st.rerun()
        except Exception as e:
            st.error(f"復元に失敗しました: {e}")


# --- 追加フォーム（1つずつ登録） ---
pair_options = []
try:
    if hasattr(logic, "PAIR_MAP") and isinstance(logic.PAIR_MAP, dict):
        pair_options = list(logic.PAIR_MAP.keys())
except Exception:
    pair_options = []
if "USD/JPY (ドル円)" not in pair_options:
    pair_options = ["USD/JPY (ドル円)"] + pair_options

with st.sidebar.expander("➕ ポジションを追加", expanded=False):
    add_pair = st.selectbox("ペア", pair_options, index=0)
    add_dir = st.radio("方向", ["LONG（買い）", "SHORT（売り）"], horizontal=True)
    add_risk = st.number_input("このポジのリスク（%）", min_value=0.0, max_value=10.0, value=float(risk_percent), step=0.1)
    add_lots = st.number_input("枚数（1枚=1万通貨）", min_value=0.0, max_value=200.0, value=1.0, step=1.0)
    add_entry = st.number_input("建値（価格）", value=0.0, format="%.6f")
    add_sl = st.number_input("損切（SL）※任意", value=0.0, format="%.6f")
    add_tp = st.number_input("利確（TP）※任意", value=0.0, format="%.6f")
    add_horizon = st.selectbox("想定期間", ["WEEK（1週間）", "MONTH（1か月）"], index=0)
    if st.button("追加する", key="btn_add_position_manual"):
        # ✅ 手動入力でも risk% は必ず再計算（入力値は参考扱い）
        rp_auto = float(add_risk)
        try:
            if float(add_entry) > 0 and float(add_sl) > 0 and float(add_lots) > 0 and float(capital) > 0:
                try:
                    usd_jpy_now = float(current_rate)
                except Exception:
                    usd_jpy_now = float(st.session_state.get('usd_jpy_current', 0.0) or 0.0)
                q = _infer_quote_ccy_from_label(add_pair)
                conv = _jpy_conversion_factor_from_quote(q, usd_jpy_now)
                loss_per_lot = abs(float(add_entry) - float(add_sl)) * 10000.0 * float(conv)
                max_loss = loss_per_lot * float(add_lots)
                rp_auto = float(max_loss) / float(capital) * 100.0
        except Exception:
            pass

        st.session_state.portfolio_positions.append({
            "pair": add_pair,
            "direction": "LONG" if "LONG" in add_dir else "SHORT",
            "risk_percent": float(rp_auto),
            "lots": float(add_lots),
            "entry_price": float(add_entry),
            "stop_loss": float(add_sl) if add_sl else 0.0,
            "take_profit": float(add_tp) if add_tp else 0.0,
            "horizon": "MONTH" if "MONTH" in add_horizon else "WEEK",
            "entry_time": datetime.now(TOKYO).isoformat(),
        })
        st.success("追加しました。")
        st.rerun()

# --- 一覧（編集/削除） ---
with st.sidebar.expander("📋 一覧（編集/削除）", expanded=False):
    if st.session_state.portfolio_positions:
        _dfp = pd.DataFrame(st.session_state.portfolio_positions)
        if "lots" not in _dfp.columns:
            _dfp["lots"] = 0.0
        # 表示列を整える
        cols = [c for c in ["pair","direction","risk_percent","lots","entry_price","stop_loss","take_profit","horizon","entry_time"] if c in _dfp.columns]
        _dfp = _dfp[cols]
        edited = st.data_editor(
            _dfp,
            use_container_width=True,
            num_rows="dynamic",
            key="portfolio_editor",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("反映", key="btn_apply_portfolio_editor"):
                recs = []
                for r in edited.to_dict(orient="records"):
                    if not isinstance(r, dict):
                        continue
                    pair = str(r.get("pair", "") or "").strip()
                    if not pair:
                        continue

                    d_raw = str(r.get("direction", "LONG") or "").upper()
                    direction = "SHORT" if ("SHORT" in d_raw or "売" in d_raw) else "LONG"

                    h_raw = str(r.get("horizon", "WEEK") or "").upper()
                    horizon = "MONTH" if ("MONTH" in h_raw or "1か月" in h_raw) else "WEEK"

                    def _to_float(v, default=0.0):
                        try:
                            return float(v)
                        except Exception:
                            return float(default)

                    rp_auto = _to_float(r.get("risk_percent", 0.0), 0.0)
                    try:
                        ep = _to_float(r.get("entry_price", 0.0), 0.0)
                        slp = _to_float(r.get("stop_loss", 0.0), 0.0)
                        lotsv = _to_float(r.get("lots", 0.0), 0.0)
                        if ep > 0 and slp > 0 and lotsv > 0 and float(capital) > 0:
                            try:
                                usd_jpy_now = float(current_rate)
                            except Exception:
                                usd_jpy_now = float(st.session_state.get('usd_jpy_current', 0.0) or 0.0)
                            q = _infer_quote_ccy_from_label(pair)
                            conv = _jpy_conversion_factor_from_quote(q, usd_jpy_now)
                            loss_per_lot = abs(ep - slp) * 10000.0 * float(conv)
                            max_loss = loss_per_lot * lotsv
                            rp_auto = float(max_loss) / float(capital) * 100.0
                    except Exception:
                        pass

                    recs.append({
                        "pair": pair,
                        "direction": direction,
                        "risk_percent": float(rp_auto),
                        "lots": _to_float(r.get("lots", 0.0), 0.0),
                        "entry_price": _to_float(r.get("entry_price", 0.0), 0.0),
                        "stop_loss": _to_float(r.get("stop_loss", 0.0), 0.0),
                        "take_profit": _to_float(r.get("take_profit", 0.0), 0.0),
                        "horizon": horizon,
                        "entry_time": r.get("entry_time") or datetime.now(TOKYO).isoformat(),
                    })

                st.session_state.portfolio_positions = recs
                st.success("反映しました。")
                st.rerun()
        with c2:
            del_idx = st.number_input("削除行（0始まり）", min_value=0, max_value=max(0, len(st.session_state.portfolio_positions)-1), value=0, step=1)
            if st.button("削除", key="btn_delete_portfolio_row"):
                try:
                    st.session_state.portfolio_positions.pop(int(del_idx))
                    st.success("削除しました。")
                    st.rerun()
                except Exception:
                    st.error("削除に失敗しました。")
        with c3:
            if st.button("全クリア", key="btn_clear_portfolio"):
                st.session_state.portfolio_positions = []

                st.warning("全ポジションをクリアしました。")
                st.rerun()
    else:
        st.caption("まだポジションは登録されていません。")

# --- 互換用: 既存ロジックが参照する単一保有（USD/JPY）の入力値をポートフォリオから抽出 ---
entry_price = 0.0
trade_type = "買い (Long)"
try:
    for p in reversed(st.session_state.portfolio_positions or []):
        head = ((p.get("pair") or "").split()[0] if p.get("pair") else "")
        if head == "USD/JPY":
            entry_price = float(p.get("entry_price") or 0.0)
            trade_type = "買い (Long)" if str(p.get("direction","")).upper() == "LONG" else "売り (Short)"
            break
except Exception:
    pass

# --- クオート更新 ---
st.sidebar.markdown("---")
if BACKTEST_AS_OF_TS is None:
    if st.sidebar.button("🔄 最新クオート更新"):
        st.session_state.quote = logic.get_latest_quote("JPY=X")
        st.rerun()
    q_price, q_time = st.session_state.quote
else:
    # 過去検証中は、クオート更新を無効化して「その日までの終値」を現在値として扱う
    st.sidebar.caption(f"🕰 過去検証中: {BACKTEST_AS_OF_DATE}（JST）までのデータで固定 / クオート更新は無効")
    q_price, q_time = None, None

# --- データ取得と計算 ---
usdjpy_raw, us10y_raw = logic.get_market_data(period=(BACKTEST_FETCH_PERIOD or "1y"))
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
# 過去検証モード: 評価日までにデータを固定（未来データを切り捨て）
if BACKTEST_AS_OF_TS is not None and df is not None and not df.empty:
    try:
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index <= BACKTEST_AS_OF_TS]
    except Exception:
        pass
strength = logic.get_currency_strength()

# 最新レートの補完ロジック (モバイル・時間対応)
if df is not None and not df.empty:
    if BACKTEST_AS_OF_TS is not None:
        # 過去検証時はDF末尾（評価日までの終値）を現在値として扱う
        q_price = float(df["Close"].iloc[-1])
        try:
            _idx = df.index[-1]
            _ts = pd.Timestamp(_idx)
            q_time = _ts.tz_localize("Asia/Tokyo") if getattr(_ts, "tzinfo", None) is None else _ts.tz_convert("Asia/Tokyo")
        except Exception:
            q_time = None
        st.session_state.quote = (q_price, q_time)
    last_idx = df.index[-1]
    # q_priceが未取得ならDF末尾を使用
    if q_price is None:
        q_price = float(df["Close"].iloc[-1])

    # 時間が未取得ならDFインデックスをJST変換
    if q_time is None:
        if getattr(last_idx, "tzinfo", None) is None:
            # UTCと仮定してJSTへ変換
            q_time = last_idx.tz_localize("UTC").tz_convert("Asia/Tokyo")
        else:
            q_time = last_idx.tz_convert("Asia/Tokyo")

if df is None or df.empty:
    st.error("データが取得できませんでした。logic.pyを確認してください。")
    st.stop()

# 最新レートが取得できない場合のバックアップ
current_rate = q_price if q_price else df["Close"].iloc[-1]
st.session_state['usd_jpy_current'] = float(current_rate)

# 軸同期のためにインデックスを正規化
df.index = pd.to_datetime(df.index)

# AI予想ライン反映 (機能実装)
st.sidebar.markdown("---")
if st.sidebar.button("📈 AI予想ライン反映"):
    if api_key:
        with st.spinner("AI予想を取得中..."):
            last_row = df.iloc[-1]
            context = {"price": last_row["Close"], "rsi": last_row["RSI"], "atr": last_row["ATR"]}
            st.session_state.ai_range = logic.get_ai_range(api_key, context)
            st.rerun()
    else:
        st.warning("Gemini API Key を入力してください。")

# 診断(diag)生成
try:
    diag = logic.judge_condition(df)
except Exception as e:
    diag = None
    st.error(f"judge_conditionでエラー: {e}")

# 45日表示設定
last_date = df.index[-1]
start_view = last_date - timedelta(days=45)
df_view = df.loc[df.index >= start_view]
y_min_view = float(df_view["Low"].min())
y_max_view = float(df_view["High"].max())

chart_pair_label = st.session_state.get("chart_pair_label") or "USD/JPY (ドル円)"  # ✅チャート対象（USD/JPY or 代替）
# ✅ AI予想ラインがチャート範囲外に出ても表示されるよう、Y軸レンジに予想高安を含める
if (chart_pair_label == "USD/JPY (ドル円)") and st.session_state.ai_range:
    try:
        _hi, _lo = st.session_state.ai_range
        y_min_view = min(y_min_view, float(_lo))
        y_max_view = max(y_max_view, float(_hi))
    except Exception:
        pass

# 最新レート表示 (スマホ対応・時刻フォーマット)
if q_price is not None:
    fmt_time = q_time.strftime('%Y-%m-%d %H:%M') if q_time else "時刻不明"
    st.markdown(
        f"### 💱 最新USD/JPY: **{float(q_price):.3f} 円** "
        f"<span style='color:#888; font-size:0.8em; display:block'>(更新: {fmt_time} JST)</span>",
        unsafe_allow_html=True,
    )

# --- 1. 診断パネル ---
if diag is not None:
    col_short, col_mid = st.columns(2)
    with col_short:
        st.markdown(f"""
            <div style="background-color:{diag['short']['color']}; padding:15px; border-radius:12px; border:1px solid #ddd; min-height:180px;">
                <h3 style="color:#333; margin:0; font-size:16px;">📅 1週間スパン（短期勢い）</h3>
                <h2 style="color:#333; margin:5px 0; font-size:22px;">{diag['short']['status']}</h2>
                <p style="color:#555; font-size:13px; line-height:1.5;">{diag['short']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col_mid:
        st.markdown(f"""
            <div style="background-color:{diag['mid']['color']}; padding:15px; border-radius:12px; border:1px solid #ddd; min-height:180px;">
                <h3 style="color:#333; margin:0; font-size:16px;">🗓️ 1ヶ月スパン（中期トレンド）</h3>
                <h2 style="color:#333; margin:5px 0; font-size:22px;">{diag['mid']['status']}</h2>
                <p style="color:#555; font-size:13px; line-height:1.5;">{diag['mid']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 2. 経済アラート & スリップロス推奨 ---
col_alert, col_slip = st.columns(2)
with col_alert:
    if diag is not None:
        try:
            if diag["short"]["status"] == "勢い鈍化・調整" or df["ATR"].iloc[-1] > df["ATR"].mean() * 1.5:
                st.warning("⚠️ **【警戒】ボラティリティ上昇中**")
        except Exception:
            pass
with col_slip:
    current_atr = df["ATR"].iloc[-1]
    rec_slip = max(3, int(current_atr * 10))
    st.info(f"🛡️ 推奨スリップロス: **{rec_slip} pips** (ATR:{current_atr:.3f})")

# --- 3. メインチャート (AI予想ライン & ポジション表示対応) ---

# ✅ チャート表示対象（USD/JPY or 代替ペア）を切替
df_chart = df
chart_title = "USD/JPY & AI予想"

if chart_pair_label != "USD/JPY (ドル円)":
    df_alt_chart = _get_df_for_pair(chart_pair_label, us10y_raw)
    if df_alt_chart is not None and not df_alt_chart.empty:
        df_chart = df_alt_chart
        chart_title = f"{chart_pair_label}（代替チャート）"
    else:
        st.warning("⚠️ 代替ペアのチャートデータ取得に失敗したため、USD/JPYへ戻しました。")
        chart_pair_label = "USD/JPY (ドル円)"
        st.session_state.chart_pair_label = chart_pair_label
        df_chart = df
        chart_title = "USD/JPY & AI予想"

# チャート用の表示レンジ（45日）
chart_last_date = df_chart.index[-1]
chart_start_view = chart_last_date - timedelta(days=45)
df_chart_view = df_chart.loc[df_chart.index >= chart_start_view]
y_min_view_chart = float(df_chart_view["Low"].min())
y_max_view_chart = float(df_chart_view["High"].max())

st.caption(f"📈 表示チャート: **{chart_pair_label}**")
if chart_pair_label != "USD/JPY (ドル円)":
    if st.button("↩️ USD/JPYチャートに戻す", key="btn_chart_back_usdjpy"):
        st.session_state.chart_pair_label = "USD/JPY (ドル円)"
        st.session_state.chart_overlay = None
        st.rerun()


fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=(chart_title, "米国債10年物利回り"), row_heights=[0.7, 0.3]
)
fig_main.add_trace(go.Candlestick(x=df_chart.index, open=df_chart["Open"], high=df_chart["High"], low=df_chart["Low"], close=df_chart["Close"], name="価格"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df_chart.index, y=df_chart["SMA_5"], name="5日線", line=dict(color="#00ff00", width=1.5)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df_chart.index, y=df_chart["SMA_25"], name="25日線", line=dict(color="orange", width=2)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df_chart.index, y=df_chart["SMA_75"], name="75日線", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

# ★ AI予想ライン表示機能 (赤・緑点線)
if (chart_pair_label == "USD/JPY (ドル円)") and st.session_state.ai_range:
    high_val, low_val = st.session_state.ai_range
    view_x = [chart_start_view, chart_last_date]
    fig_main.add_trace(go.Scatter(x=view_x, y=[high_val, high_val], name=f"予想最高:{high_val:.2f}", line=dict(color="red", width=2, dash="dash")), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=view_x, y=[low_val, low_val], name=f"予想最低:{low_val:.2f}", line=dict(color="green", width=2, dash="dash")), row=1, col=1)

# ★ ポートフォリオ連動表示（USD/JPYのみを本チャートに重ねる）
try:
    for p in st.session_state.portfolio_positions or []:
        pair = (p.get("pair") or "").strip()
        head = (pair.split()[0] if pair else "")
        if head != "USD/JPY":
            continue
        ep = float(p.get("entry_price") or 0.0)
        if ep <= 0:
            continue
        direction = (p.get("direction") or "").upper()
        line_color = "blue" if direction == "LONG" else "magenta"
        pos_name = f"{pair} 保有:{ep:.2f}"
        fig_main.add_trace(
            go.Scatter(
                x=[chart_start_view, chart_last_date],
                y=[ep, ep],
                name=pos_name,
                line=dict(color=line_color, width=2, dash="dashdot"),
            ),
            row=1, col=1
        )
except Exception:
    pass



# ✅ 注文戦略（Entry/TP/SL）をチャートに重ね表示（代替ペア切替対応）
overlay = st.session_state.get("chart_overlay")
if isinstance(overlay, dict) and _normalize_pair_label(overlay.get("pair_label", "")) == _normalize_pair_label(chart_pair_label):
    try:
        e = float(overlay.get("entry", 0))
        tp = float(overlay.get("tp", 0))
        sl = float(overlay.get("sl", 0))
        view_x2 = [chart_start_view, chart_last_date]
        fig_main.add_trace(go.Scatter(x=view_x2, y=[e, e], name=f"Entry:{e:.3f}", line=dict(color="yellow", width=2, dash="dot")), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=view_x2, y=[tp, tp], name=f"TP:{tp:.3f}", line=dict(color="lime", width=2, dash="dot")), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=view_x2, y=[sl, sl], name=f"SL:{sl:.3f}", line=dict(color="orange", width=2, dash="dot")), row=1, col=1)
    except Exception:
        pass

fig_main.add_trace(go.Scatter(x=df_chart.index, y=df_chart["US10Y"], name="米10年債", line=dict(color="cyan"), showlegend=True), row=2, col=1)

fig_main.update_xaxes(range=[chart_start_view, chart_last_date], row=1, col=1)
fig_main.update_xaxes(range=[chart_start_view, chart_last_date], matches='x', row=2, col=1)
fig_main.update_yaxes(range=[y_min_view_chart * 0.998, y_max_view_chart * 1.002], autorange=False, row=1, col=1)
fig_main.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(r=10, l=10))
st.plotly_chart(fig_main, use_container_width=True)

# --- 4. RSI & SBI仕様ロット計算機 ---
st.subheader("🛠️ SBI FX ロット計算機 (1万通貨単位)")
col_rsi, col_calc = st.columns([1, 1.5])

with col_rsi:
    st.markdown(f"**📉 RSI: {float(df['RSI'].iloc[-1]):.2f}**")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff5722")))
    fig_rsi.add_hline(y=70, line=dict(color="#00ff00", dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(color="#ff0000", dash="dash"))
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(height=200, template="plotly_dark", yaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_rsi, use_container_width=True)


with col_calc:
    one_lot_units = 10000

    # ✅「直近に生成した注文書（USD/JPY or 代替ペア）」に追従するロット計算
    calc_pair = st.session_state.get("calc_pair_label") or "USD/JPY (ドル円)"
    calc_ctx = st.session_state.get("calc_ctx") or {}
    calc_strategy = st.session_state.get("calc_strategy") or {}

    # 価格（対象ペア）
    try:
        pair_price = float(calc_ctx.get("price", current_rate))
    except Exception:
        pair_price = float(current_rate)

    # 通貨ペアのクオート通貨を推定（JPY or USD）
    head = (calc_pair or "").split()[0]
    quote_ccy = "JPY"
    try:
        if "/" in head:
            quote_ccy = head.split("/")[1].strip()[:3].upper()
    except Exception:
        quote_ccy = "JPY"

    # 口座通貨JPYへの換算係数（JPY建てなら1、USD建てならUSDJPYで換算）
    usd_jpy = float(current_rate)  # USD/JPYの現在値（JPY=X）
    if quote_ccy == "JPY":
        conv = 1.0
        unit_label = "円"
        step = 0.1
        default_manual = 0.5
    elif quote_ccy == "USD":
        conv = usd_jpy
        unit_label = "USD"
        step = 0.0005
        default_manual = 0.005
    else:
        # ここに来るのは今のPAIR_MAPではほぼ無い想定（念のため）
        conv = 1.0
        unit_label = quote_ccy
        step = 0.0005
        default_manual = 0.005
        st.warning(f"⚠️ ロット計算: クオート通貨が {quote_ccy} のため、厳密なJPY換算ができません。概算表示になります。")

    # 注文書がTRADEなら「SL幅（価格差）」を自動採用（＝手入力なしで2%判定できる）
    auto_stop_width = None
    try:
        if isinstance(calc_strategy, dict) and (calc_strategy.get("decision") == "TRADE"):
            e = float(calc_strategy.get("entry", 0.0) or 0.0)
            sl = float(calc_strategy.get("stop_loss", 0.0) or 0.0)
            if e > 0 and sl > 0:
                auto_stop_width = abs(e - sl)
    except Exception:
        auto_stop_width = None

    # ✅ いまのポートフォリオ合計の必要証拠金/余力（概算）
    used_margin_jpy_now = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy, leverage=leverage)
    remain_margin_jpy_now = float(capital) - float(used_margin_jpy_now)
    if remain_margin_jpy_now < 0:
        remain_margin_jpy_now = 0.0

    st.markdown("#### 🧮 リスク管理 vs 全力シミュレーション")
    st.caption(
        f"対象ペア: **{calc_pair}**（クオート通貨: {quote_ccy}） / 許容DD: {risk_percent:.1f}% / 週DDキャップ: {weekly_dd_cap_percent:.1f}%  |  "
        f"総必要証拠金: ¥{used_margin_jpy_now:,.0f} / 余力: ¥{remain_margin_jpy_now:,.0f}"
    )

    # 損切幅（価格差）: 注文書があれば自動、なければ手入力（USD/JPY基準の初期値）
    default_stop = float(auto_stop_width) if auto_stop_width is not None else float(default_manual)
    stop_w = st.number_input(
        f"想定損切幅（価格差: {unit_label}）※ 注文書がTRADEならSL幅を自動で初期値に設定",
        value=default_stop,
        step=step,
        format="%.6f" if quote_ccy == "USD" else "%.3f",
        key="lot_stop_width_input"
    )

    # 1枚（=1万通貨）の想定損失額（JPY換算）
    loss_per_lot_jpy = abs(float(stop_w)) * one_lot_units * float(conv)

    # 証拠金（JPY換算）
    # ✅SBIの「必要証拠金（1万通貨あたり）」固定値がある場合はそれを優先
    _fixed_margin = None
    try:
        _fixed_margin = float(SBI_MARGIN_10K_JPY.get(calc_pair)) if isinstance(SBI_MARGIN_10K_JPY, dict) else None
    except Exception:
        _fixed_margin = None

    if _fixed_margin and _fixed_margin > 0:
        required_margin_per_lot = float(_fixed_margin)
        margin_mode = "SBI固定"
    else:
        # フォールバック（概算）: 名目金額/レバレッジ
        notional_jpy = float(pair_price) * one_lot_units * float(conv)
        required_margin_per_lot = notional_jpy / leverage if leverage else notional_jpy
        margin_mode = "概算"

    max_lots = int(remain_margin_jpy_now / required_margin_per_lot) if required_margin_per_lot > 0 else 0

    if stop_w and float(stop_w) > 0:
        risk_amount = capital * (risk_percent / 100.0)
        safe_lots = (risk_amount / loss_per_lot_jpy) if loss_per_lot_jpy > 0 else 0.0

        c1, c2 = st.columns(2)
        with c1:
            st.error(f"""
            **💀 限界 (レバレッジ{leverage}倍)**
            - 対象ペア価格: {pair_price:.6f} ({unit_label})
            - 必要証拠金/枚({margin_mode}): ¥{required_margin_per_lot:,.0f}
            - **最大発注可能数: {max_lots} 枚**
            """)
        with c2:
            st.success(f"""
            **🛡️ 推奨 (安全重視: {risk_percent:.1f}%)**
            - 許容損失額: ¥{risk_amount:,.0f}
            - 1枚の想定損失: ¥{loss_per_lot_jpy:,.0f}
            - **推奨発注数量: {safe_lots:.2f} 枚**
            """)

        if safe_lots > max_lots and max_lots > 0:
            st.warning("⚠️ 注意：リスク許容内でも証拠金不足で発注できない可能性があります。")
        elif safe_lots < 0.1:
            st.warning("⚠️ 注意：損切幅が広すぎる/資金が小さいため、この条件では取引推奨外です（あなたの2%ルールに従うなら見送りが安全）。")


# --- 5. 通貨強弱 ---
if strength is not None and not strength.empty:
    st.subheader("📊 通貨強弱（1ヶ月）")
    fig_str = go.Figure()
    color_map = {"日本円": "#ff0000", "豪ドル": "#00ff00", "ユーロ": "#a020f0", "英ポンド": "#c0c0c0", "米ドル": "#ffd700"}
    for col in strength.columns:
        fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col, line=dict(color=color_map.get(col))))
    fig_str.update_layout(height=350, template="plotly_dark", showlegend=True, margin=dict(r=10, l=10))
    st.plotly_chart(fig_str, use_container_width=True)

# --- 6. AI実戦運用エリア (タブ化・ポジション連動連携) ---
st.divider()
st.subheader("🤖 AI軍師・実戦運用本部")

# AIに渡すデータ (ポジション情報追加)
ctx = {
    "price": float(df["Close"].iloc[-1]),
    "us10y": float(df["US10Y"].iloc[-1]) if pd.notna(df["US10Y"].iloc[-1]) else 0.0,
    "atr": float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else 0.0,
    "sma_diff": float(df["SMA_DIFF"].iloc[-1]) if pd.notna(df["SMA_DIFF"].iloc[-1]) else 0.0,
    "rsi": float(df["RSI"].iloc[-1]) if pd.notna(df["RSI"].iloc[-1]) else 50.0,
    "sma25": float(df["SMA_25"].iloc[-1]) if ("SMA_25" in df.columns and pd.notna(df["SMA_25"].iloc[-1])) else float(df["Close"].iloc[-1]),
    "sma75": float(df["SMA_75"].iloc[-1]) if ("SMA_75" in df.columns and pd.notna(df["SMA_75"].iloc[-1])) else float(df["Close"].iloc[-1]),
    "atr_avg60": float(df["ATR"].tail(60).mean()) if ("ATR" in df.columns and df["ATR"].tail(60).notna().any()) else float(df["ATR"].iloc[-1]) if ("ATR" in df.columns and pd.notna(df["ATR"].iloc[-1])) else 0.0,
    "current_time": q_time.strftime("%H:%M") if q_time else "不明",
    "is_gotobi": (BACKTEST_AS_OF_DATE.day if BACKTEST_AS_OF_DATE else datetime.now(TOKYO).day) in [5, 10, 15, 20, 25, 30],
    "capital": capital,
    "active_positions": st.session_state.portfolio_positions,
    "entry_price": entry_price,
    "as_of_date": str(BACKTEST_AS_OF_DATE) if BACKTEST_AS_OF_DATE else "LIVE",
    "trade_type": trade_type
}

tab1, tab2, tab3 = st.tabs(["📊 詳細レポート", "📝 注文戦略(日/週)", "💰 長期/ポートフォリオ"])

with tab1:
    if st.button("✨ レポート生成 (五十日/選挙対応)"):
        if api_key:
            with st.spinner("FP1級AIが分析中..."):
                report = logic.get_ai_analysis(api_key, ctx)
                st.session_state.last_ai_report = report
                st.markdown(report)
        else:
            st.warning("Gemini API Key を入力してください。")


with tab2:
    # --- 注文命令書（週1運用の中核） ---
    col_make_a, col_make_b = st.columns(2)
    with col_make_a:
        btn_make_auto = st.button(
            "📝 注文命令書作成（自動階層化・推奨）",
            key="btn_make_order_auto",
            help="AI生成→（失敗時）AI再生成→（さらに失敗時）数値フォールバックの順で、迷わず最終案を出します。"
        )
    with col_make_b:
        btn_make_strict = st.button(
            "🧠 注文命令書作成（AI厳格）",
            key="btn_make_order_strict",
            help="AIの出力が不正/失敗した場合は『見送り』で止めます（安全最優先）。"
        )

    if btn_make_auto or btn_make_strict:
        gen_policy = "AUTO_HIERARCHY" if btn_make_auto else "AI_STRICT"
        if api_key:
            if not st.session_state.last_ai_report:
                st.warning("先に『詳細レポート』を生成してください。")
            else:
                with st.spinner("資金管理・スリップロス計算中..."):
                    ctx["last_report"] = st.session_state.last_ai_report
                    ctx["panel_short"] = diag['short']['status'] if diag else "不明"
                    ctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                    base_strategy = logic.get_ai_order_strategy(api_key, ctx, generation_policy=gen_policy)
                    # --- シャドー比較（Gemini vs GPT）: Gemini出力(base_strategy)とOpenAI出力を同条件で比較 ---
                    if shadow_enabled:
                        if not openai_api_key_shadow:
                            st.session_state.last_shadow_base = base_strategy if isinstance(base_strategy, dict) else None
                            st.session_state.last_shadow_openai = None
                            st.session_state.last_shadow_diff = None
                        elif not hasattr(logic, "get_ai_order_strategy_shadow_openai"):
                            st.session_state.last_shadow_base = base_strategy if isinstance(base_strategy, dict) else None
                            st.session_state.last_shadow_openai = {"error": "logic.py に get_ai_order_strategy_shadow_openai がありません"}
                            st.session_state.last_shadow_diff = None
                        else:
                            try:
                                shadow_openai = logic.get_ai_order_strategy_shadow_openai(
                                    openai_api_key_shadow,
                                    ctx,
                                    openai_model=openai_model_shadow,
                                    override_mode=str(gen_policy),
                                    override_reason="order_strategy",
                                )
                            except Exception as _se:
                                shadow_openai = {"error": str(_se)}
                            st.session_state.last_shadow_base = base_strategy if isinstance(base_strategy, dict) else None
                            st.session_state.last_shadow_openai = shadow_openai if isinstance(shadow_openai, dict) else None
                            st.session_state.last_shadow_diff = _shadow_diff(st.session_state.last_shadow_base or {}, st.session_state.last_shadow_openai or {})
                            _write_jsonl(
                                f"logs/shadow_compare_{_today_tokyo_str()}.jsonl",
                                {
                                    "ts": datetime.now(TOKYO).isoformat(),
                                    "pair": "USD/JPY (ドル円)",
                                    "policy": str(gen_policy),
                                    "ctx": {k: ctx.get(k) for k in ["price", "atr", "atr_avg60", "rsi", "sma25", "sma75", "sma_diff", "us10y", "current_time"] if k in ctx},
                                    "gemini": st.session_state.last_shadow_base,
                                    "openai": st.session_state.last_shadow_openai,
                                    "diff": st.session_state.last_shadow_diff,
                                },
                            )
                    chosen = base_strategy
                    try:
                        if prefer_pullback_limit and isinstance(base_strategy, dict) and base_strategy.get('decision') == 'TRADE':
                            side = base_strategy.get('side', 'LONG')
                            cands = []
                            c_limit = _derive_pullback_limit_candidate('USD/JPY (ドル円)', side, ctx, df)
                            if c_limit:
                                cands.append(c_limit)
                            c_stop = dict(base_strategy)
                            c_stop['_candidate_kind'] = c_stop.get('_candidate_kind') or 'BREAKOUT_STOP'
                            c_stop = _decorate_time_rules(c_stop)
                            cands.append(c_stop)
                            c_hybrid = _derive_hybrid_confirm_market_candidate('USD/JPY (ドル円)', side, ctx, df)
                            if c_hybrid:
                                cands.append(c_hybrid)
                            usd_jpy_now = float(current_rate)
                            used_m = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy_now, leverage=leverage)
                            remain_m = float(capital) - float(used_m)
                            if remain_m < 0:
                                remain_m = 0.0
                            picked, evaluated = _evaluate_and_pick_candidates(
                                pair_label='USD/JPY (ドル円)',
                                candidates=cands,
                                capital_jpy=float(capital),
                                risk_percent_target=float(risk_percent),
                                max_risk_percent_cap=float(max_risk_percent_cap),
                                fixed_1lot_mode=bool(fixed_1lot_mode),
                                usd_jpy=usd_jpy_now,
                                remaining_margin_jpy=float(remain_m),
                                weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                                active_positions=st.session_state.portfolio_positions,
                                max_positions_per_currency=int(max_positions_per_currency),
                                leverage=leverage,
                            )
                            if picked:
                                chosen = picked
                            else:
                                chosen = {
                                    'decision': 'NO_TRADE',
                                    'side': 'NONE',
                                    'why': '固定1枚前提で、許容最大リスク%（上限）/週DDキャップ/証拠金の制約を満たす案がありません。',
                                    'notes': [f'上限リスク%={float(max_risk_percent_cap):.1f}', f'週DDキャップ={float(weekly_dd_cap_percent):.1f}'],
                                }
                            chosen['candidates'] = evaluated
                    except Exception as _e:
                        chosen = base_strategy
                        if isinstance(chosen, dict):
                            chosen.setdefault('notes', [])
                            chosen['notes'].append(f'3案生成評価で例外: {_e}')
                    st.session_state.last_strategy = chosen
                    # ✅ USD/JPY注文のEntry/TP/SLをチャートに重ね表示
                    _ov = _strategy_to_overlay("USD/JPY (ドル円)", st.session_state.last_strategy)
                    if _ov:
                        st.session_state.chart_pair_label = "USD/JPY (ドル円)"
                        st.session_state.chart_overlay = _ov

                    st.session_state.last_strategy_policy = gen_policy

                    # ✅ ロット計算機は「直近に生成した注文書のペア」に自動追従
                    st.session_state.calc_pair_label = "USD/JPY (ドル円)"
                    st.session_state.calc_ctx = dict(ctx)
                    st.session_state.calc_strategy = st.session_state.last_strategy

                    # 注文命令書を作り直したら、代替ペア関連のキャッシュはリセット（誤爆防止）
                    st.session_state.last_alt = None
                    st.session_state.last_alt_strategy = None
        else:
            st.warning("Gemini API Key を入力してください。")# --- 直近の注文命令書を表示（ボタン押下後も表示が残る） ---
    simple_view = st.checkbox('表示をシンプルにする（推奨）', value=True, key='simple_view')
    strategy = st.session_state.get("last_strategy") or {}
    if strategy:
        st.info("AI診断およびパネル診断との整合性を確認しました。")
        if simple_view and isinstance(strategy, dict):
            render_order_summary(jpize_json(strategy), pair_name="USD/JPY (ドル円)", title="📌 注文サマリー")
            with st.expander("詳細（JSON）"):
                st.json(jpize_json(strategy))
        else:
            if isinstance(strategy, dict):
                st.json(jpize_json(strategy))
            else:
                st.markdown(strategy)

        # --- 🧪 シャドー比較表示（直近の実行結果） ---
        if shadow_enabled:
            with st.expander("🧪 シャドー比較（Gemini vs GPT）", expanded=False):
                if not openai_api_key_shadow:
                    st.caption("OpenAI API Key (shadow) が未入力のため、比較はスキップされました。")
                elif not hasattr(logic, "get_ai_order_strategy_shadow_openai"):
                    st.warning("logic.py に get_ai_order_strategy_shadow_openai がありません。logic_shadow_compare.py を logic.py に差し替えてください。")
                else:
                    g = st.session_state.get("last_shadow_base") or {}
                    o = st.session_state.get("last_shadow_openai") or {}
                    d = st.session_state.get("last_shadow_diff") or {}
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Gemini（実運用）**")
                        st.json(jpize_json(g))
                    with c2:
                        st.markdown(f"**GPT（シャドー: {openai_model_shadow}）**")
                        st.json(jpize_json(o))
                    st.markdown("**差分サマリー**")
                    st.json(d)


        # ✅ 実損失ベース（固定1枚前提）の見える化
        if isinstance(strategy, dict) and strategy.get("decision") == "TRADE":
            try:
                usd_jpy_now = float(current_rate)
            except Exception:
                usd_jpy_now = 0.0
            used_m = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy_now, leverage=leverage)
            remain_m = float(capital) - float(used_m)
            if remain_m < 0:
                remain_m = 0.0

            e = float(strategy.get("entry") or ctx.get("price", 0.0) or 0.0)
            sl = float(strategy.get("stop_loss") or 0.0)
            side = str(strategy.get("side") or "LONG")
            sel = strategy.get("_eval") if isinstance(strategy.get("_eval"), dict) else _select_lots_with_fixed_mode(
                pair_label="USD/JPY (ドル円)",
                entry=e,
                stop_loss=sl,
                capital_jpy=float(capital),
                risk_percent_target=float(risk_percent),
                max_risk_percent_cap=float(max_risk_percent_cap),
                fixed_1lot_mode=bool(fixed_1lot_mode),
                usd_jpy=usd_jpy_now,
                remaining_margin_jpy=float(remain_m),
                leverage=leverage,
            )
            lots_sel = int(sel.get("lots", 0))
            st.caption(
                f"SL幅={sel.get('stop_w',0):.6f}（{sel.get('sl_pips',0):.0f}pips） / "
                f"1枚最大損失=¥{sel.get('loss_per_lot_jpy',0):,.0f} / "
                f"実質リスク(1枚)={sel.get('risk_1lot_pct',0):.2f}% / "
                f"推奨枚数={lots_sel}枚 / 実質リスク={sel.get('risk_actual_pct',0):.2f}% / "
                f"目標{float(risk_percent):.1f}%で1枚を収める必要資金=¥{sel.get('required_capital_for_target_1lot',0):,.0f}"
            )
            if lots_sel < 1:
                st.error("❌ この案は『固定1枚＋上限リスク%/証拠金/週DDキャップ』の制約で実行不可です。")

        if isinstance(strategy, dict) and strategy.get("candidates"):
            with st.expander("🧩 3案比較（押し目LIMIT / ブレイクSTOP / Hybrid）", expanded=False):
                rows = []
                for c in (strategy.get("candidates") or []):
                    if not isinstance(c, dict):
                        continue
                    ev = c.get("_eval") if isinstance(c.get("_eval"), dict) else {}
                    rows.append({
                        "案": c.get("_candidate_kind",""),
                        "entry_type": c.get("entry_type",""),
                        "entry": c.get("entry",0),
                        "SL": c.get("stop_loss",0),
                        "TP": c.get("take_profit",0),
                        "RR": round(float(c.get("_rr", 0.0) or 0.0), 2),
                        "lots": ev.get("lots", 0),
                        "実質リスク%": round(float(ev.get("risk_actual_pct", 0.0) or 0.0), 2),
                        "1枚リスク%": round(float(ev.get("risk_1lot_pct", 0.0) or 0.0), 2),
                        "OK": bool(ev.get("ok")),
                        "NG理由": ",".join(ev.get("reasons") or []) if isinstance(ev.get("reasons"), list) else "",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
        decision = ""
        try:
            decision = strategy.get("decision") if isinstance(strategy, dict) else ""
        except Exception:
            decision = ""

        # ✅ USD/JPYがTRADEなら、そのままポートフォリオに登録（週末判断/翌週制限のため）
        if decision == "TRADE" and isinstance(strategy, dict):
            if st.button("➕ ポートフォリオに登録: USD/JPY (ドル円)", key="btn_add_usdjpy_to_portfolio"):
                # ✅ 2%ルールに沿った「実行可能lots」を自動で保存（SBIは1枚=1万通貨）
                usd_jpy_now = float(current_rate)
                used_m = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy_now, leverage=leverage)
                remain_m = float(capital) - float(used_m)
                if remain_m < 0:
                    remain_m = 0.0

                e = float(strategy.get("entry") or ctx.get("price", 0.0) or 0.0)
                sl = float(strategy.get("stop_loss") or 0.0)
                tp = float(strategy.get("take_profit") or 0.0)

                sel = _select_lots_with_fixed_mode(
                    pair_label="USD/JPY (ドル円)",
                    entry=e,
                    stop_loss=sl,
                    capital_jpy=float(capital),
                    risk_percent_target=float(risk_percent),
                    max_risk_percent_cap=float(max_risk_percent_cap),
                    fixed_1lot_mode=bool(fixed_1lot_mode),
                    usd_jpy=usd_jpy_now,
                    remaining_margin_jpy=float(remain_m),
                    leverage=leverage,
                )
                lots_int = int(sel.get("lots", 0))
                risk_actual_pct = float(sel.get("risk_actual_pct", 0.0))
                req_margin_per_lot = float(sel.get("req_margin_per_lot", 0.0))
                loss_per_lot_jpy = float(sel.get("loss_per_lot_jpy", 0.0))
                stop_w = float(sel.get("stop_w", 0.0))
                quote_ccy = str(sel.get("quote_ccy", "JPY"))

                if lots_int < 1:
                    st.error(
                        "❌ 登録不可：2%ルール（損切幅）または余力（証拠金）から算出すると『発注できる枚数が0枚』です。"
                        f"（損切幅={stop_w:.6f} / 1枚想定損失=¥{loss_per_lot_jpy:,.0f} / 余力=¥{remain_m:,.0f}）"
                    )
                else:
                    if not logic.can_open_under_weekly_cap(st.session_state.portfolio_positions, float(risk_actual_pct), float(weekly_dd_cap_percent)):
                        st.error("週単位DDキャップを超えるため登録できません。")
                    elif logic.violates_currency_concentration("USD/JPY (ドル円)", st.session_state.portfolio_positions, int(max_positions_per_currency)):
                        st.error("通貨集中フィルタにより登録できません。")
                    else:
                        st.session_state.portfolio_positions.append({
                            "pair": "USD/JPY (ドル円)",
                            "direction": "LONG" if strategy.get("side") == "LONG" else "SHORT",
                            "risk_percent": float(risk_actual_pct),  # 実質リスク%（整数lotsに丸めた後）
                            "lots": float(lots_int),
                            "entry_price": float(e),
                            "stop_loss": float(sl),
                            "take_profit": float(tp),
                            "horizon": str(strategy.get("horizon") or "WEEK"),
                            "entry_time": datetime.now(TOKYO).isoformat(),
                        })
                        st.success(f"ポートフォリオに登録しました（{lots_int}枚 / 実質リスク={risk_actual_pct:.2f}% / 必要証拠金=¥{req_margin_per_lot*lots_int:,.0f}）。")
                        st.rerun()

        # ✅ ドル円が見送りなら、代替ペア提案（週DDキャップ＆通貨集中フィルタ適用）
        effective_no_trade = (decision == "NO_TRADE") or bool(force_no_trade_debug)

        if force_no_trade_debug:
            st.error("⚠️ テストモード: decisionに関係なくNO_TRADE分岐（代替ペア提案）を表示しています。実運用の注文は押さないでください。")

        if effective_no_trade:
            st.warning("USD/JPY が見送り判定のため、代替ペア候補を自動提案します（通貨集中フィルタ＆週DDキャップ適用）。")

            # 代替提案は重いので、初回だけ生成して保持（ボタンの二段押しがStreamlitで失敗しないように）
            if st.session_state.get("last_alt") is None:
                st.session_state.last_alt = logic.suggest_alternative_pair_if_usdjpy_stay(
                    api_key=api_key,
                    active_positions=st.session_state.portfolio_positions,
                    risk_percent_per_trade=float(risk_percent),
                    weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                    max_positions_per_currency=int(max_positions_per_currency),
                    exclude_pair_label="USD/JPY (ドル円)"
                )

            alt = st.session_state.get("last_alt") or {}
            if simple_view and isinstance(alt, dict):
                render_alt_summary(jpize_json(alt))
                with st.expander("詳細（JSON）"):
                    st.json(jpize_json(alt))
            else:
                st.json(jpize_json(alt))

            if isinstance(alt, dict) and alt.get("best_pair_name"):
                best_pair = alt["best_pair_name"]
                if hasattr(logic, "canonical_pair_label"):
                    try:
                        best_pair = logic.canonical_pair_label(best_pair)
                    except Exception:
                        pass
                # 代替ペアの注文戦略を生成（別ボタンでも動くように、状態を保持）
                if st.button(f"🧠 代替ペアで注文戦略を生成: {best_pair}", key="btn_make_alt_order"):
                    alt_ctx = _build_ctx_for_pair(best_pair, ctx, us10y_raw)
                    if not alt_ctx.get("_pair_ctx_ok"):
                        st.warning("⚠️ 代替ペアの最新テクニカル（RSI/ATR等）が取得できませんでした。精度が落ちるため、原則ノートレ推奨です。")
                    df_alt = _get_df_for_pair(best_pair, us10y_raw)
                    base_strategy = logic.get_ai_order_strategy(api_key, alt_ctx, generation_policy='AUTO_HIERARCHY')
                    # --- シャドー比較（代替ペア） ---
                    if shadow_enabled and hasattr(logic, "get_ai_order_strategy_shadow_openai") and openai_api_key_shadow:
                        try:
                            alt_shadow_openai = logic.get_ai_order_strategy_shadow_openai(
                                openai_api_key_shadow,
                                alt_ctx,
                                openai_model=openai_model_shadow,
                                override_mode="AUTO_HIERARCHY",
                                override_reason="alt_order_strategy",
                            )
                        except Exception as _ase:
                            alt_shadow_openai = {"error": str(_ase)}
                        _write_jsonl(
                            f"logs/shadow_compare_{_today_tokyo_str()}.jsonl",
                            {
                                "ts": datetime.now(TOKYO).isoformat(),
                                "pair": alt_pair_label,
                                "policy": "AUTO_HIERARCHY",
                                "ctx": {k: alt_ctx.get(k) for k in ["price", "atr", "atr_avg60", "rsi", "sma25", "sma75", "sma_diff", "us10y", "current_time"] if k in alt_ctx},
                                "gemini": base_strategy if isinstance(base_strategy, dict) else None,
                                "openai": alt_shadow_openai if isinstance(alt_shadow_openai, dict) else None,
                                "diff": _shadow_diff(base_strategy if isinstance(base_strategy, dict) else {}, alt_shadow_openai if isinstance(alt_shadow_openai, dict) else {}),
                            },
                        )
                    chosen = base_strategy
                    try:
                        if prefer_pullback_limit and isinstance(base_strategy, dict) and base_strategy.get('decision') == 'TRADE' and df_alt is not None:
                            side = base_strategy.get('side', 'LONG')
                            cands = []
                            c_limit = _derive_pullback_limit_candidate(best_pair, side, alt_ctx, df_alt)
                            if c_limit:
                                cands.append(c_limit)
                            c_stop = dict(base_strategy)
                            c_stop['_candidate_kind'] = c_stop.get('_candidate_kind') or 'BREAKOUT_STOP'
                            c_stop = _decorate_time_rules(c_stop)
                            cands.append(c_stop)
                            c_hybrid = _derive_hybrid_confirm_market_candidate(best_pair, side, alt_ctx, df_alt)
                            if c_hybrid:
                                cands.append(c_hybrid)
                            usd_jpy_now = float(current_rate)
                            used_m = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy_now, leverage=leverage)
                            remain_m = float(capital) - float(used_m)
                            if remain_m < 0:
                                remain_m = 0.0
                            picked, evaluated = _evaluate_and_pick_candidates(
                                pair_label=best_pair,
                                candidates=cands,
                                capital_jpy=float(capital),
                                risk_percent_target=float(risk_percent),
                                max_risk_percent_cap=float(max_risk_percent_cap),
                                fixed_1lot_mode=bool(fixed_1lot_mode),
                                usd_jpy=usd_jpy_now,
                                remaining_margin_jpy=float(remain_m),
                                weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                                active_positions=st.session_state.portfolio_positions,
                                max_positions_per_currency=int(max_positions_per_currency),
                                leverage=leverage,
                            )
                            if picked:
                                chosen = picked
                            else:
                                chosen = {
                                    'decision': 'NO_TRADE',
                                    'side': 'NONE',
                                    'why': '固定1枚前提で、許容最大リスク%（上限）/週DDキャップ/証拠金の制約を満たす案がありません。',
                                    'notes': [f'上限リスク%={float(max_risk_percent_cap):.1f}', f'週DDキャップ={float(weekly_dd_cap_percent):.1f}'],
                                }
                            chosen['candidates'] = evaluated
                    except Exception as _e:
                        chosen = base_strategy
                        if isinstance(chosen, dict):
                            chosen.setdefault('notes', [])
                            chosen['notes'].append(f'3案生成評価で例外: {_e}')
                    st.session_state.last_alt_strategy = chosen

                    # ✅ 代替ペア注文のEntry/TP/SLをチャートに重ね表示（自動で代替チャートへ切替）
                    _ov2 = _strategy_to_overlay(best_pair, st.session_state.last_alt_strategy)
                    st.session_state.chart_pair_label = best_pair
                    st.session_state.chart_overlay = _ov2

                    # ✅ ロット計算機は「代替ペアの注文書」に自動追従
                    st.session_state.calc_pair_label = best_pair
                    st.session_state.calc_ctx = dict(alt_ctx)
                    st.session_state.calc_strategy = st.session_state.last_alt_strategy

                alt_strategy = st.session_state.get("last_alt_strategy")
                if alt_strategy:
                    st.subheader("代替ペアの注文戦略")
                    if simple_view and isinstance(alt_strategy, dict):
                        render_order_summary(jpize_json(alt_strategy), pair_name=best_pair, title="📌 代替ペア注文サマリー")
                        with st.expander("詳細（JSON）"):
                            st.json(jpize_json(alt_strategy))
                    else:
                        if isinstance(alt_strategy, dict):
                            st.json(jpize_json(alt_strategy))
                        else:
                            st.markdown(alt_strategy)

                    # 代替ペアがTRADEならワンクリックでポートフォリオに登録
                    if isinstance(alt_strategy, dict) and alt_strategy.get("decision") == "TRADE":
                        if st.button(f"➕ ポートフォリオに登録: {best_pair}", key="btn_add_alt_to_portfolio"):
                            # ✅ 代替ペアでも、2%ルールに沿って「実行可能lots」を自動保存
                            usd_jpy_now = float(current_rate)
                            used_m = _portfolio_margin_used_jpy(st.session_state.portfolio_positions, usd_jpy_now, leverage=leverage)
                            remain_m = float(capital) - float(used_m)
                            if remain_m < 0:
                                remain_m = 0.0

                            # 直近の代替ペアctxを優先（価格/指標が正しい）
                            if st.session_state.get("calc_pair_label") == best_pair and isinstance(st.session_state.get("calc_ctx"), dict):
                                alt_ctx_reg = st.session_state.get("calc_ctx")
                            else:
                                alt_ctx_reg = _build_ctx_for_pair(best_pair, ctx, us10y_raw)

                            e = float((alt_strategy.get("entry") if isinstance(alt_strategy, dict) else 0.0) or alt_ctx_reg.get("price", 0.0) or 0.0)
                            sl = float((alt_strategy.get("stop_loss") if isinstance(alt_strategy, dict) else 0.0) or 0.0)
                            tp = float((alt_strategy.get("take_profit") if isinstance(alt_strategy, dict) else 0.0) or 0.0)

                            sel = _select_lots_with_fixed_mode(
                                pair_label=best_pair,
                                entry=e,
                                stop_loss=sl,
                                capital_jpy=float(capital),
                                risk_percent_target=float(risk_percent),
                                max_risk_percent_cap=float(max_risk_percent_cap),
                                fixed_1lot_mode=bool(fixed_1lot_mode),
                                usd_jpy=usd_jpy_now,
                                remaining_margin_jpy=float(remain_m),
                                leverage=leverage,
                            )
                            lots_int = int(sel.get("lots", 0))
                            risk_actual_pct = float(sel.get("risk_actual_pct", 0.0))
                            req_margin_per_lot = float(sel.get("req_margin_per_lot", 0.0))
                            loss_per_lot_jpy = float(sel.get("loss_per_lot_jpy", 0.0))
                            stop_w = float(sel.get("stop_w", 0.0))
                            quote_ccy = str(sel.get("quote_ccy", "JPY"))

                            if lots_int < 1:
                                st.error(
                                    "❌ 登録不可：2%ルール（損切幅）または余力（証拠金）から算出すると『発注できる枚数が0枚』です。"
                                    f"（損切幅={stop_w:.6f} / 1枚想定損失=¥{loss_per_lot_jpy:,.0f} / 余力=¥{remain_m:,.0f}）"
                                )
                            else:
                                if not logic.can_open_under_weekly_cap(st.session_state.portfolio_positions, float(risk_actual_pct), float(weekly_dd_cap_percent)):
                                    st.error("週単位DDキャップを超えるため登録できません。")
                                elif logic.violates_currency_concentration(best_pair, st.session_state.portfolio_positions, int(max_positions_per_currency)):
                                    st.error("通貨集中フィルタにより登録できません。")
                                else:
                                    st.session_state.portfolio_positions.append({
                                        "pair": best_pair,
                                        "direction": "LONG" if (isinstance(alt_strategy, dict) and alt_strategy.get("side") == "LONG") else "SHORT",
                                        "risk_percent": float(risk_actual_pct),
                                        "lots": float(lots_int),
                                        "entry_price": float(e),
                                        "stop_loss": float(sl),
                                        "take_profit": float(tp),
                                        "horizon": str((alt_strategy.get("horizon") if isinstance(alt_strategy, dict) else "WEEK") or "WEEK"),
                                        "entry_time": datetime.now(TOKYO).isoformat(),
                                    })
                                    st.success(f"ポートフォリオに登録しました（{lots_int}枚 / 実質リスク={risk_actual_pct:.2f}% / 必要証拠金=¥{req_margin_per_lot*lots_int:,.0f}）。")
                                    st.rerun()
            else:
                st.info("条件を満たす代替ペアがないため、今週は完全ノートレ推奨です。")
with tab3:
    st.markdown("##### ✅ 週末・月末判断（完全自動） & スワップ運用")

    # 週末判断（JSON命令）: 人が解釈しないための最重要ボタン
    col_w1, col_w2 = st.columns([1.2, 1.0])
    with col_w1:
        if st.button("✅ 週末判断（JSON命令を生成）"):
            if api_key:
                with st.spinner("週末判断（利確/損切/継続/1か月継続）を生成中..."):
                    wctx = dict(ctx)
                    # 注文戦略タブと同じ情報を渡す（週末判断の精度安定）
                    wctx["last_report"] = st.session_state.last_ai_report or ""
                    wctx["panel_short"] = diag['short']['status'] if diag else "不明"
                    wctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                    # pair_label が無ければドル円に固定（代替ペアを週末判断したい場合はポジション側でpairを保持）
                    wctx.setdefault("pair_label", "USD/JPY (ドル円)")
                    st.session_state.last_weekend = logic.get_ai_weekend_decision(api_key, wctx)
            else:
                st.warning("Gemini API Key を入力してください。")

    with col_w2:
        # 文章の長期ポートフォリオ（参考）
        if st.button("💰 長期ポートフォリオ（文章）"):
            if api_key:
                with st.spinner("スワップ・金利分析中..."):
                    st.markdown(logic.get_ai_portfolio(api_key, ctx))
            else:
                st.warning("Gemini API Key を入力してください。")

    # --- 週末判断の表示（日本語キー表示） ---
    if st.session_state.last_weekend is not None:
        st.subheader("📌 週末判断（命令）")
        try:
            st.json(jpize_json(st.session_state.last_weekend))
        except Exception:
            st.json(st.session_state.last_weekend)

        # --- 数値ルール監査（HOLD_MONTHの条件が明文化されたか） ---
        try:
            wctx2 = dict(ctx)
            wctx2["last_report"] = st.session_state.last_ai_report or ""
            wctx2["panel_short"] = diag['short']['status'] if diag else "不明"
            wctx2["panel_mid"] = diag['mid']['status'] if diag else "不明"
            wctx2.setdefault("pair_label", "USD/JPY (ドル円)")

            if hasattr(logic, "numeric_hold_month_ok"):
                ok, detail = logic.numeric_hold_month_ok(wctx2)
                st.caption("🔎 数値ルール監査（HOLD_MONTHの根拠）")
                st.json(jpize_json({
                    "structure_ok": bool(detail.get("structure_ok", False)),
                    "month_hold_line": detail.get("month_hold_line", 0),
                    "reached": bool(detail.get("reached", False)),
                    "structure_detail": detail.get("structure_detail", {}),
                }))
        except Exception:
            pass
