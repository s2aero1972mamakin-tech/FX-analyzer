import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math
import os
import json
import re
import base64
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

# --- Dev/Prod モード ---
def _is_truthy(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def is_dev_mode() -> bool:
    # 1) Streamlit Cloud Secrets を優先
    try:
        if "DEV_MODE" in st.secrets:
            return _is_truthy(st.secrets.get("DEV_MODE"))
    except Exception:
        pass
    # 2) 環境変数（ローカル用）
    return _is_truthy(os.getenv("AI_FX_DEV_MODE", ""))

DEV_MODE = is_dev_mode()

# --- 週次（ベース判定 / 水曜再判定）共有ストア（プロセス内で共有） ---
@st.cache_resource
def _global_week_store():
    return {"baseline": {}, "wed_done": set(), "wed_payload": {}, "alt_status": {}}

def _now_jst():
    return datetime.now(TOKYO)

def _week_meta_jst():
    now = _now_jst()
    iso = now.isocalendar()
    week_id = f"{iso.year}-W{iso.week:02d}"
    week_start = (now - timedelta(days=now.weekday())).date()  # 月曜
    return week_id, week_start, now

def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")

# download_buttonが環境によってはbytesで落ちにくいケースがあるため、文字列版も用意
def _json_str(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)

def _clip_text(s: str, max_len: int = 1600) -> str:
    try:
        if s is None:
            return ""
        s = str(s)
        return s if len(s) <= max_len else (s[:max_len] + " …(truncated)")
    except Exception:
        return ""

def _download_link(payload_bytes: bytes, file_name: str, label: str = "Safari用：リンクで保存") -> str:
    """download_buttonが環境で落ちない場合のフォールバック（data: URI）"""
    try:
        b64 = base64.b64encode(payload_bytes).decode("ascii")
        return f'<a href="data:application/json;base64,{b64}" download="{file_name}">⬇️ {label}</a>'
    except Exception:
        return ""


def _get_user_agent() -> str:
    """可能ならUser-Agentを取得（Streamlitのバージョンや環境差があるためtryで吸収）"""
    try:
        ctx = getattr(st, "context", None)
        if ctx is not None and hasattr(ctx, "headers"):
            h = ctx.headers
            # dict-like
            if hasattr(h, "get"):
                return h.get("User-Agent") or h.get("user-agent") or ""
            # fallback
            try:
                return h["User-Agent"]
            except Exception:
                try:
                    return h["user-agent"]
                except Exception:
                    return ""
    except Exception:
        return ""
    return ""

def _is_safari_browser() -> bool:
    """
    Safariの場合は st.download_button がHTML扱いで失敗するケースがあるため、
    Safari検出時は data: URI リンクで保存させる。
    Cloud側で強制したい場合は secrets に:
      FORCE_SAFARI_DOWNLOAD = true
    """
    try:
        if bool(st.secrets.get("FORCE_SAFARI_DOWNLOAD", False)):
            return True
    except Exception:
        pass

    ua = _get_user_agent()
    if not ua:
        return False

    # iOS Chrome = CriOS, Edge = EdgiOS, Firefox = FxiOS 等
    if ("Safari" in ua) and not any(k in ua for k in ["Chrome", "Chromium", "CriOS", "Edg", "EdgiOS", "OPR", "FxiOS"]):
        return True
    return False


def _build_decision_log(*, event: str, week_id: str, week_start_date, pair_label: str,
                        ctx: dict, strategy: dict, settings: dict, portfolio_positions: list,
                        last_ai_report: str = "", gen_policy: str = "") -> dict:
    return {
        "event": event,
        "timestamp_jst": _now_jst().isoformat(),
        "week_id": week_id,
        "week_start_date_jst": str(week_start_date),
        "pair": pair_label,
        "generation_policy": gen_policy,
        "decision": (strategy or {}).get("decision") if isinstance(strategy, dict) else "",
        "side": (strategy or {}).get("side") if isinstance(strategy, dict) else "",
        "entry": (strategy or {}).get("entry") if isinstance(strategy, dict) else None,
        "tp": (strategy or {}).get("tp") if isinstance(strategy, dict) else None,
        "sl": (strategy or {}).get("sl") if isinstance(strategy, dict) else None,
        "lots": (strategy or {}).get("lots") if isinstance(strategy, dict) else None,
        "why": (strategy or {}).get("why") if isinstance(strategy, dict) else "",
        "notes": (strategy or {}).get("notes") if isinstance(strategy, dict) else "",
        "ctx": ctx or {},
        "strategy": strategy or {},
        "last_ai_report": last_ai_report or "",
        "settings": settings or {},
        "portfolio_positions_min": [
            {
                "pair": p.get("pair"),
                "direction": p.get("direction"),
                "risk_percent": p.get("risk_percent"),
                "lots": p.get("lots"),
                "entry_price": p.get("entry_price"),
                "entry_time": p.get("entry_time"),
            }
            for p in (portfolio_positions or [])
            if isinstance(p, dict)
        ],
    }



# --- JSON download helper ---
def _json_bytes(payload: dict) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception:
        # fallback
        return str(payload).encode("utf-8")

def _week_file_name(prefix: str, week_id: str) -> str:
    safe = week_id.replace("/", "-")
    return f"{prefix}_{safe}.json"
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


def _brief_pair_report(pair_label: str, ctx: dict) -> str:
    """
    Lightweight, deterministic report so the AI doesn't reuse USD/JPY text for alternative pairs.
    Uses ctx: price, sma25, sma75, rsi, atr, atr_avg60.
    """
    try:
        price = float(ctx.get("price") or 0.0)
        sma25 = float(ctx.get("sma25") or 0.0)
        sma75 = float(ctx.get("sma75") or 0.0)
        rsi = float(ctx.get("rsi") or 50.0)
        atr = float(ctx.get("atr") or 0.0)
        atr_avg60 = float(ctx.get("atr_avg60") or atr or 0.0)
    except Exception:
        return f"{pair_label}の指標が十分に取得できませんでした。"

    # trend
    trend = "レンジ"
    if price > sma25 > sma75:
        trend = "上昇トレンド"
    elif price < sma25 < sma75:
        trend = "下降トレンド"

    # momentum
    mom = "中立"
    if rsi >= 60:
        mom = "買い優勢"
    elif rsi <= 40:
        mom = "売り優勢"

    # volatility
    vol = "平常"
    try:
        ratio = (atr / atr_avg60) if atr_avg60 else 1.0
    except Exception:
        ratio = 1.0
    if ratio >= 1.6:
        vol = "荒い"
    elif ratio <= 0.8:
        vol = "落ち着き"

    return f"{pair_label}は日足ベースで{trend}。RSIは{rsi:.1f}で{mom}、ボラは{vol}（ATR比={ratio:.2f}）。"

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
                # Override last_report with pair-specific brief to prevent mixing USD/JPY narrative
                ctx2["last_report"] = _brief_pair_report(pair_label, ctx2)
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


    # ✅ 判定根拠（見える化）：普段は閉じたまま（ストレス増やさない）
    notes = _dget(order, "注記", "notes", default=[])
    ai_veto = order.get("ai_veto") if isinstance(order, dict) else None

    # notes はコード（例: volatility_too_high_atr_spike）と自由文が混ざるので両対応
    if notes or ai_veto:
        with st.expander("判定根拠（見える化）", expanded=False):
            if notes:
                try:
                    # コードは日本語化、自由文はそのまま
                    norm = []
                    for x in (notes if isinstance(notes, list) else [notes]):
                        if x is None:
                            continue
                        s = str(x).strip()
                        if not s:
                            continue
                        if s in _REASON_JA:
                            norm.append(s)
                        else:
                            # 自由文は短く
                            norm.append(s if len(s) <= 220 else s[:220] + " …")
                    if norm:
                        st.write("・" + "\n・".join([reasons_to_ja_with_code([c]) if c in _REASON_JA else c for c in norm]))
                except Exception:
                    st.write(notes)

            if isinstance(ai_veto, dict):
                applied = bool(ai_veto.get("applied"))
                vconf = ai_veto.get("veto_confidence")
                all_codes = ai_veto.get("all_codes") or []
                verified = ai_veto.get("verified_codes") or []
                st.markdown("**AI veto（条件付き）**")
                st.write(f"- 採用: **{'はい' if applied else 'いいえ'}**")
                if vconf is not None:
                    st.write(f"- 確信度: {vconf}")
                if all_codes:
                    st.write(f"- 申告コード: {reasons_to_ja_with_code(all_codes if isinstance(all_codes, list) else [all_codes])}")
                if verified:
                    st.write(f"- 数値で検証できたコード: {reasons_to_ja_with_code(verified if isinstance(verified, list) else [verified])}")
                if (not applied) and all_codes:
                    st.info("※ 検証できないNO_TRADE（根拠が数値で裏付けられない）だったため、B+によりフォールバックで止まりにくくしています。")


    if regime or regime_why:
        with st.expander("相場モード（参考）"):
            if regime:
                st.write(f"相場モード: {regime}")
            if regime_why:
                _rw = str(regime_why)
                if ("ResourceExhausted" in _rw) or ("429" in _rw) or ("quota" in _rw):
                    st.warning("AIの利用制限(429)などで相場モード判定ができず、安全側（DEFENSIVE）で継続しています。")
                    with st.expander("詳細（原文）", expanded=False):
                        st.code(_rw)
                else:
                    st.write(regime_why)


# --- 落選理由コード → 日本語（運用向け） ---
# 表示は「日本語（code）」の併記にして、運用・開発どちらも迷わないようにします。
_REASON_JA = {
    "trend_only_gate": "トレンド相場限定の条件に合わない",
    "trend_gate_direction_not_aligned": "方向条件が一致しない（上昇/下降の並び不一致）",
    "trend_score_below_threshold": "トレンド強度が不足（trend_score不足）",
    "ma_converge_too_close": "移動平均線が接近しすぎ（レンジ寄り）",
    "rsi_neutral_zone": "RSIが中立帯（レンジ寄り）",
    "no_trade_gate": "見送りゲートに該当",
    "volatility_too_high_atr_spike": "ボラ急騰（ATRスパイク）で危険",
    "weekly_dd_cap": "週DDキャップ超過",
    "currency_concentration": "通貨集中ルール違反",
    "insufficient_margin": "必要証拠金不足",
    "risk_limit": "リスク上限に抵触",
    "ranked_lower": "優先度が低い（他候補を採用）",

    # --- B+（条件付きAI veto）用（週中の“押し目/崩れ”確認など） ---
    "atr_spike_soft": "ボラ上昇（警戒域）",
    "atr_spike_hard": "ボラ急騰（危険域）",
    "ma_converge_soft": "MA収束（警戒域）",
    "ma_converge_hard": "MA収束（レンジ濃厚）",
    "trend_score_soft": "トレンド弱め（警戒）",
    "trend_score_low": "トレンド不足（弱すぎ）",
    "rsi_overheat": "RSI過熱（反転注意）",
    "rsi_oversold": "RSI売られ過ぎ（反転注意）",
    "week_open_against": "週初方向と逆行（崩れ警戒）",

}

def reasons_to_ja_with_code(reasons) -> str:
    if not reasons:
        return ""
    out = []
    for r in reasons:
        if r is None:
            continue
        s = str(r).strip()
        if not s:
            continue
        ja = _REASON_JA.get(s)
        out.append(f"{ja}（{s}）" if ja else s)
    return " / ".join(out)


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
            rej_list = rej if isinstance(rej, list) else ([rej] if rej else [])
            rej_txt = reasons_to_ja_with_code(rej_list)
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


# ✅【追加】週次ベース判定 / 水曜再判定 状態
if "week_baseline" not in st.session_state:
    st.session_state.week_baseline = None
if "wed_recheck_payload" not in st.session_state:
    st.session_state.wed_recheck_payload = None

# --- APIキー取得 ---
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

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

# ✅【追加】デバッグ（テスト用）※DEV_MODE のときだけ表示（誤操作ゼロ）
if DEV_MODE:
    st.sidebar.subheader("🧪 デバッグ")
    force_no_trade_debug = st.sidebar.checkbox(
        "NO_TRADE分岐を強制表示（テスト用）",
        value=False,
        help="代替ペアの動線テスト用。実運用ではOFF。"
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
        st.session_state.portfolio_positions.append({
            "pair": add_pair,
            "direction": "LONG" if "LONG" in add_dir else "SHORT",
            "risk_percent": float(add_risk),
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

                    recs.append({
                        "pair": pair,
                        "direction": direction,
                        "risk_percent": _to_float(r.get("risk_percent", 0.0), 0.0),
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
if st.sidebar.button("🔄 最新クオート更新"):
    st.session_state.quote = logic.get_latest_quote("JPY=X")
    st.rerun()

q_price, q_time = st.session_state.quote

# --- データ取得と計算 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

# 最新レートの補完ロジック (モバイル・時間対応)
if df is not None and not df.empty:
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
    "is_gotobi": datetime.now(TOKYO).day in [5, 10, 15, 20, 25, 30],
    "capital": capital,
    "active_positions": st.session_state.portfolio_positions,
    "entry_price": entry_price,
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
                    st.session_state.last_strategy = logic.get_ai_order_strategy(
                        api_key,
                        ctx,
                        pair_name="USD/JPY (ドル円)",
                        portfolio_positions=st.session_state.portfolio_positions,
                        weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                        risk_percent_per_trade=float(risk_percent),
                        max_positions_per_currency=int(max_positions_per_currency),
                        generation_policy=gen_policy,
                    )

                    # ✅【週次】ベース判定（その週の最初のUSD/JPY注文命令書）を保存（端末またぎの再判定に利用）
                    _week_id, _week_start, _now = _week_meta_jst()
                    _store = _global_week_store()
                    if _week_id not in _store["baseline"]:
                        _baseline_payload = _build_decision_log(
                            event="BASELINE",
                            week_id=_week_id,
                            week_start_date=_week_start,
                            pair_label="USD/JPY (ドル円)",
                            ctx=dict(ctx),
                            strategy=st.session_state.last_strategy if isinstance(st.session_state.last_strategy, dict) else {},
                            settings={
                                "capital_jpy": float(capital),
                                "risk_percent_per_trade": float(risk_percent),
                                "weekly_dd_cap_percent": float(weekly_dd_cap_percent),
                                "max_positions_per_currency": int(max_positions_per_currency),
                                "leverage": int(leverage),
                            },
                            portfolio_positions=list(st.session_state.portfolio_positions),
                            last_ai_report=_clip_text(st.session_state.last_ai_report, 1600),
                            gen_policy=gen_policy,
                        )
                        _store["baseline"][_week_id] = _baseline_payload
                    st.session_state.week_baseline = _store["baseline"].get(_week_id)

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


        # --- 📁 週次ログ（1クリック保存） ---
        _wk_id, _wk_start, _wk_now = _week_meta_jst()
        _st = _global_week_store()
        _baseline = _st.get("baseline", {}).get(_wk_id)
        _wed = _st.get("wed_payload", {}).get(_wk_id)
        with st.expander("📁 週次ログ（保存）", expanded=False):
            st.caption("※保存先はサーバではなく、この端末のブラウザのダウンロードです。iPhone/iPadはダウンロード後にブラウザの↓から「ファイルに保存」を選ぶとiCloud経由でMacでも見られます。")
            is_safari = _is_safari_browser()

            if _baseline:
                _b = _json_bytes(_baseline)
                _fname = _week_file_name("baseline", _wk_id)

                if is_safari:
                    st.caption("SafariではダウンロードボタンがHTML扱いで失敗する場合があるため、リンク保存を既定にしています。")
                    _lnk = _download_link(_b, _fname, label="Safari: タップして保存")
                    if _lnk:
                        st.markdown(_lnk, unsafe_allow_html=True)
                else:
                    _dl1 = st.download_button(
                        "📥 BASELINE（今週のベース判定）を保存",
                        data=_b,
                        file_name=_fname,
                        mime="application/json",
                        key=f"dl_baseline_{_wk_id}"
                    )
                    if _dl1:
                        st.success("BASELINEログのダウンロードを開始しました（ブラウザのダウンロード一覧をご確認ください）。")

                    _lnk = _download_link(_b, _fname, label="うまくいかない時（リンクで保存）")
                    if _lnk:
                        st.markdown(_lnk, unsafe_allow_html=True)

                with st.expander("うまく保存できない時（コピー用：JSON）", expanded=False):
                    st.code(_json_str(_baseline), language="json")
            else:
                st.caption("今週のBASELINEログは未作成です（注文命令書作成後に自動生成されます）。")

            if _wed:
                _b2 = _json_bytes(_wed)
                _fname2 = _week_file_name("wed_recheck", _wk_id)

                if is_safari:
                    st.caption("Safariではリンク保存を既定にしています。")
                    _lnk2 = _download_link(_b2, _fname2, label="Safari: タップして保存")
                    if _lnk2:
                        st.markdown(_lnk2, unsafe_allow_html=True)
                else:
                    _dl2 = st.download_button(
                        "📥 WED_RECHECK（水曜再判定）を保存",
                        data=_b2,
                        file_name=_fname2,
                        mime="application/json",
                        key=f"dl_wed_{_wk_id}"
                    )
                    if _dl2:
                        st.success("WED_RECHECKログのダウンロードを開始しました（ブラウザのダウンロード一覧をご確認ください）。")

                    _lnk2 = _download_link(_b2, _fname2, label="うまくいかない時（リンクで保存）")
                    if _lnk2:
                        st.markdown(_lnk2, unsafe_allow_html=True)

                with st.expander("うまく保存できない時（コピー用：JSON）", expanded=False):
                    st.code(_json_str(_wed), language="json")
            else:
                st.caption("水曜再判定ログは未作成です。")

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

                lots_int, risk_actual_pct, req_margin_per_lot, loss_per_lot_jpy, stop_w, quote_ccy = _recommend_lots_int_and_risk(
                    "USD/JPY (ドル円)", e, sl, float(capital), float(risk_percent), usd_jpy_now, remain_m, leverage=leverage
                )

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
            # ✅【週次】代替提案の有無を週ストアに保存（=水曜再判定の表示条件に利用）
            _week_id2, _week_start2, _now2 = _week_meta_jst()
            _store2 = _global_week_store()
            try:
                _store2.setdefault("alt_status", {})
                _store2["alt_status"][_week_id2] = {
                    "best_pair_name": (alt.get("best_pair_name") if isinstance(alt, dict) else None),
                    "blocked": (alt.get("blocked") if isinstance(alt, dict) else None),
                    "reason": (alt.get("reason") if isinstance(alt, dict) else None),
                }
            except Exception:
                pass

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
                    st.session_state.last_alt_strategy = logic.get_ai_order_strategy(
                        api_key,
                        alt_ctx,
                        pair_name=best_pair,
                        portfolio_positions=st.session_state.portfolio_positions,
                        weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                        risk_percent_per_trade=float(risk_percent),
                        max_positions_per_currency=int(max_positions_per_currency),
                        generation_policy='AUTO_HIERARCHY',
                    )
                    # ✅【週次】代替ペアの最終判定（TRADE/NO_TRADE）を週ストアに保存（=水曜再判定条件に利用）
                    try:
                        _wkx, _wks, _wkn = _week_meta_jst()
                        _stx = _global_week_store()
                        _stx.setdefault("alt_status", {})
                        _stx["alt_status"].setdefault(_wkx, {})
                        if isinstance(st.session_state.last_alt_strategy, dict):
                            _stx["alt_status"][_wkx]["alt_strategy_decision"] = st.session_state.last_alt_strategy.get("decision")
                            _stx["alt_status"][_wkx]["best_pair_name"] = best_pair
                    except Exception:
                        pass

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

                            lots_int, risk_actual_pct, req_margin_per_lot, loss_per_lot_jpy, stop_w, quote_ccy = _recommend_lots_int_and_risk(
                                best_pair, e, sl, float(capital), float(risk_percent), usd_jpy_now, remain_m, leverage=leverage
                            )

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
    # --- 🗓 水曜1回だけ再判定（週1回ロック） ---
    _wk_id3, _wk_start3, _wk_now3 = _week_meta_jst()
    _st3 = _global_week_store()
    _baseline3 = _st3.get("baseline", {}).get(_wk_id3)
    _alt_stat = _st3.get("alt_status", {}).get(_wk_id3) if isinstance(_st3.get("alt_status", {}), dict) else None
    _baseline_dec = ""
    try:
        if isinstance(_baseline3, dict):
            _baseline_dec = _baseline3.get("decision") or (_baseline3.get("strategy") or {}).get("decision") or ""
    except Exception:
        _baseline_dec = ""

    _is_wed_jst = (_wk_now3.weekday() == 2)  # Monday=0, Wednesday=2
    _wed_done = (_wk_id3 in _st3.get("wed_done", set()))

    # 「月曜NO_TRADEで代替ペアも無」の週にだけ出す（=候補なし/ブロック）
    _alt_has_candidate = False
    try:
        if isinstance(_alt_stat, dict):
            _alt_has_candidate = bool(_alt_stat.get("best_pair_name"))
    except Exception:
        _alt_has_candidate = False

    _alt_decision = ""
    try:
        if isinstance(_alt_stat, dict):
            _alt_decision = str(_alt_stat.get("alt_strategy_decision") or "")
    except Exception:
        _alt_decision = ""

    # 表示条件:
    # - 今週BASELINEがNO_TRADE
    # - 水曜（JST）
    # - まだ今週実行していない
    # - さらに「代替候補が無い」または「代替ペアでもNO_TRADEが確定している」週のみ表示
    show_wed_recheck = bool(_baseline3) and (_baseline_dec == "NO_TRADE") and _is_wed_jst and (not _wed_done) and ( (not _alt_has_candidate) or (_alt_decision == "NO_TRADE") )

    st.markdown("---")
    st.subheader("🗓 水曜再判定（今週1回のみ）")
    if not _baseline3:
        st.caption("今週のベース判定（BASELINE）が未保存です。先に『注文命令書作成』を1回実行してください。")
    else:
        st.caption(f"今週のベース判定: **{_baseline_dec or '不明'}** / 代替候補: **{'あり' if _alt_has_candidate else 'なし'}** / 代替判定: **{_alt_decision or '未確定'}** / 水曜: **{'はい' if _is_wed_jst else 'いいえ'}** / 既に実行済み: **{'はい' if _wed_done else 'いいえ'}**")

    if show_wed_recheck:
        if st.button("🔁 水曜再判定を実行（今週1回のみ）", key="btn_wed_recheck"):
            if not api_key:
                st.warning("Gemini API Key を入力してください。")
            elif not st.session_state.last_ai_report:
                st.warning("先に『詳細レポート』を生成してください。")
            else:
                with st.spinner("水曜再判定中..."):
                    # 最新のctx（この実行時点の価格/指標）で再判定
                    ctx_re = dict(ctx)
                    ctx_re["last_report"] = st.session_state.last_ai_report
                    ctx_re["panel_short"] = diag['short']['status'] if diag else "不明"
                    ctx_re["panel_mid"] = diag['mid']['status'] if diag else "不明"

                    wed_strategy = logic.get_ai_order_strategy(
                        api_key,
                        ctx_re,
                        pair_name="USD/JPY (ドル円)",
                        portfolio_positions=st.session_state.portfolio_positions,
                        weekly_dd_cap_percent=float(weekly_dd_cap_percent),
                        risk_percent_per_trade=float(risk_percent),
                        max_positions_per_currency=int(max_positions_per_currency),
                        generation_policy="AUTO_HIERARCHY",
                    )

                    wed_payload = _build_decision_log(
                        event="WED_RECHECK",
                        week_id=_wk_id3,
                        week_start_date=_wk_start3,
                        pair_label="USD/JPY (ドル円)",
                        ctx=dict(ctx_re),
                        strategy=wed_strategy if isinstance(wed_strategy, dict) else {},
                        settings={
                            "capital_jpy": float(capital),
                            "risk_percent_per_trade": float(risk_percent),
                            "weekly_dd_cap_percent": float(weekly_dd_cap_percent),
                            "max_positions_per_currency": int(max_positions_per_currency),
                            "leverage": int(leverage),
                        },
                        portfolio_positions=list(st.session_state.portfolio_positions),
                        last_ai_report=_clip_text(st.session_state.last_ai_report, 1600),
                        gen_policy="AUTO_HIERARCHY",
                    )

                    _st3.setdefault("wed_payload", {})
                    _st3["wed_payload"][_wk_id3] = wed_payload
                    _st3.setdefault("wed_done", set())
                    _st3["wed_done"].add(_wk_id3)
                    st.session_state.wed_recheck_payload = wed_payload

                    # 画面にも結果を出す
                    st.success("水曜再判定を保存しました（週1回ロック済み）。下の『週次ログ（保存）』からダウンロードできます。")

                    # チャート重ね表示（TRADEなら）
                    _ov3 = _strategy_to_overlay("USD/JPY (ドル円)", wed_strategy)
                    if _ov3:
                        st.session_state.chart_pair_label = "USD/JPY (ドル円)"
                        st.session_state.chart_overlay = _ov3

                    st.rerun()
    else:
        st.caption("水曜再判定ボタンは条件を満たしたときだけ表示されます（BASELINE=NO_TRADE かつ 代替候補なし かつ 水曜 かつ 未実行）。")

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
