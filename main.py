import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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

# ✅【追加】週末判断（JSON）状態保持
if "last_weekend" not in st.session_state:
    st.session_state.last_weekend = None

# ✅【追加】ポートフォリオ（複数ポジション）状態
if "portfolio_positions" not in st.session_state:
    # 各要素: {"pair": str, "direction": "LONG/SHORT", "risk_percent": float, "entry_price": float, "entry_time": iso}
    st.session_state.portfolio_positions = []

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

# ✅【追加】デバッグ（テスト用）
st.sidebar.subheader("🧪 デバッグ")
force_no_trade_debug = st.sidebar.checkbox("NO_TRADE分岐を強制表示（テスト用）", value=False, help="代替ペアの動線テスト用。実運用ではOFF。")


leverage = 25  # 固定

# 2. ポジション情報 (AI連動 & チャート表示用)
st.sidebar.markdown("---")
st.sidebar.subheader("📂 保有ポジション")
entry_price = st.sidebar.number_input("保有価格 (円) ※なしは0", value=0.0, format="%.3f")
trade_type = st.sidebar.radio("保有タイプ", ["買い (Long)", "売り (Short)"], index=0)

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

# ✅ AI予想ラインがチャート範囲外に出ても表示されるよう、Y軸レンジに予想高安を含める
if st.session_state.ai_range:
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
fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り"), row_heights=[0.7, 0.3]
)
fig_main.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5日線", line=dict(color="#00ff00", width=1.5)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25日線", line=dict(color="orange", width=2)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75日線", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

# ★ AI予想ライン表示機能 (赤・緑点線)
if st.session_state.ai_range:
    high_val, low_val = st.session_state.ai_range
    view_x = [start_view, last_date]
    fig_main.add_trace(go.Scatter(x=view_x, y=[high_val, high_val], name=f"予想最高:{high_val:.2f}", line=dict(color="red", width=2, dash="dash")), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=view_x, y=[low_val, low_val], name=f"予想最低:{low_val:.2f}", line=dict(color="green", width=2, dash="dash")), row=1, col=1)

# ★ ポジション連動表示機能 (青・ピンク線)
if entry_price > 0:
    line_color = "blue" if "買い" in trade_type else "magenta"
    pos_name = f"保有:{entry_price:.2f}"
    fig_main.add_trace(go.Scatter(x=[start_view, last_date], y=[entry_price, entry_price], name=pos_name, line=dict(color=line_color, width=2, dash="dashdot")), row=1, col=1)

fig_main.add_trace(go.Scatter(x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan"), showlegend=True), row=2, col=1)

fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
fig_main.update_xaxes(range=[start_view, last_date], matches='x', row=2, col=1)
fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)
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
    required_margin_per_lot = (current_rate * one_lot_units) / leverage
    max_lots = int(capital / required_margin_per_lot)

    st.markdown("#### 🧮 リスク管理 vs 全力シミュレーション")
    stop_p = st.number_input("想定損切幅 (円) ※例: 0.5円逆行で損切", value=0.5, step=0.1)

    if stop_p > 0:
        risk_amount = capital * (risk_percent / 100)
        safe_lots = risk_amount / (stop_p * one_lot_units)

        c1, c2 = st.columns(2)
        with c1:
            st.error(f"""
            **💀 限界 (レバレッジ25倍)**
            - 必要証拠金/枚: ¥{required_margin_per_lot:,.0f}
            - **最大発注可能数: {max_lots} 枚**
            """)
        with c2:
            st.success(f"""
            **🛡️ 推奨 (安全重視)**
            - 許容損失額: ¥{risk_amount:,.0f}
            - **推奨発注数量: {safe_lots:.1f} 枚**
            """)

        if safe_lots > max_lots:
            st.warning("⚠️ 注意：リスク許容範囲内ですが、証拠金不足で発注できない可能性があります。")
        elif safe_lots < 0.1:
            st.warning("⚠️ 注意：損切幅が広すぎるか資金不足のため、取引推奨外です。")

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
    if st.button("📝 注文命令書作成", key="btn_make_order"):
        if api_key:
            if not st.session_state.last_ai_report:
                st.warning("先に『詳細レポート』を生成してください。")
            else:
                with st.spinner("資金管理・スリップロス計算中..."):
                    ctx["last_report"] = st.session_state.last_ai_report
                    ctx["panel_short"] = diag['short']['status'] if diag else "不明"
                    ctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                    st.session_state.last_strategy = logic.get_ai_order_strategy(api_key, ctx)
                    # 注文命令書を作り直したら、代替ペア関連のキャッシュはリセット（誤爆防止）
                    st.session_state.last_alt = None
                    st.session_state.last_alt_strategy = None
        else:
            st.warning("Gemini API Key を入力してください。")

    # --- 直近の注文命令書を表示（ボタン押下後も表示が残る） ---
    strategy = st.session_state.get("last_strategy") or {}
    if strategy:
        st.info("AI診断およびパネル診断との整合性を確認しました。")
        if isinstance(strategy, dict):
            st.json(jpize_json(strategy))
        else:
            st.markdown(strategy)

        decision = ""
        try:
            decision = strategy.get("decision") if isinstance(strategy, dict) else ""
        except Exception:
            decision = ""

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
            st.json(jpize_json(alt))

            if isinstance(alt, dict) and alt.get("best_pair_name"):
                best_pair = alt["best_pair_name"]

                # 代替ペアの注文戦略を生成（別ボタンでも動くように、状態を保持）
                if st.button(f"🧠 代替ペアで注文戦略を生成: {best_pair}", key="btn_make_alt_order"):
                    alt_ctx = _build_ctx_for_pair(best_pair, ctx, us10y_raw)
                    if not alt_ctx.get("_pair_ctx_ok"):
                        st.warning("⚠️ 代替ペアの最新テクニカル（RSI/ATR等）が取得できませんでした。精度が落ちるため、原則ノートレ推奨です。")
                    st.session_state.last_alt_strategy = logic.get_ai_order_strategy(api_key, alt_ctx)

                alt_strategy = st.session_state.get("last_alt_strategy")
                if alt_strategy:
                    st.subheader("代替ペアの注文戦略")
                    if isinstance(alt_strategy, dict):
                        st.json(jpize_json(alt_strategy))
                    else:
                        st.markdown(alt_strategy)

                    # 代替ペアがTRADEならワンクリックでポートフォリオに登録
                    if isinstance(alt_strategy, dict) and alt_strategy.get("decision") == "TRADE":
                        if st.button(f"➕ ポートフォリオに登録: {best_pair}", key="btn_add_alt_to_portfolio"):
                            if not logic.can_open_under_weekly_cap(st.session_state.portfolio_positions, float(risk_percent), float(weekly_dd_cap_percent)):
                                st.error("週単位DDキャップを超えるため登録できません。")
                            elif logic.violates_currency_concentration(best_pair, st.session_state.portfolio_positions, int(max_positions_per_currency)):
                                st.error("通貨集中フィルタにより登録できません。")
                            else:
                                st.session_state.portfolio_positions.append({
                                    "pair": best_pair,
                                    "direction": "LONG" if (isinstance(alt_strategy, dict) and alt_strategy.get("side") == "LONG") else "SHORT",
                                    "risk_percent": float(risk_percent),
                                    "entry_price": float((alt_strategy.get("entry") if isinstance(alt_strategy, dict) else 0.0) or ctx.get("price", 0.0) or 0.0),
                                    "entry_time": datetime.now(TOKYO).isoformat()
                                })
                                st.success("ポートフォリオに登録しました。")
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
