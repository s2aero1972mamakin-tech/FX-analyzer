import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logic 
import json

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer 2026")
st.title("🤖 AI連携型 マルチ通貨ポートフォリオ分析ツール")

# --- 状態保持の初期化 ---
if "ai_range" not in st.session_state: st.session_state.ai_range = None
if "quote" not in st.session_state: st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state: st.session_state.last_ai_report = "" 
if "scan_result" not in st.session_state: st.session_state.scan_result = None

# --- APIキー ---
try: default_key = st.secrets.get("GEMINI_API_KEY", "")
except: default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# =================================================
# サイドバー: ポートフォリオ管理 & スキャナー
# =================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 AI市場スキャナー")

if st.sidebar.button("🚀 全ペアからチャンスを探す"):
    if api_key:
        with st.spinner("主要通貨ペアを分析中..."):
            res_json = logic.scan_best_pair(api_key)
            if res_json:
                data = json.loads(res_json)
                st.session_state.scan_result = data
                st.sidebar.success("分析完了！")
            else:
                st.sidebar.error("分析失敗")
    else:
        st.sidebar.warning("API Keyが必要です")

if st.session_state.scan_result:
    best = st.session_state.scan_result
    st.sidebar.info(f"👑 推奨: **{best.get('best_pair_name')}**")
    st.sidebar.caption(f"理由: {best.get('reason')}")

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 分析対象ペア選択")
selected_pair_label = st.sidebar.selectbox(
    "トレード対象", 
    list(logic.PAIR_MAP.keys()), 
    index=0
)
target_symbol = logic.PAIR_MAP[selected_pair_label]
target_pair_name = selected_pair_label.split(" ")[0]

st.sidebar.markdown("---")
st.sidebar.subheader("💰 ポートフォリオ状況")

# 資金管理計算用
total_unrealized_pl = 0.0
total_margin_used = 0.0
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)

# ポジション1
with st.sidebar.expander("ポジション1 (主要)", expanded=True):
    p1_pair = st.selectbox("ペア", ["NONE"] + list(logic.PAIR_MAP.keys()), key="p1_pair")
    if p1_pair != "NONE":
        p1_price = st.number_input("取得価格", 0.0, step=0.01, key="p1_price")
        p1_lots = st.number_input("数量(万通貨)", 0.0, step=0.1, key="p1_lots")
        p1_side = st.radio("売買", ["Long", "Short"], key="p1_side", horizontal=True)
        # 簡易現在値入力(本来はAPI取得推奨)
        p1_cur = st.number_input("現在値(概算)", value=p1_price, step=0.01, key="p1_cur")
        
        if p1_lots > 0:
            units = p1_lots * 10000
            margin = (p1_cur * units) / 25.0
            total_margin_used += margin
            diff = (p1_cur - p1_price) if p1_side == "Long" else (p1_price - p1_cur)
            pl = diff * units
            total_unrealized_pl += pl
            st.caption(f"損益: {int(pl):,}円 / 証拠金: {int(margin):,}円")

# ポジション2
with st.sidebar.expander("ポジション2 (追加)", expanded=False):
    p2_pair = st.selectbox("ペア", ["NONE"] + list(logic.PAIR_MAP.keys()), key="p2_pair")
    if p2_pair != "NONE":
        p2_price = st.number_input("取得価格", 0.0, step=0.01, key="p2_price")
        p2_lots = st.number_input("数量(万通貨)", 0.0, step=0.1, key="p2_lots")
        p2_side = st.radio("売買", ["Long", "Short"], key="p2_side", horizontal=True)
        p2_cur = st.number_input("現在値(概算)", value=p2_price, step=0.01, key="p2_cur")
        if p2_lots > 0:
            units = p2_lots * 10000
            margin = (p2_cur * units) / 25.0
            total_margin_used += margin
            diff = (p2_cur - p2_price) if p2_side == "Long" else (p2_price - p2_cur)
            pl = diff * units
            total_unrealized_pl += pl

st.sidebar.info(f"合計含み損益: {int(total_unrealized_pl):,} 円")
st.sidebar.warning(f"使用中証拠金: {int(total_margin_used):,} 円")

# ✅ 【復活】通貨強弱チャート (サイドバー下部)
st.sidebar.markdown("---")
st.sidebar.subheader("💪 通貨強弱 (直近1ヶ月)")
strength_df = logic.get_currency_strength()
if not strength_df.empty:
    st.sidebar.line_chart(strength_df)
else:
    st.sidebar.caption("データ取得中...")

# =================================================
# メイン画面処理
# =================================================

# データ取得 (選択されたペアを使用)
usdjpy_raw, us10y_raw = logic.get_market_data(symbol=target_symbol)
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)

if df is None or df.empty:
    st.error(f"データ取得エラー: {target_symbol}")
    st.stop()

# 最新レート
current_price = df["Close"].iloc[-1]
q_time = df.index[-1]
fmt_time = q_time.strftime('%Y-%m-%d %H:%M')

st.markdown(
    f"### 💱 {target_pair_name} 現在レート: **{current_price:.3f}** "
    f"<span style='color:#888; font-size:0.8em'>(更新: {fmt_time})</span>", 
    unsafe_allow_html=True
)

# 診断生成
diag = logic.judge_condition(df)

# チャート表示 (連動グラフ)
last_date = df.index[-1]
start_view = last_date - timedelta(days=60) # 期間を少し長めに
df_view = df.loc[df.index >= start_view]

# ✅ 【修正】3段構成チャート (価格 / RSI / 米国債)
# 2段目のRSIは、選択したペア(target_symbol)に基づいて計算されたものが表示されます。
fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=(f"{target_pair_name} Price & MA", "RSI (14) - Overbought/Oversold", "US 10Y Yield")
)

# 1段目: 価格とMA
fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['Open'], high=df_view['High'], low=df_view['Low'], close=df_view['Close'], name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_25'], line=dict(color='orange', width=1), name='SMA25'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_view.index, y=df_view['SMA_75'], line=dict(color='blue', width=1), name='SMA75'), row=1, col=1)

# 2段目: RSI (買われすぎ/売られすぎ)
fig.add_trace(go.Scatter(x=df_view.index, y=df_view['RSI'], line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
# 70と30のラインを明確に引く
fig.add_shape(type="line", x0=df_view.index[0], x1=df_view.index[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dot"), row=2, col=1)
fig.add_shape(type="line", x0=df_view.index[0], x1=df_view.index[-1], y0=30, y1=30, line=dict(color="blue", width=1, dash="dot"), row=2, col=1)
# 買われすぎ(70以上)エリアを背景色で強調
# (Plotlyの仕様上、shapeで塗りつぶすのは複雑になるため、ラインのみで対応)

# 3段目: 米国債利回り (US10Y)
if "US10Y" in df_view.columns and not df_view["US10Y"].isnull().all():
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['US10Y'], line=dict(color='green', width=1), name='US10Y Yield'), row=3, col=1)

fig.update_layout(height=800, margin=dict(l=0, r=0, t=30, b=0), showlegend=False) # 高さ調整
st.plotly_chart(fig, use_container_width=True)

# =================================================
# タブ機能
# =================================================
tab1, tab2, tab3 = st.tabs(["📊 詳細レポート", "📝 注文戦略(AI)", "📅 週末ホールド判定(数値)"])

# コンテキスト作成
ctx = {
    "price": current_price,
    "sma25": df["SMA_25"].iloc[-1],
    "sma75": df["SMA_75"].iloc[-1],
    "rsi": df["RSI"].iloc[-1],
    "atr": df["ATR"].iloc[-1],
    "atr_avg60": df["ATR"].rolling(60).mean().iloc[-1] if len(df)>60 else 0,
    "us10y": df["US10Y"].iloc[-1] if "US10Y" in df.columns else 0, 
    "capital": capital,
    "open_price": df["Open"].iloc[-1] 
}

with tab1:
    if st.button("✨ 市場レポート生成"):
        if api_key:
            with st.spinner("分析中..."):
                ctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                report = logic.get_ai_analysis(api_key, ctx)
                st.session_state.last_ai_report = report 
                st.markdown(report)
        else: st.warning("API Key Required")

with tab2:
    st.markdown("#### 戦略立案 (全イベント対応汎用版)")
    
    # 資金シミュレーション表示
    equity = capital + total_unrealized_pl
    free_margin = equity - total_margin_used
    st.markdown(f"**有効証拠金**: {int(equity):,}円 / **発注余力**: {int(free_margin):,}円")
    
    if st.button("📝 注文命令書作成"):
        if api_key:
            if not st.session_state.last_ai_report:
                st.warning("先にレポートを生成してください（一貫性のため）")
            else:
                with st.spinner(f"{target_pair_name} の戦略を策定中..."):
                    ctx["last_report"] = st.session_state.last_ai_report
                    ctx["panel_short"] = diag['short']['status'] if diag else "不明"
                    ctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                    
                    strategy = logic.get_ai_order_strategy(api_key, ctx, pair_name=target_pair_name)
                    st.json(strategy)
                    
                    if strategy.get("decision") == "TRADE":
                        ent = strategy.get("entry", 0)
                        sl = strategy.get("stop_loss", 0)
                        risk_val = abs(ent - sl) * 10000 
                        if risk_val > 0:
                            allowable_loss = free_margin * 0.02 
                            lots = allowable_loss / risk_val
                            st.success(f"推奨ロット数: **{lots:.2f}万通貨** (余力の2%リスク許容)")
        else:
            st.warning("API Key Required")

with tab3:
    st.markdown("#### 週末/月末 ホールド可否判定 (数値ルール主導)")
    st.info("💡 **ルール**: 含み益が **2.0円 (200pips)** 以上ならHOLD、それ以外は決済推奨。")
    
    col1, col2 = st.columns(2)
    with col1:
        eval_pair = st.selectbox("診断対象", list(logic.PAIR_MAP.keys()), key="eval_pair")
    with col2:
        eval_price = st.number_input("取得単価", 0.0, step=0.01, key="eval_entry")
        eval_type = st.radio("タイプ", ["Long", "Short"], key="eval_type")
        
    if st.button("🚀 判定実行"):
        if api_key and eval_price > 0:
            with st.spinner("数値ルール照合中..."):
                d_sym = logic.PAIR_MAP[eval_pair]
                d_df, _ = logic.get_market_data(symbol=d_sym, period="5d")
                
                if d_df is not None:
                    curr = d_df["Close"].iloc[-1]
                    eval_ctx = {
                        "price": curr,
                        "entry_price": eval_price,
                        "trade_type": eval_type
                    }
                    res = logic.get_ai_weekend_decision(api_key, eval_ctx, symbol=eval_pair)
                    st.markdown("---")
                    st.markdown(res)
                else:
                    st.error("データ取得失敗")
