import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logic

# --- 1. ページ構成・基本設定 ---
st.set_page_config(
    layout="wide", 
    page_title="AI-FX Pro Terminal", 
    initial_sidebar_state="expanded"
)

# カスタムCSSでUIの微調整
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 10px; }
    .diag-card { padding: 20px; border-radius: 15px; border: 1px solid #30363d; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. セッションステート保持 ---
if "ai_range" not in st.session_state: st.session_state.ai_range = None
if "quote" not in st.session_state: st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state: st.session_state.last_ai_report = ""
if "order_strategy" not in st.session_state: st.session_state.order_strategy = ""

# --- 3. サイドバー・コントロールパネル ---
with st.sidebar:
    st.header("🤖 AI Control Panel")
    api_key = st.text_input("Gemini API Key", type="password", help="Gemini 1.5 Flash API Key")
    
    st.divider()
    st.subheader("📊 Trade Configuration")
    entry_price = st.number_input("エントリー価格 (JPY)", value=0.0, format="%.3f")
    trade_type = st.radio("ポジション", ["買い（ロング）", "売り（ショート）"])
    
    st.divider()
    if st.button("🔄 市場データの強制更新", use_container_width=True):
        with st.spinner("Fetching latest quotes..."):
            st.session_state.quote = logic.get_latest_quote("JPY=X")
        st.rerun()

    if st.button("📈 AI予想レンジを反映", use_container_width=True):
        if api_key:
            current_p = st.session_state.quote[0] if st.session_state.quote[0] else 150.0
            st.session_state.ai_range = logic.get_ai_range(api_key, {"price": current_p})
            st.rerun()
        else:
            st.error("APIキーを入力してください")

# --- 4. データ取得・指標計算 ---
# logic.pyのキャッシュ機構付き関数を呼び出し
with st.spinner("Analyzing Market Data..."):
    usdjpy_raw, us10y_raw = logic.get_market_data()
    df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
    
    # 【重要】グラフの同期崩れを防ぐためのインデックスDateTime化
    df.index = pd.to_datetime(df.index)
    strength = logic.get_currency_strength()

# 最新クオートの確定
q_price, q_time = st.session_state.quote
if q_price is None: 
    q_price = float(df["Close"].iloc[-1])
    q_time = df.index[-1]

# --- 5. FP1級/AI診断パネル ---
st.title("🤖 AI-FX 統合診断ターミナル")
st.caption(f"Last Update: {q_time} | Current: {q_price:.3f} JPY")

diag = logic.judge_condition(df)
if diag:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
            <div class="diag-card" style="background:{diag['short']['color']}22; border-left: 5px solid {diag['short']['color']};">
                <h4 style="color:{diag['short']['color']};">📅 短期トレンド（5日線乖離）</h4>
                <h2 style="margin:0;">{diag['short']['status']}</h2>
                <p>{diag['short']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
            <div class="diag-card" style="background:{diag['mid']['color']}22; border-left: 5px solid {diag['mid']['color']};">
                <h4 style="color:{diag['mid']['color']};">🗓️ 中期診断（RSI/SMA/FP1級）</h4>
                <h2 style="margin:0;">{diag['mid']['status']}</h2>
                <p>{diag['mid']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 6. 同期メインチャート（軸ズレ修正版） ---
st.subheader("📈 テクニカル & ファンダメンタル同期チャート")

last_date = df.index[-1]
start_view = last_date - timedelta(days=60)

# サブプロット設定
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.04, 
    row_heights=[0.7, 0.3],
    subplot_titles=("USD/JPY & Indicators", "US 10Y Treasury Yield")
)

# グラフ1：メイン価格
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], 
    name="USD/JPY", increasing_line_color='#00ff88', decreasing_line_color='#ff3366'
), row=1, col=1)

# 移動平均線
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5SMA", line=dict(color="#00e5ff", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25SMA", line=dict(color="#ff9100", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75SMA", line=dict(color="#d500f9", width=1.2)), row=1, col=1)

# AI予想レンジの水平線（add_hlineだと軸がズレやすいためScatterで描画）
if st.session_state.ai_range:
    h, l = st.session_state.ai_range
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[h, h], name="AI上限", line=dict(color="#ff5252", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[l, l], name="AI下限", line=dict(color="#4caf50", dash="dash")), row=1, col=1)

# グラフ2：米10年債金利
fig.add_trace(go.Scatter(
    x=df.index, y=df["US10Y"], name="US10Y", line=dict(color="#00b0ff", width=2),
    fill='tozeroy', fillcolor='rgba(0, 176, 255, 0.1)'
), row=2, col=1)

# 【重要】軸の同期と表示範囲の固定
fig.update_xaxes(range=[start_view, last_date], row=2, col=1)
fig.update_xaxes(matches='x', showgrid=True, gridcolor='#333')
fig.update_yaxes(showgrid=True, gridcolor='#333')
fig.update_layout(
    height=800, 
    template="plotly_dark", 
    xaxis_rangeslider_visible=False,
    margin=dict(l=50, r=50, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- 7. RSI & 通貨強弱 ---
col_rsi, col_str = st.columns(2)

with col_rsi:
    st.subheader("📊 RSI (Relative Strength Index)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#ffa726", width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff5252")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#4caf50")
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(height=300, template="plotly_dark", margin=dict(t=20, b=20))
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_str:
    st.subheader("🌎 Currency Strength Index")
    if not strength.empty:
        fig_s = go.Figure()
        for col in strength.columns:
            fig_s.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col))
        fig_s.update_layout(height=300, template="plotly_dark", margin=dict(t=20, b=20))
        st.plotly_chart(fig_s, use_container_width=True)

# --- 8. AI分析・ロボ注文生成 ---
st.divider()
st.header("✨ AI Financial Advisor & Robot Order")

col_a1, col_a2 = st.columns(2)

with col_a1:
    if st.button("🔍 FP1級AI詳細レポートを生成", use_container_width=True):
        if not api_key: st.error("APIキーが必要です")
        else:
            with st.spinner("Analyzing political and economic factors..."):
                ctx = {
                    "price": q_price,
                    "us10y": df["US10Y"].iloc[-1],
                    "rsi": df["RSI"].iloc[-1],
                    "atr": df["ATR"].iloc[-1],
                    "sma_diff": df["SMA_DIFF"].iloc[-1]
                }
                st.session_state.last_ai_report = logic.get_ai_analysis(api_key, ctx)
    
    if st.session_state.last_ai_report:
        st.info("### AI Analysis Report")
        st.write(st.session_state.last_ai_report)

with col_a2:
    if st.button("🤖 最適IFDOCO注文票を作成", use_container_width=True):
        if not st.session_state.last_ai_report:
            st.warning("先に「AI詳細レポート」を生成してください")
        else:
            with st.spinner("Calculating optimal entry/exit..."):
                ctx = {
                    "price": q_price,
                    "atr": df["ATR"].iloc[-1],
                    "last_report": st.session_state.last_ai_report,
                    "panel_short": diag['short']['status'],
                    "panel_mid": diag['mid']['status']
                }
                st.session_state.order_strategy = logic.get_ai_order_strategy(api_key, ctx)
    
    if st.session_state.order_strategy:
        st.success("### AI Recommended Strategy")
        st.markdown(st.session_state.order_strategy)

# --- 9. ポートフォリオ助言 ---
with st.expander("💼 AI推奨アセットアロケーション"):
    if st.button("最適配分を計算"):
        if api_key:
            st.write(logic.get_ai_portfolio(api_key, {}))
        else: st.error("APIキーが必要です")
