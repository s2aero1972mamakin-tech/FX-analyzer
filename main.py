import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logic

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer")
st.title("🤖 AI診断・同期グラフ FXツール")

# --- セッションステート保持 ---
if "ai_range" not in st.session_state:
    st.session_state.ai_range = None
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state:
    st.session_state.last_ai_report = ""

# --- サイドバー設定 ---
st.sidebar.header("⚙️ 設定")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.divider()
st.sidebar.subheader("📈 トレード設定")
entry_price = st.sidebar.number_input("エントリー価格 (円)", value=0.0, format="%.3f")
trade_type = st.sidebar.radio("ポジション種別", ["買い（ロング）", "売り（ショート）"])

if st.sidebar.button("🔄 最新クオート更新"):
    st.session_state.quote = logic.get_latest_quote("JPY=X")
    st.rerun()

# --- データ取得・計算 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)

# 【修正：軸同期の要】インデックスの型を確実にDateTimeへ
df.index = pd.to_datetime(df.index)
strength = logic.get_currency_strength()

# 現在価格確定
q_price, q_time = st.session_state.quote
if q_price is None:
    q_price = float(df["Close"].iloc[-1])
    q_time = df.index[-1]

# --- 1. 診断パネル (元のHTML/CSS構成を維持) ---
diag = logic.judge_condition(df)
if diag:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
            <div style="background:{diag['short']['color']}; padding:20px; border-radius:12px; border:1px solid #ccc;">
                <h3>📅 短期診断</h3>
                <h2>{diag['short']['status']}</h2>
                <p>{diag['short']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
            <div style="background:{diag['mid']['color']}; padding:20px; border-radius:12px; border:1px solid #ccc;">
                <h3>🗓️ 中期診断 (FP1級ロジック)</h3>
                <h2>{diag['mid']['status']}</h2>
                <p>{diag['mid']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 2. 同期メインチャート (横軸崩れ修正版) ---
st.subheader(f"📈 USD/JPY & 米金利 同期チャート (現在値: {q_price:.3f}円)")

last_date = df.index[-1]
start_view = last_date - timedelta(days=45)

# サブプロット作成：shared_xaxes を有効化
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3],
    subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り")
)

# グラフ1：メインチャート
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="USD/JPY"
), row=1, col=1)

# 各移動平均線
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5日線", line=dict(color="#00ff00")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25日線", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75日線", line=dict(color="purple")), row=1, col=1)

# AI予想レンジ (add_hline)
if st.session_state.ai_range:
    h, l = st.session_state.ai_range
    fig.add_hline(y=h, line_dash="dash", line_color="red", annotation_text="予想上限", row=1, col=1)
    fig.add_hline(y=l, line_dash="dash", line_color="green", annotation_text="予想下限", row=1, col=1)

# グラフ2：米10年債
fig.add_trace(go.Scatter(
    x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan")
), row=2, col=1)

# 【修正：軸の強制同期】 matches='x' で操作をリンク
fig.update_xaxes(range=[start_view, last_date], row=2, col=1)
fig.update_xaxes(matches='x')
fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=30, b=30))

st.plotly_chart(fig, use_container_width=True)

# --- 3. RSI & 通貨強弱 ---
c_rsi, c_str = st.columns(2)
with c_rsi:
    st.subheader("📈 RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="orange")))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(height=250, template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)

with c_str:
    st.subheader("📊 通貨強弱")
    if not strength.empty:
        fig_s = go.Figure()
        for c in strength.columns:
            fig_s.add_trace(go.Scatter(x=strength.index, y=strength[c], name=c))
        fig_s.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_s, use_container_width=True)

# --- 4. AIレポート・ロボ注文 (元のロジックを復元) ---
st.divider()
col_rep, col_ord = st.columns(2)

with col_rep:
    if st.button("✨ AI詳細レポート生成", use_container_width=True):
        if api_key:
            with st.spinner("分析中..."):
                ctx = {"price": q_price, "us10y": df["US10Y"].iloc[-1], "rsi": df["RSI"].iloc[-1], "atr": df["ATR"].iloc[-1], "sma_diff": df["SMA_DIFF"].iloc[-1]}
                st.session_state.last_ai_report = logic.get_ai_analysis(api_key, ctx)
        else:
            st.warning("APIキーを入力してください")
    
    if st.session_state.last_ai_report:
        st.markdown("### 📝 AI市場分析")
        st.info(st.session_state.last_ai_report)

with col_ord:
    if st.button("🤖 ロボ注文票を生成", use_container_width=True):
        if st.session_state.last_ai_report:
            with st.spinner("注文構築中..."):
                ctx = {"price": q_price, "atr": df["ATR"].iloc[-1], "last_report": st.session_state.last_ai_report, "panel_short": diag['short']['status'], "panel_mid": diag['mid']['status']}
                order_txt = logic.get_ai_order_strategy(api_key, ctx)
                st.markdown("### 📋 推奨IFDOCO注文")
                st.success(order_txt)
        else:
            st.warning("先にレポートを生成してください")

# サイドバーへの予想反映ボタン
if st.sidebar.button("📈 AI予想レンジを反映"):
    if api_key:
        st.session_state.ai_range = logic.get_ai_range(api_key, {"price": q_price})
        st.rerun()

# --- 5. ポートフォリオ助言 ---
with st.expander("💼 AI推奨アセットアロケーション"):
    if st.button("最適配分を計算"):
        if api_key:
            st.write(logic.get_ai_portfolio(api_key, {}))
