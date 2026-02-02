import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import pytz

import logic  # ← ここでimportできている前提

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer")
st.title("🤖 AI連携型 USD/JPY 戦略分析ツール")

# --- APIキー取得 ---
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# --- サイドバー設定 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 トレード設定")
entry_price = st.sidebar.number_input("エントリー価格 (円)", value=0.0, format="%.3f")
trade_type = st.sidebar.radio("ポジション種別", ["買い（ロング）", "売り（ショート）"])

# --- クオート更新（429回避：ボタンのみ） ---
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 最新クオート更新"):
    st.session_state.quote = logic.get_latest_quote("JPY=X")

q_price, q_time = st.session_state.quote

# --- データ取得 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

if df is None or df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

df.index = pd.to_datetime(df.index)

# --- QUOTE フォールバック（日足終値） ---
if q_price is None:
    q_price = float(df["Close"].iloc[-1])
    q_time = pd.Timestamp(df.index[-1]).tz_localize("Asia/Tokyo")

# --- 最新価格表示 ---
st.markdown(
    f"## 💱 現在のUSD/JPY：**{q_price:.3f} 円**  "
    f"<span style='color:#888; font-size:0.9em'>(更新: {q_time.strftime('%Y-%m-%d %H:%M JST')})</span>",
    unsafe_allow_html=True,
)

# --- 判定ロジック ---
try:
    diag = logic.judge_condition(df)
except Exception:
    diag = None

last_date = df.index[-1]
start_view = last_date - timedelta(days=45)
df_view = df.loc[df.index >= start_view]

# --- 1. 診断パネル ---
if diag is not None:
    col_short, col_mid = st.columns(2)

    with col_short:
        st.markdown(f"""
        <div style="background-color:{diag['short']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd;">
            <h3>📅 短期（1週間）</h3>
            <h2>{diag['short']['status']}</h2>
            <p>{diag['short']['advice']}</p>
            <p><b>現在値: {diag['price']:.3f} 円</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        st.markdown(f"""
        <div style="background-color:{diag['mid']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd;">
            <h3>🗓️ 中期（1ヶ月）</h3>
            <h2>{diag['mid']['status']}</h2>
            <p>{diag['mid']['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 2. メインチャート ---
fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
    subplot_titles=("USD/JPY", "米国10年債利回り")
)

fig_main.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="USD/JPY"
), row=1, col=1)

fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="SMA5"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="SMA25"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="SMA75"), row=1, col=1)

if entry_price > 0:
    fig_main.add_hline(y=entry_price, line_dash="dot", line_color="yellow")

fig_main.add_trace(go.Scatter(
    x=df.index, y=df["US10Y"], name="US10Y", line=dict(color="cyan")
), row=2, col=1)

fig_main.update_layout(
    height=650,
    template="plotly_dark",
    legend=dict(x=1.02, y=1.0),
    margin=dict(r=260),
    xaxis_rangeslider_visible=False
)

fig_main.update_xaxes(range=[start_view, last_date])
st.plotly_chart(fig_main, use_container_width=True)

# --- 3. RSI ---
st.subheader(f"📈 RSI（現在: {df['RSI'].iloc[-1]:.2f}）")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dash")
fig_rsi.add_hline(y=30, line_dash="dash")
fig_rsi.update_layout(template="plotly_dark", height=250)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- 4. 通貨強弱（順序固定・USD含む） ---
st.subheader("📊 通貨強弱（1ヶ月）")
if strength is not None and not strength.empty:
    fig_str = go.Figure()
    for c in ["USD", "JPY", "EUR", "GBP", "AUD"]:
        if c in strength.columns:
            fig_str.add_trace(go.Scatter(x=strength.index, y=strength[c], name=c))
    fig_str.update_layout(
        template="plotly_dark",
        height=400,
        legend=dict(x=1.02, y=1.0),
        margin=dict(r=260),
    )
    st.plotly_chart(fig_str, use_container_width=True)

# --- 5. AI詳細レポート ---
st.divider()
if st.button("✨ Gemini AI 詳細レポート"):
    if api_key:
        with st.spinner("分析中..."):
            last_row = df.iloc[-1]
            jst = pytz.timezone("Asia/Tokyo")
            now_jst = datetime.now(jst)

            context = {
                "price": float(last_row["Close"]),
                "us10y": float(last_row["US10Y"]) if pd.notna(last_row["US10Y"]) else 0.0,
                "atr": float(last_row["ATR"]),
                "sma_diff": float(last_row["SMA_DIFF"]),
                "rsi": float(last_row["RSI"]),
                "current_time": now_jst.strftime("%H:%M"),
                "is_gotobi": now_jst.day in [5, 10, 15, 20, 25, 30],
            }

            st.markdown(logic.get_ai_analysis(api_key, context))
    else:
        st.warning("Gemini API Key を入力してください。")
