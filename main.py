import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logic  # ← logic.pyが必要

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer")
st.title("🤖 AI連携型 USD/JPY 戦略分析ツール")

# --- 状態保持の初期化 ---
if "ai_range" not in st.session_state:
    st.session_state.ai_range = None
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state:
    st.session_state.last_ai_report = "" 

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

# --- クオート更新 ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 最新クオート更新（429回避）"):
    st.session_state.quote = logic.get_latest_quote("JPY=X")
    st.rerun()

q_price, q_time = st.session_state.quote

# --- データ取得と計算 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

if (q_price is None) and (df is not None) and (not df.empty):
    q_price = float(df["Close"].iloc[-1])
    q_time = pd.Timestamp(df.index[-1]).tz_localize("Asia/Tokyo")

if df is None or df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

# 軸同期のためにインデックスを正規化
df.index = pd.to_datetime(df.index)

# AI予想ライン反映ボタン
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

# 最新レート表示
if q_price is not None:
    st.markdown(
        f"### 💱 最新USD/JPY: **{float(q_price):.3f} 円** "
        f"<span style='color:#888; font-size:0.9em'>(更新: {(q_time.strftime('%Y-%m-%d %H:%M JST') if q_time else '時刻不明')})</span>",
        unsafe_allow_html=True,
    )

# --- 1. 診断パネル ---
if diag is not None:
    col_short, col_mid = st.columns(2)
    with col_short:
        st.markdown(f"""
            <div style="background-color:{diag['short']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:220px;">
                <h3 style="color:#333; margin:0; font-size:16px;">📅 1週間スパン（短期勢い）</h3>
                <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['short']['status']}</h2>
                <p style="color:#555; font-size:14px; line-height:1.6;">{diag['short']['advice']}</p>
                <p style="color:#666; font-size:14px; font-weight:bold; margin-top:10px;">現在値: {diag['price']:.3f} 円</p>
            </div>
        """, unsafe_allow_html=True)
    with col_mid:
        st.markdown(f"""
            <div style="background-color:{diag['mid']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:220px;">
                <h3 style="color:#333; margin:0; font-size:16px;">🗓️ 1ヶ月スパン（中期トレンド）</h3>
                <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['mid']['status']}</h2>
                <p style="color:#555; font-size:14px; line-height:1.6;">{diag['mid']['advice']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- 2. 経済アラート ---
if diag is not None:
    try:
        if diag["short"]["status"] == "勢い鈍化・調整" or df["ATR"].iloc[-1] > df["ATR"].mean() * 1.5:
            st.warning("⚠️ **【警戒】ボラティリティ上昇中または重要局面です**")
    except Exception: pass

# --- 3. メインチャート（同期 & 予想ライン） ---
fig_main = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08, 
    subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り"),
    row_heights=[0.7, 0.3]
)

fig_main.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5日線", line=dict(color="#00ff00", width=1.5)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25日線", line=dict(color="orange", width=2)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75日線", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

# 予想ライン
if st.session_state.ai_range:
    h_val, l_val = st.session_state.ai_range
    fig_main.add_hline(y=h_val, line_dash="dash", line_color="red", annotation_text=f"上限:{h_val:.2f}", row=1, col=1)
    fig_main.add_hline(y=l_val, line_dash="dash", line_color="green", annotation_text=f"下限:{l_val:.2f}", row=1, col=1)

if entry_price > 0:
    fig_main.add_hline(y=entry_price, line_dash="dot", line_color="yellow", annotation_text="購入単価", row=1, col=1)

fig_main.add_trace(go.Scatter(x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan")), row=2, col=1)

fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
fig_main.update_xaxes(range=[start_view, last_date], matches='x', row=2, col=1)
fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)
fig_main.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(r=240))
st.plotly_chart(fig_main, use_container_width=True)

# --- 4. RSI ---
st.subheader(f"📈 RSI（過熱感: {float(df['RSI'].iloc[-1]):.2f}）")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#ff5722")))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="#00ff00")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="#ff0000")
fig_rsi.update_xaxes(range=[start_view, last_date])
fig_rsi.update_layout(height=250, template="plotly_dark", yaxis=dict(range=[0, 100]), margin=dict(r=240))
st.plotly_chart(fig_rsi, use_container_width=True)

# --- 5. 通貨強弱 ---
if strength is not None and not strength.empty:
    st.subheader("📊 通貨強弱（1ヶ月）")
    fig_str = go.Figure()
    color_map = {"日本円": "#ff0000", "豪ドル": "#00ff00", "ユーロ": "#a020f0", "英ポンド": "#c0c0c0", "米ドル": "#ffd700"}
    for col in strength.columns:
        fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col, line=dict(color=color_map.get(col))))
    fig_str.update_layout(height=400, template="plotly_dark", margin=dict(r=240))
    st.plotly_chart(fig_str, use_container_width=True)

# --- 6. AI詳細レポート & ポートフォリオ ---
st.divider()
col_rep, col_port = st.columns(2)
if col_rep.button("✨ Gemini AI 詳細レポート"):
    if api_key:
        with st.spinner("分析中..."):
            last_row = df.iloc[-1]
            jst = pytz.timezone("Asia/Tokyo")
            now_jst = datetime.now(jst)
            context = {
                "price": float(last_row["Close"]),
                "us10y": float(last_row["US10Y"]) if pd.notna(last_row["US10Y"]) else 0.0,
                "atr": float(last_row["ATR"]) if pd.notna(last_row["ATR"]) else 0.0,
                "sma_diff": float(last_row["SMA_DIFF"]) if pd.notna(last_row["SMA_DIFF"]) else 0.0,
                "rsi": float(last_row["RSI"]) if pd.notna(last_row["RSI"]) else 50.0,
                "current_time": now_jst.strftime("%H:%M"),
                "is_gotobi": now_jst.day in [5, 10, 15, 20, 25, 30],
            }
            report = logic.get_ai_analysis(api_key, context)
            st.session_state.last_ai_report = report 
            st.markdown(report)
    else: st.warning("Gemini API Key を入力してください。")

if col_port.button("💰 最適ポートフォリオ提示"):
    if api_key:
        with st.spinner("計算中..."):
            st.markdown(logic.get_ai_portfolio(api_key, {}))
    else: st.warning("Gemini API Key を入力してください。")

# --- 7. ロボ的注文戦略セクション ---
st.divider()
st.subheader("🤖 AIトレード命令書（診断連動型）")
if st.button("📝 診断に基づいた注文価格を算出"):
    if api_key:
        if not st.session_state.last_ai_report:
            st.warning("先に『✨ Gemini AI 詳細レポート』を実行してください。")
        else:
            with st.spinner("診断連動中..."):
                last_row = df.iloc[-1]
                context = {
                    "price": float(last_row["Close"]),
                    "atr": float(last_row["ATR"]),
                    "last_report": st.session_state.last_ai_report,
                    "panel_short": diag['short']['status'] if diag else "不明",
                    "panel_mid": diag['mid']['status'] if diag else "不明"
                }
                strategy = logic.get_ai_order_strategy(api_key, context)
                st.info("AI診断およびパネル診断との整合性を確認しました。")
                st.markdown(strategy)
    else:
        st.warning("Gemini API Key を入力してください。")
