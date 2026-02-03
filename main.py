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

# --- 修正点1: 状態保持の初期化 (最上部で実行し、再描画後もデータを保持) ---
if "ai_range" not in st.session_state:
    st.session_state.ai_range = None
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)

# --- APIキー取得（secretsまたは手入力） ---
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
    st.rerun() # 確実に反映させるため追加

q_price, q_time = st.session_state.quote

# --- データ取得と計算 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

# QUOTEが取れない場合、日足終値で必ず埋める
if (q_price is None) and (df is not None) and (not df.empty):
    q_price = float(df["Close"].iloc[-1])
    q_time = pd.Timestamp(df.index[-1]).tz_localize("Asia/Tokyo")

if df is None or df.empty:
    st.error("データが取得できませんでした。")
    st.stop()

df.index = pd.to_datetime(df.index)

# ✅ 修正点2: AI予想ライン反映ボタンの処理 (描画前にsession_stateへ値をセット)
if st.sidebar.button("📈 AI予想ライン反映"):
    if api_key:
        with st.spinner("AI予想を取得中..."):
            last_row = df.iloc[-1]
            context = {"price": last_row["Close"], "rsi": last_row["RSI"], "atr": last_row["ATR"]}
            # セッションに保存することで再実行後も描画が可能になる
            st.session_state.ai_range = logic.get_ai_range(api_key, context)
            st.rerun() # データをセットした直後に画面を更新してグラフに反映させる
    else:
        st.warning("Gemini API Key を入力してください。")

# 診断(diag)生成
try:
    diag = logic.judge_condition(df)
except Exception as e:
    diag = None
    st.error(f"judge_conditionでエラー: {e}")

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

# --- 1. 診断パネル (既存のHTML装飾をすべて維持) ---
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

# --- 2. 経済アラート (既存ロジック維持) ---
if diag is not None:
    try:
        if diag["short"]["status"] == "勢い鈍化・調整" or df["ATR"].iloc[-1] > df["ATR"].mean() * 1.5:
            st.warning("⚠️ **【警戒】ボラティリティ上昇中または重要局面です**")
            st.info("経済カレンダーを確認し、雇用統計やFOMC等の重要指標前後はポジション管理を徹底してください。")
    except Exception: pass

# --- 3. メインチャート ---
fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
    subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り")
)

fig_main.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5日線", line=dict(color="#00ff00", width=1.5)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25日線", line=dict(color="orange", width=2)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75日線", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

# ✅ 修正点3: AI予想ラインの描画 (セッションに保存された値を使う)
if st.session_state.ai_range:
    high_val, low_val = st.session_state.ai_range
    fig_main.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]], y=[high_val, high_val],
        name=f"予想最高:{high_val:.2f}", line=dict(color="red", width=2, dash="dash"),
        showlegend=True 
    ), row=1, col=1)
    fig_main.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]], y=[low_val, low_val],
        name=f"予想最低:{low_val:.2f}", line=dict(color="green", width=2, dash="dash"),
        showlegend=True
    ), row=1, col=1)

# 購入単価ライン (既存維持)
if entry_price > 0:
    fig_main.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[entry_price, entry_price], name=f"購入単価:{entry_price:.2f}", line=dict(color="yellow", width=2, dash="dot")), row=1, col=1)
    current_price = float(df["Close"].iloc[-1])
    pips = (current_price - entry_price) if trade_type == "買い（ロング）" else (entry_price - current_price)
    profit_color = "#228B22" if pips >= 0 else "#B22222"
    st.sidebar.markdown(f"""<div style="background-color:{profit_color}; padding:10px; border-radius:8px; text-align:center; border: 1px solid white;"><span style="color:white; font-weight:bold; font-size:16px;">損益状況: {pips:+.3f} 円</span></div>""", unsafe_allow_html=True)

# ✅ 修正点4: 米10年債の凡例修正 (showlegend=Trueを明示)
fig_main.add_trace(go.Scatter(
    x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan"), showlegend=True
), row=2, col=1)

fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
fig_main.update_xaxes(range=[start_view, last_date], showticklabels=True, row=2, col=1)
fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)

fig_main.update_layout(
    height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
    showlegend=True, legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
    margin=dict(r=240)
)
st.plotly_chart(fig_main, use_container_width=True)

# --- 4. RSI (✅ 修正点5: 30の凡例をラインの下に配置) ---
current_rsi = float(df["RSI"].iloc[-1])
st.subheader(f"📈 RSI（現在の過熱感: {current_rsi:.2f}）")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name=f"RSI(14): {current_rsi:.1f}", line=dict(color="#ff5722")))
fig_rsi.add_hline(y=70, line=dict(color="#00ff00", dash="dash"), annotation_text="70：買われすぎ", annotation_position="top right")
fig_rsi.add_hline(y=30, line=dict(color="#ff0000", dash="dash"), annotation_text="30:売られすぎ", annotation_position="bottom right") # 位置修正
fig_rsi.update_xaxes(range=[start_view, last_date])
fig_rsi.update_layout(height=250, template="plotly_dark", yaxis=dict(range=[0, 100]), showlegend=True, margin=dict(r=240))
st.plotly_chart(fig_rsi, use_container_width=True)

# --- 5. 通貨強弱 (既存の配色を維持) ---
if strength is not None and not strength.empty:
    st.subheader("📊 通貨強弱（1ヶ月）")
    fig_str = go.Figure()
    color_map = {"日本円": "#ff0000", "豪ドル": "#00ff00", "ユーロ": "#a020f0", "英ポンド": "#c0c0c0", "米ドル": "#ffd700"}
    for col in strength.columns:
        fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col, line=dict(color=color_map.get(col))))
    fig_str.update_layout(height=400, template="plotly_dark", showlegend=True, margin=dict(r=240))
    st.plotly_chart(fig_str, use_container_width=True)

# --- 6. AI詳細レポート & ポートフォリオ (五十日判定等の既存ロジックを完全維持) ---
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
            st.markdown(logic.get_ai_analysis(api_key, context))
    else: st.warning("Gemini API Key を入力してください。")

if col_port.button("💰 最適ポートフォリオ提示"):
    if api_key:
        with st.spinner("計算中..."):
            st.markdown(logic.get_ai_portfolio(api_key, {}))
    else: st.warning("Gemini API Key を入力してください。")
