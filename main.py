import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logic
import pandas as pd
from datetime import datetime, timedelta

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer")
st.title("🤖 AI連携型 USD/JPY 戦略分析ツール")

# --- APIキー取得 ---
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# --- サイドバーに設定追加 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 トレード設定")
entry_price = st.sidebar.number_input("エントリー価格 (円)", value=0.0, format="%.3f")
trade_type = st.sidebar.radio("ポジション種別", ["買い（ロング）", "売り（ショート）"])

# --- データ取得 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

if df is not None and not df.empty:
    df.index = pd.to_datetime(df.index)
    last_date = df.index[-1]
    # ★表示スパンを45日間に設定
    start_view = last_date - timedelta(days=45)
    
    # ズーム範囲内の高値・安値を計算してY軸を最適化
    mask = (df.index >= start_view)
    df_view = df.loc[mask]
    y_min_view = float(df_view['Low'].min())
    y_max_view = float(df_view['High'].max())
    
    # --- 1. 診断パネル ---
    diag = logic.judge_condition(df)
    if diag:
        col_short, col_mid = st.columns(2)
        with col_short:
            st.markdown(f"""
                <div style="background-color:{diag['short']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:200px;">
                    <h3 style="color:#333; margin:0; font-size:16px;">📅 1週間スパン（短期勢い）</h3>
                    <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['short']['status']}</h2>
                    <p style="color:#555; font-size:14px; line-height:1.4;">{diag['short']['advice']}</p>
                    <p style="color:#666; font-size:14px; font-weight:bold; margin-top:10px;">現在値: {diag['price']:.3f} 円</p>
                </div>
            """, unsafe_allow_html=True)
        with col_mid:
            st.markdown(f"""
                <div style="background-color:{diag['mid']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:200px;">
                    <h3 style="color:#333; margin:0; font-size:16px;">🗓️ 1ヶ月スパン（中期トレンド）</h3>
                    <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['mid']['status']}</h2>
                    <p style="color:#555; font-size:14px; line-height:1.4;">{diag['mid']['advice']}</p>
                </div>
            """, unsafe_allow_html=True)

    # --- 2. 経済カレンダー用アラート ---
    if diag['short']['status'] == "勢い鈍化・調整" or df['ATR'].iloc[-1] > df['ATR'].mean() * 1.5:
        st.warning("⚠️ **【警戒】ボラティリティ上昇中または重要局面です**")
        st.info("経済カレンダーを確認し、雇用統計やFOMC等の重要指標前後はポジション管理を徹底してください。")

    st.markdown("---") 

    # --- 3. メインチャート ---
    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                             subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り"))

    # ロウソク足
    fig_main.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name="価格", legend="legend1"), row=1, col=1)
    # 移動平均線
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="5日線", 
                                  line=dict(color='#00ff00', width=1.5), legend="legend1"), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="25日線", 
                                  line=dict(color='orange', width=2), legend="legend1"), row=1, col=1)

    # 損益分岐点（エントリー価格）
    if entry_price > 0:
        fig_main.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]], y=[entry_price, entry_price], 
            name=f"購入単価:{entry_price:.2f}", line=dict(color="yellow", width=2, dash="dot"), legend="legend1"
        ), row=1, col=1)
        
        current_price = df['Close'].iloc[-1]
        pips = (current_price - entry_price) if trade_type == "買い（ロング）" else (entry_price - current_price)
        profit_color = "#228B22" if pips >= 0 else "#B22222"
        st.sidebar.markdown(f"""
            <div style="background-color:{profit_color}; padding:10px; border-radius:8px; text-align:center; border: 1px solid white;">
                <span style="color:white; font-weight:bold; font-size:16px;">損益状況: {pips:+.3f} 円</span>
            </div>
        """, unsafe_allow_html=True)

    # AI予想ライン
    if api_key and st.sidebar.button("📈 AI予想ライン反映"):
        last_row = df.iloc[-1]
        context = {"price": last_row['Close'], "us10y": last_row['US10Y'], "atr": last_row['ATR'], 
                   "sma_diff": (last_row['Close'] - last_row['SMA_25']) / last_row['SMA_25'] * 100, "rsi": last_row['RSI']}
        ai_range = logic.get_ai_range(api_key, context)
        if ai_range:
            fig_main.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[ai_range[0], ai_range[0]], 
                                          name=f"予想最高:{ai_range[0]:.2f}", 
                                          line=dict(color="red", dash="dash"), legend="legend1"), row=1, col=1)
            fig_main.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[ai_range[1], ai_range[1]], 
                                          name=f"予想最低:{ai_range[1]:.2f}", 
                                          line=dict(color="green", dash="dash"), legend="legend1"), row=1, col=1)

    # 米10年債
    fig_main.add_trace(go.Scatter(x=df.index, y=df['US10Y'], name="米10年債", 
                                  line=dict(color='cyan'), legend="legend2"), row=2, col=1)

    # ★表示期間の強制固定（45日間）
    fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
    fig_main.update_xaxes(range=[start_view, last_date], showticklabels=True, row=2, col=1)
    
    # ★Y軸の自動ズーム設定
    fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)

    fig_main.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(y=0.98, x=1.02), legend2=dict(y=0.45, x=1.02), showlegend=True)
    st.plotly_chart(fig_main, use_container_width=True)

    # --- 4. RSI（凡例と数値を完全表示） ---
    current_rsi = df['RSI'].iloc[-1]
    st.subheader(f"📈 RSI（現在の過熱感: {current_rsi:.2f}）")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name=f"RSI(14): {current_rsi:.1f}", line=dict(color='#ff5722')))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"), annotation_text="買われすぎ")
    fig_rsi.add_hline(y=30, line=dict(color="cyan", dash="dash"), annotation_text="売られすぎ")
    
    # RSIの期間もメインと同期
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(
        height=250, template="plotly_dark", yaxis=dict(range=[0, 100]),
        showlegend=True, legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- 5. 通貨強弱 ---
    if strength is not None and not strength.empty:
        st.subheader("📊 通貨強弱（1ヶ月）")
        fig_str = go.Figure()
        for col in strength.columns:
            fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col))
        # 30日前から表示
        fig_str.update_layout(height=400, template="plotly_dark", xaxis=dict(range=[last_date - timedelta(days=30), last_date]),
                              showlegend=True, legend=dict(yanchor="top", y=1, xanchor="left", x=1.02))
        st.plotly_chart(fig_str, use_container_width=True)

    # --- 6. AI詳細レポート ---
    st.divider()
    if st.button("✨ Gemini AI 詳細レポート"):
        if api_key:
            with st.spinner('分析中...'):
                last_row = df.iloc[-1]
                context = {"price": last_row['Close'], "us10y": last_row['US10Y'], "atr": last_row['ATR'], 
                           "sma_diff": (last_row['Close'] - last_row['SMA_25']) / last_row['SMA_25'] * 100, "rsi": last_row['RSI']}
                st.markdown(logic.get_ai_analysis(api_key, context))
