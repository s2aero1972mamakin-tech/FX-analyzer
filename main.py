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

# --- データ取得 ---
usdjpy_raw, us10y_raw = logic.get_market_data()
df = logic.calculate_indicators(usdjpy_raw, us10y_raw)
strength = logic.get_currency_strength()

if df is not None and not df.empty:
    df.index = pd.to_datetime(df.index)
    
    # 表示範囲の設定（直近45日間で見やすくズーム）
    last_date = df.index[-1]
    start_view = last_date - timedelta(days=45)
    
    # --- 診断パネル（5日線/25日線 2枚パネル） ---
    diag = logic.judge_condition(df)
    if diag:
        col_short, col_mid = st.columns(2)
        with col_short:
            st.markdown(f"""
                <div style="background-color:{diag['short']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:200px;">
                    <h3 style="color:#333; margin:0; font-size:16px;">📅 1週間スパン（短期勢い：5日線基準）</h3>
                    <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['short']['status']}</h2>
                    <p style="color:#555; font-size:14px; line-height:1.4;">{diag['short']['advice']}</p>
                    <p style="color:#666; font-size:14px; font-weight:bold; margin-top:10px;">現在値: {diag['price']:.3f} 円</p>
                </div>
            """, unsafe_allow_html=True)
        with col_mid:
            st.markdown(f"""
                <div style="background-color:{diag['mid']['color']}; padding:20px; border-radius:12px; border:1px solid #ddd; min-height:200px;">
                    <h3 style="color:#333; margin:0; font-size:16px;">🗓️ 1ヶ月スパン（中期トレンド：25日線基準）</h3>
                    <h2 style="color:#333; margin:10px 0; font-size:24px;">{diag['mid']['status']}</h2>
                    <p style="color:#555; font-size:14px; line-height:1.4;">{diag['mid']['advice']}</p>
                </div>
            """, unsafe_allow_html=True)
            
    # --- 経済カレンダー用のアラート（簡易版：直近のボラティリティから警告） ---
　　 if diag['short']['status'] == "勢い鈍化・調整" or df['ATR'].iloc[-1] > df['ATR'].mean():
   　　　t.warning("⚠️ 重要指標や急変動の警戒期間です。ストップ注文の確認を推奨します。")

    # --- メインチャート ---
    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                             subplot_titles=("USD/JPY & AI予想 (直近分析)", "米国債10年物利回り"))

    # 1段目: USD/JPY
    fig_main.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name="ドル円価格", legend="legend1"), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], name="5日線(1週)", 
                                  line=dict(color='#00ff00', width=1.5), legend="legend1"), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_25'], name="25日線(1月)", 
                                  line=dict(color='orange', width=2), legend="legend1"), row=1, col=1)

    # --- メインチャートの描画部分 (fig_main.add_trace の後に追加) ---
if entry_price > 0:
    # 損益分岐点の水平線
    fig_main.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]], 
        y=[entry_price, entry_price], 
        name=f"エントリー: {entry_price:.3f}円", 
        line=dict(color="yellow", width=2, dash="dot"),
        legend="legend1"
    ), row=1, col=1)
    
    # 現在の損益状況をパネル付近に表示
    current_price = df['Close'].iloc[-1]
    pips = (current_price - entry_price) if entry_price != 0 else 0
    profit_color = "#00ff00" if pips >= 0 else "#ff4b4b"
    st.sidebar.markdown(f"""
        <div style="background-color:{profit_color}; padding:10px; border-radius:5px; text-align:center;">
            <span style="color:white; font-weight:bold;">現在の損益: {pips:+.3f} 円</span>
        </div>
    """, unsafe_allow_html=True)

    # AI予想ライン（凡例に動的な価格を含める修正）
    if api_key and st.sidebar.button("📈 AI予想ライン反映"):
        last_row = df.iloc[-1]
        context = {"price": last_row['Close'], "us10y": last_row['US10Y'], "atr": last_row['ATR'], 
                   "sma_diff": (last_row['Close'] - last_row['SMA_25']) / last_row['SMA_25'] * 100, "rsi": last_row['RSI']}
        ai_range = logic.get_ai_range(api_key, context)
        if ai_range:
            # 凡例名に価格（{ai_range[0]:.2f}円）を組み込む
            fig_main.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[ai_range[0], ai_range[0]], 
                                          name=f"予想最高: {ai_range[0]:.2f}円", 
                                          line=dict(color="red", dash="dash"), legend="legend1"), row=1, col=1)
            fig_main.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[ai_range[1], ai_range[1]], 
                                          name=f"予想最低: {ai_range[1]:.2f}円", 
                                          line=dict(color="green", dash="dash"), legend="legend1"), row=1, col=1)

    # 2段目: 米10年債
    fig_main.add_trace(go.Scatter(x=df.index, y=df['US10Y'], name="米10年債利回り", 
                                  line=dict(color='cyan'), legend="legend2"), row=2, col=1)

    # 軸の設定（日付ゲージを表示）
    fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
    fig_main.update_xaxes(range=[start_view, last_date], showticklabels=True, row=2, col=1)
    
    y_min = float(df.loc[start_view:, 'Low'].min())
    y_max = float(df.loc[start_view:, 'High'].max())
    fig_main.update_yaxes(range=[y_min * 0.995, y_max * 1.005], autorange=False, row=1, col=1)

    # 凡例位置とレイアウト（2段目の凡例をy=0.45付近に固定）
    fig_main.update_layout(
        height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(title="【価格・指標】", yanchor="top", y=0.98, xanchor="left", x=1.02),
        legend2=dict(title="【金利】", yanchor="top", y=0.45, xanchor="left", x=1.02),
        margin=dict(r=160, l=50, t=50, b=50),
        showlegend=True
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # --- RSIチャート（凡例あり） ---
    st.subheader("📈 RSI（直近過熱感）")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI(14)", line=dict(color='#ff5722')))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dash"), annotation_text="買われすぎ(70)")
    fig_rsi.add_hline(y=30, line=dict(color="cyan", dash="dash"), annotation_text="売られすぎ(30)")
    
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(
        height=300, template="plotly_dark", yaxis=dict(range=[0, 100]),
        showlegend=True, legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- 通貨強弱 ---
    if strength is not None and not strength.empty:
        st.subheader("📊 通貨強弱（直近1ヶ月）")
        fig_str = go.Figure()
        for col in strength.columns:
            fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col))
        
        fig_str.update_layout(
            height=400, template="plotly_dark",
            xaxis=dict(range=[last_date - timedelta(days=30), last_date]),
            legend=dict(title="【通貨】", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_str, use_container_width=True)

    # --- AI詳細レポート（ロジック側のプロンプトは完全版を想定） ---
    st.divider()
    if st.button("✨ Gemini AI 詳細レポート"):
        if api_key:
            with st.spinner('分析中...'):
                last_row = df.iloc[-1]
                context = {
                    "price": last_row['Close'], "us10y": last_row['US10Y'], "atr": last_row['ATR'], 
                    "sma_diff": (last_row['Close'] - last_row['SMA_25']) / last_row['SMA_25'] * 100, "rsi": last_row['RSI']
                }

                st.markdown(logic.get_ai_analysis(api_key, context))
