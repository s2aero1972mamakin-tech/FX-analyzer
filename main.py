import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logic  # ← logic.pyが必要

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer 2026")
st.title("🤖 AI連携型 USD/JPY 戦略分析ツール (SBI仕様)")

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

# --- サイドバー設定 (資金管理機能追加) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI FX 資金管理")

# 1. 資金管理入力
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 10.0, 2.0, help="負けた時に資金の何%を失う覚悟があるか。プロは2%推奨。")
leverage = 25  # 固定

st.sidebar.markdown("---")
entry_price = st.sidebar.number_input("エントリー価格 (円)", value=0.0, format="%.3f")

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

# 最新レートが取得できない場合のバックアップ
current_rate = q_price if q_price else df["Close"].iloc[-1]

# 軸同期のためにインデックスを正規化
df.index = pd.to_datetime(df.index)

# AI予想ライン反映
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

# --- 2. 経済アラート & スリップロス推奨 ---
col_alert, col_slip = st.columns(2)
with col_alert:
    if diag is not None:
        try:
            if diag["short"]["status"] == "勢い鈍化・調整" or df["ATR"].iloc[-1] > df["ATR"].mean() * 1.5:
                st.warning("⚠️ **【警戒】ボラティリティ上昇中**")
        except Exception: pass
with col_slip:
    # ATRに基づく推奨スリップロス計算
    current_atr = df["ATR"].iloc[-1]
    rec_slip = max(3, int(current_atr * 10)) 
    st.info(f"🛡️ 現在の推奨スリップロス: **{rec_slip} pips (銭)** (ATR:{current_atr:.3f})")

# --- 3. メインチャート ---
fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("USD/JPY & AI予想", "米国債10年物利回り"), row_heights=[0.7, 0.3])
fig_main.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_5"], name="5日線", line=dict(color="#00ff00", width=1.5)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_25"], name="25日線", line=dict(color="orange", width=2)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df["SMA_75"], name="75日線", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

if st.session_state.ai_range:
    high_val, low_val = st.session_state.ai_range
    view_x = [start_view, last_date]
    fig_main.add_trace(go.Scatter(x=view_x, y=[high_val, high_val], name=f"予想最高:{high_val:.2f}", line=dict(color="red", width=2, dash="dash")), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=view_x, y=[low_val, low_val], name=f"予想最低:{low_val:.2f}", line=dict(color="green", width=2, dash="dash")), row=1, col=1)

if entry_price > 0:
    fig_main.add_trace(go.Scatter(x=[start_view, last_date], y=[entry_price, entry_price], name=f"購入単価:{entry_price:.2f}", line=dict(color="yellow", width=2, dash="dot")), row=1, col=1)

fig_main.add_trace(go.Scatter(x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan"), showlegend=True), row=2, col=1)

fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
fig_main.update_xaxes(range=[start_view, last_date], matches='x', row=2, col=1)
fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)
fig_main.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(r=240))
st.plotly_chart(fig_main, use_container_width=True)

# --- 4. RSI & SBI仕様ロット計算機 ---
st.subheader("🛠️ SBI FX ロット計算機 (1万通貨単位)")
col_rsi, col_calc = st.columns([1, 1.5])

with col_rsi:
    st.markdown(f"**📉 RSI（過熱感）: {float(df['RSI'].iloc[-1]):.2f}**")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff5722")))
    fig_rsi.add_hline(y=70, line=dict(color="#00ff00", dash="dash"))
    fig_rsi.add_hline(y=30, line=dict(color="#ff0000", dash="dash"))
    fig_rsi.update_xaxes(range=[start_view, last_date])
    fig_rsi.update_layout(height=200, template="plotly_dark", yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_rsi, use_container_width=True)

with col_calc:
    # SBI仕様の証拠金計算
    one_lot_units = 10000  # 1万通貨
    required_margin_per_lot = (current_rate * one_lot_units) / leverage # 1万通貨あたりの必要証拠金
    max_lots = int(capital / required_margin_per_lot) # 全力で買える枚数

    st.markdown("#### 🧮 リスク管理 vs 全力シミュレーション")
    
    # 損切幅の入力
    stop_p = st.number_input("想定損切幅 (円) ※例: 0.5円逆行で損切", value=0.5, step=0.1)
    
    if stop_p > 0:
        # リスク許容額に基づく推奨ロット
        risk_amount = capital * (risk_percent / 100)
        safe_lots = risk_amount / (stop_p * one_lot_units) # 推奨ロット数(小数)
        
        # 表示用整形
        c1, c2 = st.columns(2)
        with c1:
            st.error(f"""
            **💀 限界 (レバレッジ25倍)**
            - 必要証拠金(1万通貨): ¥{required_margin_per_lot:,.0f}
            - **最大発注可能数: {max_lots} 枚** (万通貨)
            """)
        with c2:
            st.success(f"""
            **🛡️ 推奨 (安全重視)**
            - 許容損失額: ¥{risk_amount:,.0f} ({risk_percent}%)
            - **推奨発注数量: {safe_lots:.1f} 枚** (万通貨)
            """)
            
        if safe_lots > max_lots:
            st.warning("⚠️ 注意：リスク許容範囲内ですが、証拠金不足で発注できない可能性があります。")
        elif safe_lots < 0.1:
            st.warning("⚠️ 注意：損切幅が広すぎるか資金不足のため、1000通貨単位(0.1枚)での取引を推奨します。")

# --- 5. 通貨強弱 ---
if strength is not None and not strength.empty:
    st.subheader("📊 通貨強弱（1ヶ月）")
    fig_str = go.Figure()
    color_map = {"日本円": "#ff0000", "豪ドル": "#00ff00", "ユーロ": "#a020f0", "英ポンド": "#c0c0c0", "米ドル": "#ffd700"}
    for col in strength.columns:
        fig_str.add_trace(go.Scatter(x=strength.index, y=strength[col], name=col, line=dict(color=color_map.get(col))))
    fig_str.update_layout(height=400, template="plotly_dark", showlegend=True, margin=dict(r=240))
    st.plotly_chart(fig_str, use_container_width=True)

# --- 6. AI実戦運用エリア (タブ化) ---
st.divider()
st.subheader("🤖 AI軍師・実戦運用本部")

tab1, tab2, tab3 = st.tabs(["📊 詳細レポート", "📝 注文戦略(日/週)", "💰 長期/ポートフォリオ"])

with tab1:
    if st.button("✨ レポート生成 (五十日/選挙対応)"):
        if api_key:
            with st.spinner("FP1級AIが分析中..."):
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
                    "capital": capital
                }
                report = logic.get_ai_analysis(api_key, context)
                st.session_state.last_ai_report = report 
                st.markdown(report)
        else: st.warning("Gemini API Key を入力してください。")

with tab2:
    if st.button("📝 注文命令書作成"):
        if api_key:
            if not st.session_state.last_ai_report:
                st.warning("先に『詳細レポート』を生成してください。")
            else:
                with st.spinner("資金管理・スリップロス計算中..."):
                    last_row = df.iloc[-1]
                    context = {
                        "price": float(last_row["Close"]),
                        "atr": float(last_row["ATR"]),
                        "last_report": st.session_state.last_ai_report,
                        "panel_short": diag['short']['status'] if diag else "不明",
                        "panel_mid": diag['mid']['status'] if diag else "不明",
                        "capital": capital
                    }
                    strategy = logic.get_ai_order_strategy(api_key, context)
                    st.info("AI診断およびパネル診断との整合性を確認しました。")
                    st.markdown(strategy)
        else:
            st.warning("Gemini API Key を入力してください。")

with tab3:
    st.markdown("##### 週末・月末判断 & スワップ運用")
    if st.button("💰 長期ポートフォリオ＆週末診断"):
        if api_key:
            with st.spinner("スワップ・金利分析中..."):
                st.markdown(logic.get_ai_portfolio(api_key, {}))
        else: st.warning("Gemini API Key を入力してください。")

