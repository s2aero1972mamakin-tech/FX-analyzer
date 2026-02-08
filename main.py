import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logic  # ← logic.pyが必要
import json

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="AI-FX Analyzer 2026")
st.title("🤖 AI連携型 マルチ通貨ポートフォリオ分析ツール (SBI仕様)")

# --- 状態保持の初期化 ---
if "ai_range" not in st.session_state:
    st.session_state.ai_range = None
if "quote" not in st.session_state:
    st.session_state.quote = (None, None)
if "last_ai_report" not in st.session_state:
    st.session_state.last_ai_report = "" 
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

# --- APIキー取得 ---
try:
    default_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    default_key = ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# ==========================================
# ✅ 【追加】AI市場スキャナー (サイドバー最上部)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 AI市場スキャナー")
if st.sidebar.button("🚀 全ペアからチャンスを探す"):
    if api_key:
        with st.spinner("全通貨ペアをスキャン中..."):
            res_json = logic.scan_best_pair(api_key)
            if res_json:
                data = json.loads(res_json)
                st.session_state.scan_result = data
                st.sidebar.success("完了")
            else:
                st.sidebar.error("失敗")
    else:
        st.sidebar.warning("API Keyが必要です")

if st.session_state.scan_result:
    best = st.session_state.scan_result
    st.sidebar.info(f"👑 推奨: **{best.get('best_pair_name')}**")
    st.sidebar.caption(f"理由: {best.get('reason')}")

# ==========================================
# ✅ 【追加】分析対象ペア選択
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 分析対象ペア")
# logic.PAIR_MAP がある前提
pair_options = list(logic.PAIR_MAP.keys()) if hasattr(logic, 'PAIR_MAP') else ["USD/JPY (ドル円)"]
selected_pair_label = st.sidebar.selectbox("対象通貨", pair_options, index=0)
target_symbol = logic.PAIR_MAP[selected_pair_label] if hasattr(logic, 'PAIR_MAP') else "JPY=X"
target_pair_name = selected_pair_label.split(" ")[0]

# --- サイドバー設定 (資金管理機能追加) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💰 SBI FX 資金管理")

# 1. 資金管理入力
capital = st.sidebar.number_input("軍資金 (JPY)", value=300000, step=10000)
risk_percent = st.sidebar.slider("1トレード許容損失 (%)", 1.0, 10.0, 2.0, help="負けた時に資金の何%を失う覚悟があるか。プロは2%推奨。")
leverage = 25  # 固定

# 2. ポジション情報 (AI連動 & チャート表示用)
# ✅ 【修正】複数ポジション対応
st.sidebar.markdown("---")
st.sidebar.subheader("📂 ポートフォリオ状況")

# ポジション1
with st.sidebar.expander("ポジション1 (主要)", expanded=True):
    p1_pair = st.selectbox("ペア", ["NONE"] + pair_options, key="p1_pair")
    entry_price = st.number_input("保有価格", value=0.0, format="%.3f", key="p1_price")
    p1_lots = st.number_input("数量(万通貨)", 0.0, step=0.1, key="p1_lots")
    trade_type = st.radio("売買", ["買い (Long)", "売り (Short)"], index=0, key="p1_type")

# ポジション2 (追加)
with st.sidebar.expander("ポジション2 (追加)", expanded=False):
    p2_pair = st.selectbox("ペア", ["NONE"] + pair_options, key="p2_pair")
    p2_entry = st.number_input("保有価格", value=0.0, format="%.3f", key="p2_price")
    p2_lots = st.number_input("数量(万通貨)", 0.0, step=0.1, key="p2_lots")
    p2_type = st.radio("売買", ["買い (Long)", "売り (Short)"], index=0, key="p2_type")

# 合計計算用
total_margin_used = 0.0
# (簡易計算：現在レートがまだ取れていないため、取得価格ベースで概算)
if p1_lots > 0: total_margin_used += (entry_price * p1_lots * 10000) / 25
if p2_lots > 0: total_margin_used += (p2_entry * p2_lots * 10000) / 25
st.sidebar.warning(f"使用中証拠金(概算): {int(total_margin_used):,} 円")


# --- クオート更新 ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 最新クオート更新"):
    # ✅ 選択されたシンボルで更新
    st.session_state.quote = logic.get_latest_quote(target_symbol)
    st.rerun()

q_price, q_time = st.session_state.quote

# --- データ取得と計算 ---
# ✅ 選択されたシンボルを渡す
usdjpy_raw, us10y_raw = logic.get_market_data(symbol=target_symbol)
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
        if last_idx.tzinfo is None:
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

# 最新レート表示 (スマホ対応・時刻フォーマット)
if q_price is not None:
    fmt_time = q_time.strftime('%Y-%m-%d %H:%M') if q_time else "時刻不明"
    st.markdown(
        f"### 💱 最新 {target_pair_name}: **{float(q_price):.3f}** "
        f"<span style='color:#888; font-size:0.8em; display:block'>(更新: {fmt_time} JST)</span>",
        unsafe_allow_html=True,
    )

# --- 1. 診断パネル ---
if diag is not None:
    col_short, col_mid = st.columns(2)
    with col_short:
        # 重複していた価格表示の行を削除しました
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
        except Exception: pass
with col_slip:
    # ATRに基づく推奨スリップロス計算
    current_atr = df["ATR"].iloc[-1]
    rec_slip = max(3, int(current_atr * 10)) 
    st.info(f"🛡️ 推奨スリップロス: **{rec_slip} pips** (ATR:{current_atr:.3f})")

# --- 3. メインチャート (AI予想ライン & ポジション表示対応) ---
# ✅ タイトルを選択ペア名に変更
fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"{target_pair_name} & AI予想", "米国債10年物利回り"), row_heights=[0.7, 0.3])
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
    # 買いなら青、売りならピンク
    line_color = "blue" if "買い" in trade_type else "magenta"
    pos_name = f"保有:{entry_price:.2f}"
    fig_main.add_trace(go.Scatter(x=[start_view, last_date], y=[entry_price, entry_price], name=pos_name, line=dict(color=line_color, width=2, dash="dashdot")), row=1, col=1)

fig_main.add_trace(go.Scatter(x=df.index, y=df["US10Y"], name="米10年債", line=dict(color="cyan"), showlegend=True), row=2, col=1)

fig_main.update_xaxes(range=[start_view, last_date], row=1, col=1)
fig_main.update_xaxes(range=[start_view, last_date], matches='x', row=2, col=1)
fig_main.update_yaxes(range=[y_min_view * 0.998, y_max_view * 1.002], autorange=False, row=1, col=1)
fig_main.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(r=10, l=10)) # マージン調整でスマホ対応
st.plotly_chart(fig_main, use_container_width=True) # モバイル対応: コンテナ幅に合わせる

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
    # ★ SBI仕様の証拠金計算と推奨表示
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
        
        # 表示用整形 (エラー表示と成功表示を正しく使い分け)
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
    fig_str.update_layout(height=350, template="plotly_dark", showlegend=True, margin=dict(r=10, l=10)) # スマホ用にマージン削減
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
    "current_time": q_time.strftime("%H:%M") if q_time else "不明",
    "is_gotobi": datetime.now(pytz.timezone("Asia/Tokyo")).day in [5, 10, 15, 20, 25, 30],
    "capital": capital,
    "entry_price": entry_price, # 保有価格 (Pos1)
    "trade_type": trade_type    # 保有タイプ (Pos1)
}

tab1, tab2, tab3 = st.tabs(["📊 詳細レポート", "📝 注文戦略(日/週)", "💰 長期/ポートフォリオ"])

with tab1:
    if st.button("✨ レポート生成"):
        if api_key:
            with st.spinner("FP1級AIが分析中..."):
                report = logic.get_ai_analysis(api_key, ctx)
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
                    ctx["last_report"] = st.session_state.last_ai_report
                    ctx["panel_short"] = diag['short']['status'] if diag else "不明"
                    ctx["panel_mid"] = diag['mid']['status'] if diag else "不明"
                    # ✅ 選択されたペア名を渡す
                    strategy = logic.get_ai_order_strategy(api_key, ctx, pair_name=target_pair_name)
                    st.info("AI診断およびパネル診断との整合性を確認しました。")
                    st.markdown(strategy)
        else:
            st.warning("Gemini API Key を入力してください。")

with tab3:
    st.markdown("##### 週末・月末判断 & スワップ運用")
    # ✅ 【修正】週末ホールド判定 (数値ルール) をここに統合
    
    st.info("💡 **ホールド基準**: 含み益が **2.0円 (200pips)** 以上ならHOLD、それ以外は決済推奨。")
    
    # 判定対象の選択
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        # ポジション1か2か、または新規か選べるように
        hold_target = st.selectbox("診断するポジション", ["ポジション1 (主要)", "ポジション2 (追加)"])
    
    # 選択に応じた値のセット
    if hold_target == "ポジション1 (主要)":
        h_pair_label = p1_pair
        h_price = entry_price
        h_type = trade_type
    else:
        h_pair_label = p2_pair
        h_price = p2_entry
        h_type = p2_type
        
    if st.button("🚀 週末ホールド判定を実行"):
        if api_key and h_price > 0 and "NONE" not in str(h_pair_label):
            with st.spinner("数値ルール照合中..."):
                # シンボル特定
                # logic.PAIR_MAPがあれば使う、なければデフォルト
                h_sym = logic.PAIR_MAP[h_pair_label] if hasattr(logic, 'PAIR_MAP') else "JPY=X"
                
                # 専用データ取得
                d_df, _ = logic.get_market_data(symbol=h_sym, period="5d")
                if d_df is not None:
                    h_curr = d_df["Close"].iloc[-1]
                    h_ctx = {
                        "price": h_curr,
                        "entry_price": h_price,
                        "trade_type": h_type
                    }
                    # AI判断呼び出し
                    res = logic.get_ai_weekend_decision(api_key, h_ctx, symbol=h_pair_label)
                    st.markdown("---")
                    st.markdown(res)
                else:
                    st.error("データ取得失敗")
        else:
            st.warning("有効なポジション情報を入力してください。")
            
    st.divider()
    if st.button("📊 ポートフォリオ配分案 (AI)"):
        if api_key:
             st.markdown(logic.get_ai_portfolio(api_key, ctx))
