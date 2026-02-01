import yfinance as yf
import pandas as pd
import google.generativeai as genai
import datetime

# --- 1. 市場データ取得 ---
def get_market_data(period="1y"):
    try:
        usdjpy_df = yf.Ticker("JPY=X").history(period=period)
        us10y_df = yf.Ticker("^TNX").history(period=period)
        if usdjpy_df.empty or us10y_df.empty: return None, None
        usdjpy_df.index = usdjpy_df.index.tz_localize(None)
        us10y_df.index = us10y_df.index.tz_localize(None)
        return usdjpy_df, us10y_df
    except: return None, None

# --- 2. 指標計算（5日移動平均線を追加） ---
def calculate_indicators(df, us10y):
    if df is None or us10y is None: return None
    new_df = df[['Open', 'High', 'Low', 'Close']].copy()
    
    # 移動平均線（5日、25日、75日）
    new_df['SMA_5'] = new_df['Close'].rolling(window=5).mean()
    new_df['SMA_25'] = new_df['Close'].rolling(window=25).mean()
    new_df['SMA_75'] = new_df['Close'].rolling(window=75).mean()
    
    # RSI
    delta = new_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    new_df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR & 金利差
    high_low = new_df['High'] - new_df['Low']
    high_close = (new_df['High'] - new_df['Close'].shift()).abs()
    low_close = (new_df['Low'] - new_df['Close'].shift()).abs()
    new_df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
    new_df['US10Y'] = us10y['Close'].ffill()
    return new_df

# --- 3. 通貨強弱 ---
def get_currency_strength():
    pairs = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "AUD": "AUDUSD=X"}
    strength_data = pd.DataFrame()
    for name, sym in pairs.items():
        try:
            ticker = yf.Ticker(sym)
            d = ticker.history(period="1mo")['Close']
            d.index = d.index.tz_localize(None)
            if name == "JPY":
                strength_data[name] = (1/d).pct_change().cumsum() * 100
            else:
                strength_data[name] = d.pct_change().cumsum() * 100
        except: continue
    if not strength_data.empty:
        strength_data["USD"] = strength_data.mean(axis=1) * -1
        return strength_data.ffill().dropna()
    return pd.DataFrame()

# --- 4. 売買判断（5日線基準の短期診断） ---
def judge_condition(df):
    if df is None or len(df) < 2: 
        return None

    last, prev = df.iloc[-1], df.iloc[-2]
    rsi = last['RSI']
    price = last['Close']
    sma5 = last['SMA_5']
    sma25 = last['SMA_25']
    sma75 = last['SMA_75']

    # 中期（1ヶ月）診断
    if rsi > 70:
        mid_s, mid_c, mid_a = "‼️ 利益確定検討", "#ffeb3b", f"RSI({rsi:.1f})が70超。中期的な買われすぎ局面です。"
    elif rsi < 30:
        mid_s, mid_c, mid_a = "押し目買い検討", "#00bcd4", f"RSI({rsi:.1f})が30以下。中期的な仕込みの好機です。"
    elif sma25 > sma75 and prev['SMA_25'] <= prev['SMA_75']:
        mid_s, mid_c, mid_a = "強気・上昇開始", "#ccffcc", "25日線が75日線を上抜け。中期トレンドが上向きに転換しました。"
    else:
        mid_s, mid_c, mid_a = "ステイ・静観", "#e0e0e0", "明確なシグナル待ち。FPの視点では無理なエントリーを避ける時期です。"

    # 短期（1週間）診断（5日線基準）
    if price > sma5:
        short_s, short_c, short_a = "上昇継続（短期）", "#e3f2fd", f"価格が5日線({sma5:.2f})の上を維持。1週間スパンの勢いは強いです。"
    else:
        short_s, short_c, short_a = "勢い鈍化・調整", "#fce4ec", f"価格が5日線({sma5:.2f})を下回りました。短期的な利確検討ラインです。"

    return {
        "short": {"status": short_s, "color": short_c, "advice": short_a},
        "mid": {"status": mid_s, "color": mid_c, "advice": mid_a},
        "price": price
    }

# --- 5. AI統合アルゴリズム（プロンプト1文字も省略なし） ---
def get_active_model(api_key):
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods: return m.name
    except: pass
    return "models/gemini-1.5-flash"

def get_ai_analysis(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p, u, a, s, r = context_data.get('price', 0.0), context_data.get('us10y', 0.0), context_data.get('atr', 0.0), context_data.get('sma_diff', 0.0), context_data.get('rsi', 50.0)
        now = datetime.datetime.now().strftime('%Y年%m月%d日')
        
        prompt = f"""
        今日は {now} です。プロのアナリストとして、FP2級（ファイナンシャル・プランニング技能士）の知識を持つユーザーに対してUSD/JPYを分析してください。
        
        【市場データ】
        - ドル円価格: {p:.2f}円
        - 日米金利差(10年債): {u:.2f}%
        - ボラティリティ(ATR): {a:.2f}
        - SMA25乖離率: {s:.2f}%
        - RSI(14日): {r:.1f}

        【分析依頼：以下の項目に沿ってFPに分かりやすく回答してください】
        1. 【ファンダメンタルズ】日米金利差の現状を「金融資産運用の利回り」の観点から解説
        2. 【地政学・外部要因】インフレや景気後退、政治リスクがどう影響しているか（FPの景気サイクルに基づき解説）
        3. 【テクニカル】乖離率とRSI({r:.1f})から見て、今は「割安」か「割高」か。
        4. 【具体的戦略】NISAや外貨建資産のバランスを考える際のアドバイスのように、出口戦略（利確）を含めた今後1週間の戦略を提示
        """
        response = model.generate_content(prompt)
        return f"✅ 成功\n\n{response.text}"
    except Exception as e: return f"AI分析エラー: {str(e)}"

def get_ai_range(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        prompt = f"""
        現在のドル円は {p:.2f}円です。
        直近のテクニカルとファンダメンタルズから、今後1週間の「予想最高値」と「予想最安値」を予測してください。
        回答は必ず以下の形式（半角数字のみ）で返してください。
        [最高値, 最安値]
        """
        response = model.generate_content(prompt)
        import re
        nums = re.findall(r"\d+\.\d+|\d+", response.text)
        return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None
    except: return None