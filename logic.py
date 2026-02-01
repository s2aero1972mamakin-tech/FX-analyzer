import yfinance as yf
import pandas as pd
import google.generativeai as genai
import datetime

# --- 1. 市場データ取得 ---
def get_market_data(period="6mo"):
    try:
        usdjpy = yf.download("JPY=X", period=period)
        us10y = yf.download("^TNX", period=period)
        if usdjpy.empty or us10y.empty: return None, None
        return usdjpy, us10y
    except: return None, None

# --- 2. 指標計算（ATR厳密版・SMA75含む） ---
def calculate_indicators(df, us10y):
    if df is None or us10y is None: return None
    new_df = df[['Open', 'High', 'Low', 'Close']].copy()
    new_df = new_df.ffill()

    # 移動平均線 (5, 25, 75)
    new_df['SMA_5'] = new_df['Close'].rolling(window=5).mean()
    new_df['SMA_25'] = new_df['Close'].rolling(window=25).mean()
    new_df['SMA_75'] = new_df['Close'].rolling(window=75).mean()
    
    # RSI (14日)
    delta = new_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    new_df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # ATR (True Rangeを用いた厳密な計算)
    high_low = new_df['High'] - new_df['Low']
    high_close = (new_df['High'] - new_df['Close'].shift()).abs()
    low_close = (new_df['Low'] - new_df['Close'].shift()).abs()
    new_df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()
    
    new_df['US10Y'] = us10y['Close'].ffill()
    return new_df

# --- 3. 診断ロジック（SMA75条件含む） ---
def judge_condition(df):
    if df is None or len(df) < 2: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi, price = last['RSI'], last['Close']
    sma5, sma25, sma75 = last['SMA_5'], last['SMA_25'], last['SMA_75']

    # 中期（1ヶ月）診断
    if rsi > 70:
        mid_s, mid_c, mid_a = "‼️ 利益確定検討", "#ffeb3b", f"RSI({rsi:.1f})が70超。中期的な買われすぎ局面です。"
    elif rsi < 30:
        mid_s, mid_c, mid_a = "押し目買い検討", "#00bcd4", f"RSI({rsi:.1f})が30以下。中期的な仕込みの好機です。"
    elif sma25 > sma75 and prev['SMA_25'] <= prev['SMA_75']:
        mid_s, mid_c, mid_a = "強気・上昇開始", "#ccffcc", "25日線が75日線を上抜け。中期トレンドが上向きに転換しました。"
    else:
        mid_s, mid_c, mid_a = "ステイ・静観", "#e0e0e0", "明確なシグナル待ち。FPの視点では無理なエントリーを避ける時期です。"

    # 短期（1週間）診断 (5日線基準)
    if price > sma5:
        short_s, short_c, short_a = "上昇継続（短期）", "#e3f2fd", f"価格が5日線({sma5:.2f})の上を維持。1週間スパンの勢いは強いです。"
    else:
        short_s, short_c, short_a = "勢い鈍化・調整", "#fce4ec", f"価格が5日線({sma5:.2f})を下回りました。短期的な利確検討ラインです。"

    return {"short": {"status": short_s, "color": short_c, "advice": short_a}, "mid": {"status": mid_s, "color": mid_c, "advice": mid_a}, "price": price}

# --- 4. AI予想レンジ ---
def get_ai_range(api_key, context):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"現在のUSD/JPYは{context['price']:.2f}円です。今後1週間の予想最高値と最安値を[最高, 最低]の形式（数字のみ）で返してください。"
        response = model.generate_content(prompt)
        import re
        nums = re.findall(r"\d+\.\d+|\d+", response.text)
        return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None
    except: return None

# --- 5. AI詳細レポート（全プロンプト統合版） ---
def get_ai_analysis(api_key, context):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 朝の特別ロジック
    morning_logic = ""
    if "08:00" <= context['current_time'] <= "10:30":
        morning_logic = f"【重要：仲値分析モード】現在{context['current_time']}です。09:55の仲値に向けた実需（{'五十日' if context['is_gotobi'] else ''}）の動きと、その後の反落リスクを考慮せよ。"

    prompt = f"""
    あなたはFP1級を保持する、極めて優秀な為替戦略家です。
    特に今週は「衆議院選挙」を控えた極めて重要な1週間であることを強く認識してください。
    {morning_logic}
    
    【市場データ】
    - ドル円価格: {context['price']:.3f}円
    - 日米金利差(10年債): {context['us10y']:.2f}%
    - ボラティリティ(ATR): {context['atr']:.3f}
    - SMA25乖離率: {context['sma_diff']:.2f}%
    - RSI(14日): {context['rsi']:.1f}

    【分析依頼：以下の4項目に沿ってFPに分かりやすく回答してください】
    1. 【ファンダメンタルズ】日米金利差の現状を「金融資産運用の利回り」の観点から解説
    2. 【地政学・外部要因】インフレや景気後退、政治リスクがどう影響しているか（FPの景気サイクルに基づき解説）特に今週は「衆議院選挙」を控えた極めて重要な1週間であることを強く認識してください。
    3. 【テクニカル】乖離率とRSI({context['rsi']:.1f})から見て、今は「割安」か「割高」か。
    4. 【具体的戦略】NISAや外貨建資産のバランスを考える際のアドバイスのように、出口戦略（利確）を含めた今後1週間の戦略を提示

    【レポート構成：必ず以下の4項目に沿って記述してください】
    1. 現在の相場環境の要約
    2. 上記データ（特に金利差とボラティリティ）から読み解くリスク
    3. 具体的な戦略（エントリー・利確・損切の目安価格を具体的に提示）
    4. 経済カレンダーを踏まえた、今週の警戒イベントへの助言
    
    回答は親しみやすくも、プロの厳格さを感じる日本語でお願いします。
    """
    response = model.generate_content(prompt)
    return response.text

# --- 6. 通貨強弱 ---
def get_currency_strength():
    pairs = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "AUD": "AUDUSD=X"}
    strength_data = pd.DataFrame()
    for name, sym in pairs.items():
        try:
            d = yf.download(sym, period="1mo")['Close']
            strength_data[name] = (1/d if name == "JPY" else d).pct_change().cumsum() * 100
        except: continue
    if not strength_data.empty:
        strength_data["USD"] = strength_data.mean(axis=1) * -1
        return strength_data.ffill().dropna()
    return pd.DataFrame()

