import yfinance as yf
import pandas as pd
import google.generativeai as genai

def get_market_data():
    usdjpy = yf.download("JPY=X", period="6mo", interval="1d")
    us10y = yf.download("^TNX", period="6mo", interval="1d")
    return usdjpy, us10y

def calculate_indicators(usdjpy, us10y):
    df = usdjpy.copy()
    # 週末対策：最新の有効なデータで埋める
    df = df.ffill()
    df['US10Y'] = us10y['Close'].ffill()
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_25'] = df['Close'].rolling(window=25).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    df['ATR'] = high_low.rolling(window=14).mean()
    
    return df

def judge_condition(df):
    last = df.iloc[-1]
    price = last['Close']
    sma5 = last['SMA_5']
    sma25 = last['SMA_25']
    
    # 短期判断
    if price > sma5:
        short_status, short_advice, short_color = "上昇傾向", "短期的な買いが優勢です。", "#e1f5fe"
    else:
        short_status, short_advice, short_color = "勢い鈍化・調整", "上値が重くなっています。慎重に。", "#fff3e0"
        
    # 中期判断
    if price > sma25:
        mid_status, mid_advice, mid_color = "上昇トレンド継続", "押し目買いを検討できる局面です。", "#e8f5e9"
    else:
        mid_status, mid_advice, mid_color = "トレンド転換注意", "25日線を割り込んでいます。警戒が必要です。", "#ffebee"
        
    return {
        "price": price,
        "short": {"status": short_status, "advice": short_advice, "color": short_color},
        "mid": {"status": mid_status, "advice": mid_advice, "color": mid_color}
    }

def get_ai_range(api_key, context):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        USD/JPY 現在価格:{context['price']:.2f}, RSI:{context['rsi']:.1f}, ATR:{context['atr']:.2f}, 時刻:{context['current_time']}。
        本日(または本日以降)の予想最高値と予想最低値を「最高:1XX.XX, 最低:1XX.XX」の形式で1行で回答してください。
        """
        response = model.generate_content(prompt)
        text = response.text
        # 簡易パース
        import re
        nums = re.findall(r'\d+\.\d+', text)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
    except:
        pass
    return None

def get_ai_analysis(api_key, context):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # ★朝専用のロジックをプロンプトに組み込む
    morning_logic = ""
    if "08:00" <= context['current_time'] <= "10:30":
        morning_logic = f"""
        【東京市場・朝の重要事項】
        現在は{context['current_time']}です。09:55の「仲値」公示に向けた実需の動きに注目してください。
        {'本日は五十日(ごとおび)のため、通常より仲値に向けてドル買い圧力が強まりやすいです。' if context['is_gotobi'] else ''}
        この時間帯特有の「仲値に向けた急騰」と「10時過ぎの反落リスク」を考慮し、安易な高値掴みへの警戒や、押し目買いの有効性を判定してください。
        """

    prompt = f"""
    あなたはFP1級の資格を持つ、慎重かつ的確なFXストラテジストです。
    以下のデータを分析し、プロの視点でレポートを記述してください。
    
    {morning_logic}
    
    【市場データ】
    ・ドル円価格: {context['price']:.3f}円
    ・米10年債利回り: {context['us10y']:.2f}%
    ・RSI: {context['rsi']:.2f}
    ・ATR(変動幅): {context['atr']:.3f}
    ・25日移動平均乖離率: {context['sma_diff']:.2f}%
    
    【レポート構成】
    1. 現在の相場環境の要約
    2. 上記データ（特に金利差とボラティリティ）から読み解くリスク
    3. 具体的な戦略（エントリー・利確・損切の目安価格を具体的に提示）
    4. 経済カレンダーを踏まえた、今週の警戒イベントへの助言
    
    回答は親しみやすくも、プロの厳格さを感じる日本語でお願いします。
    """
    response = model.generate_content(prompt)
    return response.text

def get_currency_strength():
    # 簡易版：主要通貨の終値を取得して変化率を計算
    tickers = {"USD": "JPY=X", "EUR": "EURJPY=X", "GBP": "GBPJPY=X", "AUD": "AUDJPY=X"}
    data = yf.download(list(tickers.values()), period="1mo")['Close']
    data = (data / data.iloc[0]) * 100 # 初日を100として正規化
    data.columns = tickers.keys()
    return data
