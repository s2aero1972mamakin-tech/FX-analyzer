import yfinance as yf
import pandas as pd
import google.generativeai as genai
import datetime
import pytz
import requests
BUILD_ID = "2026-02-02_01"
TOKYO = pytz.timezone("Asia/Tokyo")

def get_latest_quote(symbol="JPY=X"):
    """
    最新価格とその時刻を返す。
    1) yfinance fast_info
    2) yfinance intraday (1m)
    3) 為替APIフォールバック（USDJPY）
    """
    # --- 1) fast_info ---
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info or {}
        price = fi.get("last_price") or fi.get("lastPrice")
        ts = fi.get("last_timestamp") or fi.get("lastTimestamp")
        if price is not None and ts:
            qt = pd.to_datetime(ts, unit="s", utc=True)
            return float(price), qt
    except Exception:
        pass

    # --- 2) intraday 1m（Cloudで5mより成功することがある） ---
    try:
        intraday = yf.download(
            symbol,
            period="1d",
            interval="1m",
            progress=False,
            threads=False,
        )
        if not intraday.empty:
            close = intraday["Close"].dropna()
            if len(close) > 0:
                price = float(close.iloc[-1])
                qt = close.index[-1]
                if getattr(qt, "tz", None) is None:
                    qt = qt.tz_localize("UTC")
                return price, qt
    except Exception:
        pass

    # --- 3) 為替APIフォールバック（USD→JPY を取得）---
    # ※無料で叩ける公開API。yfinanceが死んでも取れることが多い
    #    open.er-api.com は "USD" 基準で JPY を返す
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if r.status_code == 200:
            j = r.json()
            rate = j["rates"]["JPY"]  # 1 USD = ??? JPY
            # 時刻はAPI側の更新時刻に寄せたいが、まずは「今」扱いでUTCを入れる
            qt = pd.Timestamp.utcnow().tz_localize("UTC")
            return float(rate), qt
    except Exception:
        pass

    return None, None


# --- 1. 市場データ取得（週末でも「今日行」を捏造しない版） ---
def get_market_data(period="1y"):
    try:
        ticker = yf.Ticker("JPY=X")
        usdjpy_df = ticker.history(period=period)  # daily
        us10y_df = yf.Ticker("^TNX").history(period=period)

        if usdjpy_df.empty or us10y_df.empty:
            return None, None

        # 最新クオート（価格 + 時刻）
        current_price, quote_time = get_latest_quote("JPY=X")

        if current_price is not None and quote_time is not None:
            # ① クオート時刻をJSTに寄せて日付を切る（重要）
            qt_tokyo = quote_time.tz_convert(TOKYO)
            quote_day = qt_tokyo.normalize()

            # ② 履歴の最終日もJSTに寄せて日付を切る（重要）
            last_idx = usdjpy_df.index[-1]
            if getattr(last_idx, "tz", None) is not None:
                last_day = last_idx.tz_convert(TOKYO).normalize()
            else:
                last_day = pd.Timestamp(last_idx).tz_localize(TOKYO).normalize()

            if last_day < quote_day:
                # 新しい日（JST）なら、その日付で行を追加
                new_row = usdjpy_df.iloc[-1:].copy()
                new_row.index = [quote_day]
                new_row["Open"] = new_row["High"] = new_row["Low"] = new_row["Close"] = float(current_price)
                usdjpy_df = pd.concat([usdjpy_df, new_row])
            else:
                # 同じ日（JST）なら最終行のCloseを更新
                usdjpy_df.iloc[-1, usdjpy_df.columns.get_loc("Close")] = float(current_price)

            # デバッグ表示（必要な間だけON）
            print("QUOTE_TIME_UTC:", quote_time)
            print("QUOTE_TIME_JST:", quote_time.tz_convert(TOKYO))
            print("QUOTE_DAY_JST :", quote_time.tz_convert(TOKYO).normalize())

        # tz 제거（以降の処理/描画が素直になる）
        if getattr(usdjpy_df.index, "tz", None) is not None:
            usdjpy_df.index = usdjpy_df.index.tz_localize(None)
        if getattr(us10y_df.index, "tz", None) is not None:
            us10y_df.index = us10y_df.index.tz_localize(None)

        return usdjpy_df, us10y_df

    except Exception as e:
        print("get_market_data ERROR:", e)
        return None, None


# --- 2. 指標計算 ---
def calculate_indicators(df, us10y):
    if df is None or us10y is None: return None
    new_df = df[['Open', 'High', 'Low', 'Close']].copy()
    
    # 指標の計算（bak版のSMA25, 75を維持しつつSMA5を追加）
    new_df['SMA_5'] = new_df['Close'].rolling(window=5).mean()
    new_df['SMA_25'] = new_df['Close'].rolling(window=25).mean()
    new_df['SMA_75'] = new_df['Close'].rolling(window=75).mean()
    
    delta = new_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    new_df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
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

# --- 4. 売買判断（詳細版） ---
def judge_condition(df):
    if df is None or len(df) < 2: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi, price = last['RSI'], last['Close']
    sma5, sma25, sma75 = last['SMA_5'], last['SMA_25'], last['SMA_75']

    # 中期判断
    if rsi > 70:
        mid_s, mid_c, mid_a = "‼️ 利益確定検討", "#ffeb3b", f"RSI({rsi:.1f})が70超。中期的な買われすぎ局面です。"
    elif rsi < 30:
        mid_s, mid_c, mid_a = "押し目買い検討", "#00bcd4", f"RSI({rsi:.1f})が30以下。中期的な仕込みの好機です。"
    elif sma25 > sma75 and prev['SMA_25'] <= prev['SMA_75']:
        mid_s, mid_c, mid_a = "強気・上昇開始", "#ccffcc", "ゴールデンクロス。中期トレンドが上向きに転換しました。"
    else:
        mid_s, mid_c, mid_a = "ステイ・静観", "#e0e0e0", "明確なシグナル待ち。FPの視点では無理なエントリーを避ける時期です。"

    # 短期判断
    if price > sma5:
        short_s, short_c, short_a = "上昇継続（短期）", "#e3f2fd", f"価格が5日線({sma5:.2f})の上を維持。勢いは強いです。"
    else:
        short_s, short_c, short_a = "勢い鈍化・調整", "#fce4ec", f"価格が5日線({sma5:.2f})を下回りました。短期的な調整局面です。"

    return {"short": {"status": short_s, "color": short_c, "advice": short_a}, "mid": {"status": mid_s, "color": mid_c, "advice": mid_a}, "price": price}

# --- 5. AI分析（FP1級・衆院選プロンプト完全版） ---
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
        
        # あなたの渾身のプロンプトを完全復元
        prompt = f"""
    あなたはFP1級を保持する、極めて優秀な為替戦略家です。
    特に今週は「衆議院選挙」を控えた極めて重要な1週間であることを強く認識してください。
    
    【市場データ】
    - ドル円価格: {p:.3f}円
    - 日米金利差(10年債): {u:.2f}%
    - ボラティリティ(ATR): {a:.3f}
    - SMA25乖離率: {s:.2f}%
    - RSI(14日): {r:.1f}

    【分析依頼：以下の4項目に沿ってFPに分かりやすく回答してください】
    1. 【ファンダメンタルズ】日米金利差の現状を「金融資産運用の利回り」の観点から解説
    2. 【地政学・外部要因】インフレや景気後退、政治リスクがどう影響しているか（FPの景気サイクルに基づき解説）特に今週は「衆議院選挙」を控えた極めて重要な1週間であることを強く認識してください。
    3. 【テクニカル】乖離率とRSI({r:.1f})から見て、今は「割安」か「割高」か。
    4. 【具体的戦略】NISAや外貨建資産のバランスを考える際のアドバイスのように、出口戦略（利確）を含めた今後1週間の戦略を提示

    【レポート構成：必ず以下の4項目に沿って記述してください】
    1. 現在の相場環境の要約
    2. 上記データ（特に金利差とボラティリティ）から読み解くリスク
    3. 具体的な戦略（エントリー・利確・損切の目安価格を具体的に提示）
    4. 経済カレンダーを踏まえた、今週の警戒イベントへの助言
    
    回答は親しみやすくも、プロの厳格さを感じる日本語でお願いします。
        """
        response = model.generate_content(prompt)
        return f"✅ 成功\n\n{response.text}"
    except Exception as e: return f"AI分析エラー: {str(e)}"

# --- bak版から復元した予想レンジ機能 ---
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

# --- bak版から復元したポートフォリオ機能 ---
def get_ai_portfolio(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p, u, s = context_data.get('price', 0.0), context_data.get('us10y', 0.0), context_data.get('sma_diff', 0.0)
        prompt = f"""
        あなたはFP1級技能士です。以下のデータに基づき、日本円、米ドル、ユーロ、豪ドル、英ポンドの最適な資産配分（合計100%）を提案してください。
        価格:{p:.2f}円, 金利差:{u:.2f}%, 乖離率:{s:.2f}%
        回答は必ず [日本円, 米ドル, ユーロ, 豪ドル, 英ポンド] の形式（数字のみ）で返してください。
        その後に、理由をFPの視点で簡潔に添えてください。
        """
        response = model.generate_content(prompt)
        return response.text
    except: return "ポートフォリオ分析に失敗しました。"














