import yfinance as yf
import pandas as pd
import google.generativeai as genai
import datetime
import pytz
import requests

BUILD_ID = "2026-02-02_02"
TOKYO = pytz.timezone("Asia/Tokyo")

# 取得失敗時の理由をここに残す（main.pyで表示できる）
LAST_FETCH_ERROR = ""


def _set_err(msg: str):
    global LAST_FETCH_ERROR
    LAST_FETCH_ERROR = msg


def _to_jst(ts):
    if ts is None:
        return None
    try:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(TOKYO)
    except Exception:
        return ts


# =====================================================
# Yahoo Chart API 直叩きフォールバック
# =====================================================
def _yahoo_chart(symbol: str, rng: str = "1y", interval: str = "1d"):
    """
    Yahoo Finance unofficial chart API.
    symbol: "JPY=X" or "^TNX"
    rng: "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"
    interval: "1m","2m","5m","15m","30m","60m","90m","1d","1wk","1mo"
    """
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": rng, "interval": interval}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            _set_err(f"Yahoo chart HTTP {r.status_code}: {r.text[:120]}")
            return None

        j = r.json()
        res = j.get("chart", {}).get("result", None)
        if not res:
            _set_err(f"Yahoo chart no result: {j.get('chart', {}).get('error')}")
            return None

        res0 = res[0]
        ts = res0.get("timestamp", [])
        quote = res0.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quote:
            _set_err("Yahoo chart missing timestamp/quote")
            return None

        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(TOKYO).tz_localize(None)

        df = pd.DataFrame(
            {
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            },
            index=idx,
        )
        # 欠損が混ざるので掃除
        df = df.dropna(subset=["Close"])
        if df.empty:
            _set_err("Yahoo chart df empty after dropna")
            return None
        return df
    except Exception as e:
        _set_err(f"Yahoo chart exception: {e}")
        return None


# =====================================================
# 最新為替レート（3段 + Yahoo直叩き）
# =====================================================
def get_latest_quote(symbol="JPY=X"):
    """
    最新価格と時刻(JST)を返す。
    yfinance全滅時でも Yahoo chart API で返す。
    """
    # 1) yfinance fast_info
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info or {}
        price = fi.get("last_price") or fi.get("lastPrice")
        ts = fi.get("last_timestamp") or fi.get("lastTimestamp")
        if price is not None and ts:
            qt = pd.to_datetime(ts, unit="s", utc=True)
            return float(price), _to_jst(qt)
    except Exception:
        pass

    # 2) yfinance intraday
    try:
        intraday = yf.download(symbol, period="1d", interval="1m", progress=False, threads=False)
        if intraday is not None and not intraday.empty:
            close = intraday["Close"].dropna()
            if len(close) > 0:
                price = float(close.iloc[-1])
                qt = close.index[-1]
                qt = pd.Timestamp(qt)
                if getattr(qt, "tzinfo", None) is None:
                    qt = qt.tz_localize("UTC")
                return price, _to_jst(qt)
    except Exception:
        pass

    # 3) Yahoo chart API (1d, 1m)
    df = _yahoo_chart(symbol, rng="1d", interval="1m")
    if df is not None and not df.empty:
        price = float(df["Close"].iloc[-1])
        qt = pd.Timestamp(df.index[-1]).tz_localize(TOKYO)
        return price, _to_jst(qt)

    # 4) 為替APIフォールバック（USD→JPY）
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if r.status_code == 200:
            rate = r.json()["rates"]["JPY"]
            qt = pd.Timestamp.utcnow().tz_localize("UTC")
            return float(rate), _to_jst(qt)
    except Exception:
        pass

    return None, None


# =====================================================
# 市場データ取得（yfinance→Yahoo chartへフォールバック）
# =====================================================
def get_market_data(period="1y"):
    """
    usdjpy_df, us10y_df を返す。
    どちらかが取れなくても usdjpy さえあれば返す（us10yはNoneでもOK）。
    """
    usdjpy_df = None
    us10y_df = None

    # 1) yfinance history
    try:
        usdjpy_df = yf.Ticker("JPY=X").history(period=period)
        if usdjpy_df is not None and not usdjpy_df.empty:
            if getattr(usdjpy_df.index, "tz", None) is not None:
                usdjpy_df.index = usdjpy_df.index.tz_localize(None)
    except Exception:
        usdjpy_df = None

    try:
        us10y_df = yf.Ticker("^TNX").history(period=period)
        if us10y_df is not None and not us10y_df.empty:
            if getattr(us10y_df.index, "tz", None) is not None:
                us10y_df.index = us10y_df.index.tz_localize(None)
    except Exception:
        us10y_df = None

    # 2) yfinance download fallback
    if usdjpy_df is None or usdjpy_df.empty:
        try:
            usdjpy_df = yf.download("JPY=X", period=period, interval="1d", progress=False, threads=False)
        except Exception:
            usdjpy_df = None

    if us10y_df is None or us10y_df.empty:
        try:
            us10y_df = yf.download("^TNX", period=period, interval="1d", progress=False, threads=False)
        except Exception:
            us10y_df = None

    # 3) Yahoo chart API fallback（ここが今回の本命）
    if usdjpy_df is None or getattr(usdjpy_df, "empty", True):
        usdjpy_df = _yahoo_chart("JPY=X", rng=period, interval="1d")

    if us10y_df is None or getattr(us10y_df, "empty", True):
        us10y_df = _yahoo_chart("^TNX", rng=period, interval="1d")

    # usdjpyが取れないなら終了
    if usdjpy_df is None or getattr(usdjpy_df, "empty", True):
        if not LAST_FETCH_ERROR:
            _set_err("All sources failed for JPY=X")
        return None, None

    # 最新クオートをCloseに反映（取れた場合のみ）
    q_price, q_time = get_latest_quote("JPY=X")
    if q_price is not None and "Close" in usdjpy_df.columns:
        try:
            usdjpy_df.iloc[-1, usdjpy_df.columns.get_loc("Close")] = float(q_price)
        except Exception:
            pass

    return usdjpy_df, us10y_df


# =====================================================
# 指標計算（US10Yが無くても動く）
# =====================================================
def calculate_indicators(df, us10y):
    if df is None or getattr(df, "empty", True):
        return None

    need_cols = ["Open", "High", "Low", "Close"]
    for c in need_cols:
        if c not in df.columns:
            return None

    new_df = df[need_cols].copy()

    new_df["SMA_5"] = new_df["Close"].rolling(window=5).mean()
    new_df["SMA_25"] = new_df["Close"].rolling(window=25).mean()
    new_df["SMA_75"] = new_df["Close"].rolling(window=75).mean()

    delta = new_df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    new_df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    high_low = new_df["High"] - new_df["Low"]
    high_close = (new_df["High"] - new_df["Close"].shift()).abs()
    low_close = (new_df["Low"] - new_df["Close"].shift()).abs()
    new_df["ATR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(window=14).mean()

    new_df["SMA_DIFF"] = (new_df["Close"] - new_df["SMA_25"]) / new_df["SMA_25"] * 100

    # US10Y（無ければNaN列）
    if us10y is not None and (not getattr(us10y, "empty", True)) and ("Close" in us10y.columns):
        new_df["US10Y"] = us10y["Close"].reindex(new_df.index).ffill()
    else:
        new_df["US10Y"] = float("nan")

    return new_df


# =====================================================
# 通貨強弱
# =====================================================
def get_currency_strength():
    pairs = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "AUD": "AUDUSD=X"}
    strength_data = pd.DataFrame()
    for name, sym in pairs.items():
        try:
            # yfinanceが死んでる環境向けに Yahoo chart API も使う
            d = None
            try:
                t = yf.Ticker(sym).history(period="1mo")
                if t is not None and not t.empty:
                    d = t["Close"]
            except Exception:
                d = None

            if d is None or len(d) == 0:
                tmp = _yahoo_chart(sym, rng="1mo", interval="1d")
                if tmp is not None and not tmp.empty:
                    d = tmp["Close"]

            if d is None or len(d) == 0:
                continue

            d.index = pd.to_datetime(d.index)
            if name == "JPY":
                strength_data[name] = (1 / d).pct_change().cumsum() * 100
            else:
                strength_data[name] = d.pct_change().cumsum() * 100
        except Exception:
            pass

    return strength_data


# =====================================================
# 判定ロジック（あなたの版を維持）
# =====================================================
def judge_condition(df):
    if df is None or len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi, price = last["RSI"], last["Close"]
    sma5, sma25, sma75 = last["SMA_5"], last["SMA_25"], last["SMA_75"]

    if rsi > 70:
        mid_s, mid_c, mid_a = "‼️ 利益確定検討", "#ffeb3b", f"RSI({rsi:.1f})が70超。中期的な買われすぎ局面です。"
    elif rsi < 30:
        mid_s, mid_c, mid_a = "押し目買い検討", "#00bcd4", f"RSI({rsi:.1f})が30以下。中期的な仕込みの好機です。"
    elif sma25 > sma75 and prev["SMA_25"] <= prev["SMA_75"]:
        mid_s, mid_c, mid_a = "強気・上昇開始", "#ccffcc", "ゴールデンクロス。中期トレンドが上向きに転換しました。"
    else:
        mid_s, mid_c, mid_a = "ステイ・静観", "#e0e0e0", "明確なシグナル待ち。FPの視点では無理なエントリーを避ける時期です。"

    if price > sma5:
        short_s, short_c, short_a = "上昇継続（短期）", "#e3f2fd", f"価格が5日線({sma5:.2f})の上を維持。勢いは強いです。"
    else:
        short_s, short_c, short_a = "勢い鈍化・調整", "#fce4ec", f"価格が5日線({sma5:.2f})を下回りました。短期的な調整局面です。"

    return {
        "short": {"status": short_s, "color": short_c, "advice": short_a},
        "mid": {"status": mid_s, "color": mid_c, "advice": mid_a},
        "price": price,
    }


# =====================================================
# --- AI分析群（あなたのコードを完全保持） ---
# =====================================================
def get_active_model(api_key):
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except:
        pass
    return "models/gemini-1.5-flash"


def get_ai_analysis(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p, u, a, s, r = context_data.get('price', 0.0), context_data.get('us10y', 0.0), context_data.get('atr', 0.0), context_data.get('sma_diff', 0.0), context_data.get('rsi', 50.0)

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
    except Exception as e:
        return f"AI分析エラー: {str(e)}"


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
    except:
        return None


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
    except:
        return "ポートフォリオ分析に失敗しました。"
