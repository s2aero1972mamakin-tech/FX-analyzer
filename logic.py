import yfinance as yf
import pandas as pd
import google.generativeai as genai
import pytz
import requests
import time
from datetime import datetime

TOKYO = pytz.timezone("Asia/Tokyo")

# 取得失敗時の理由をここに残す（main.pyで表示できる）
LAST_FETCH_ERROR = ""

# -----------------------------
# 超軽量TTLキャッシュ（Streamlit再実行対策）
# -----------------------------
# key -> (expire_epoch, value)
_TTL_CACHE = {}


def _cache_get(key):
    try:
        exp, val = _TTL_CACHE.get(key, (0, None))
        if time.time() <= exp:
            return val
    except Exception:
        pass
    return None


def _cache_set(key, val, ttl_sec):
    try:
        _TTL_CACHE[key] = (time.time() + float(ttl_sec), val)
    except Exception:
        pass


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


def _requests_get_json(url, params=None, timeout=15):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    return r, r.json() if r.status_code == 200 else None


# =====================================================
# Yahoo Chart API 直叩きフォールバック（TTLキャッシュ付き）
# =====================================================
def _yahoo_chart(symbol: str, rng: str = "1y", interval: str = "1d", ttl_sec: int = 900):
    cache_key = f"yahoo_chart::{symbol}::{rng}::{interval}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": rng, "interval": interval}
        r, j = _requests_get_json(url, params=params, timeout=15)

        if r.status_code != 200 or j is None:
            preview = ""
            try:
                preview = (r.text or "")[:120]
            except Exception:
                preview = ""
            _set_err(f"Yahoo chart HTTP {r.status_code}: {preview}")
            _cache_set(cache_key, None, 30)
            return None

        res = j.get("chart", {}).get("result", None)
        if not res:
            _set_err(f"Yahoo chart no result: {j.get('chart', {}).get('error')}")
            _cache_set(cache_key, None, 30)
            return None

        res0 = res[0]
        ts = res0.get("timestamp", [])
        quote = res0.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quote:
            _set_err("Yahoo chart missing timestamp/quote")
            _cache_set(cache_key, None, 30)
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

        df = df.dropna(subset=["Close"])
        if df.empty:
            _set_err("Yahoo chart df empty after dropna")
            _cache_set(cache_key, None, 30)
            return None

        _cache_set(cache_key, df, ttl_sec)
        return df

    except Exception as e:
        _set_err(f"Yahoo chart exception: {e}")
        _cache_set(cache_key, None, 30)
        return None


# =====================================================
# 最新為替レート
# =====================================================
def get_latest_quote(symbol="JPY=X"):
    df = _yahoo_chart(symbol, rng="1d", interval="1m", ttl_sec=60)
    if df is not None and not df.empty:
        price = float(df["Close"].iloc[-1])
        qt = pd.Timestamp(df.index[-1]).tz_localize(TOKYO)
        return price, _to_jst(qt)

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
# 市場データ取得
# =====================================================
def get_market_data(period="1y"):
    usdjpy_df = None
    us10y_df = None

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

    if usdjpy_df is None or getattr(usdjpy_df, "empty", True):
        try:
            usdjpy_df = yf.download("JPY=X", period=period, interval="1d", progress=False, threads=False)
        except Exception:
            usdjpy_df = None

    if us10y_df is None or getattr(us10y_df, "empty", True):
        try:
            us10y_df = yf.download("^TNX", period=period, interval="1d", progress=False, threads=False)
        except Exception:
            us10y_df = None

    if usdjpy_df is None or getattr(usdjpy_df, "empty", True):
        usdjpy_df = _yahoo_chart("JPY=X", rng=period, interval="1d", ttl_sec=900)

    if us10y_df is None or getattr(us10y_df, "empty", True):
        us10y_df = _yahoo_chart("^TNX", rng=period, interval="1d", ttl_sec=900)

    if usdjpy_df is None or getattr(usdjpy_df, "empty", True):
        if not LAST_FETCH_ERROR:
            _set_err("All sources failed for JPY=X")
        return None, None

    return usdjpy_df, us10y_df


# =====================================================
# 指標計算
# =====================================================
def calculate_indicators(df, us10y):
    if df is None or getattr(df, "empty", True):
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] for c in df.columns]
    except Exception:
        pass

    need_cols = ["Open", "High", "Low", "Close"]
    for c in need_cols:
        if c not in df.columns:
            return None

    new_df = df[need_cols].copy()

    for c in need_cols:
        if isinstance(new_df[c], pd.DataFrame):
            new_df[c] = new_df[c].iloc[:, 0]
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")

    new_df = new_df.dropna(subset=["Close"])
    if new_df.empty:
        return None

    new_df["SMA_5"] = new_df["Close"].rolling(5).mean()
    new_df["SMA_25"] = new_df["Close"].rolling(25).mean()
    new_df["SMA_75"] = new_df["Close"].rolling(75).mean()

    delta = new_df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    new_df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    high_low = new_df["High"] - new_df["Low"]
    high_close = (new_df["High"] - new_df["Close"].shift()).abs()
    low_close = (new_df["Low"] - new_df["Close"].shift()).abs()
    new_df["ATR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    new_df["SMA_DIFF"] = (new_df["Close"] - new_df["SMA_25"]) / new_df["SMA_25"] * 100

    if us10y is not None and not getattr(us10y, "empty", True):
        try:
            if isinstance(us10y.columns, pd.MultiIndex):
                us10y = us10y.copy()
                us10y.columns = [c[0] for c in us10y.columns]

            if "Close" in us10y.columns:
                s = us10y["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce")
                new_df["US10Y"] = s.reindex(new_df.index).ffill()
            else:
                new_df["US10Y"] = float("nan")
        except Exception:
            new_df["US10Y"] = float("nan")
    else:
        new_df["US10Y"] = float("nan")

    return new_df


# =====================================================
# 通貨強弱
# =====================================================
def get_currency_strength():
    pairs = {"日本円": "JPY=X", "ユーロ": "EURUSD=X", "英ポンド": "GBPUSD=X", "豪ドル": "AUDUSD=X"}
    strength_data = pd.DataFrame()

    for name, sym in pairs.items():
        try:
            d = None
            try:
                t = yf.Ticker(sym).history(period="1mo")
                if t is not None and not t.empty:
                    d = t["Close"]
            except Exception:
                d = None

            if d is None or len(d) == 0:
                tmp = _yahoo_chart(sym, rng="1mo", interval="1d", ttl_sec=900)
                if tmp is not None and not tmp.empty:
                    d = tmp["Close"]

            if d is None or len(d) == 0:
                continue

            d.index = pd.to_datetime(d.index)
            if name == "日本円":
                strength_data[name] = (1 / d).pct_change().cumsum() * 100
            else:
                strength_data[name] = d.pct_change().cumsum() * 100
        except Exception:
            pass

    if not strength_data.empty:
        # 他のAIが指摘していた点を考慮しつつ、米ドル（USD）を追加
        strength_data["米ドル"] = strength_data.mean(axis=1) * -1
        return strength_data.ffill().dropna()

    return strength_data


# =====================================================
# 判定ロジック
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
# AI分析（FP1級・衆院選プロンプト完全版）
# =====================================================
def get_active_model(api_key):
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                return m.name
    except Exception:
        pass
    return "models/gemini-1.5-flash"


def get_ai_analysis(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p, u, a, s, r = (
            context_data.get("price", 0.0),
            context_data.get("us10y", 0.0),
            context_data.get("atr", 0.0),
            context_data.get("sma_diff", 0.0),
            context_data.get("rsi", 50.0),
        )

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
2. 【地政学・外部要因】インフレや景気後退、政治リスクがどう影響しているか
   （FPの景気サイクルに基づき解説）
   特に今週は「衆議院選挙」を控えた極めて重要な1週間であることを強く認識してください。
3. 【テクニカル】乖離率とRSI({r:.1f})から見て、今は「割安」か「割高」か。
4. 【具体的戦略】NISAや外貨建資産のバランスを考える際のアアドバイスのように、
   出口戦略（利確）を含めた今後1週間の戦略を提示

【レポート構成：必ず以下の4項目に沿って記述してください】
1. 現在の相場環境の要約
2. 上記データ（特に金利差とボラティリティ）から読み解くリスク
3. 具体的な戦略（エントリー・利確・損切の目安価格を具体的に提示）
4. 経済カレンダーを踏まえた、今週の警戒イベントへの助言

回答は親しみやすくも、プロの厳格さを感じる日本語でお願いします。
        """

    except Exception as e:
        return f"AI分析エラー: {str(e)}"


def get_ai_portfolio(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        prompt = f"""
        あなたはFP1級技能士です。
        日本円、米ドル、ユーロ、豪ドル、英ポンドの
        最適配分（合計100%）を
        [日本円, 米ドル, ユーロ, 豪ドル, 英ポンド]
        形式で提示してください。
        その際、各通貨の現状と今後の見通しを含めて解説してください。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "ポートフォリオ分析に失敗しました。"

# main.pyで使用されているため維持
def get_ai_range(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        prompt = f"現在のドル円は {p:.2f}円です。今後1週間の[最高値, 最安値]を半角数字のみで返してください。"
        res = model.generate_content(prompt).text
        import re
        nums = re.findall(r"\d+\.\d+|\d+", res)
        return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None
    except:
        return None

# =====================================================
# ✅ 新規追加: ロボ的注文戦略生成
# =====================================================
def get_ai_order_strategy(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        a = context_data.get('atr', 0.0)
        r = context_data.get('rsi', 50.0)
        s = context_data.get('sma25', 0.0)
        
        prompt = f"""
あなたはFX投資ロボットの意思決定エンジンです。
以下のデータに基づき、最も利益期待値の高い「具体的な注文票」を作成してください。

【現在のデータ】
- 現在価格: {p:.3f} 円
- SMA25(25日線): {s:.3f} 円
- ATR(ボラティリティ): {a:.3f} 円
- RSI: {r:.1f}

【出力ルール】
1. 推奨注文方式を1つ選択（指値, IFD, OCO, IFDOCO）
2. エントリー価格、利確価格、損切価格をすべて「円」で明示
3. なぜその価格設定にしたのか、ATRやSMA25を引用して短く論理的に説明

【出力フォーマット例】
🤖 **ロボ指示：IFDOCO注文を推奨**
- **ENTRY**: 150.120 円
- **LIMIT (利確)**: 152.000 円
- **STOP (損切)**: 149.500 円
- **根拠**: 現在はSMA25を上抜けており、ATR1.5倍の幅を損切に設定。RSI過熱前までの上昇を狙います。
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"注文戦略生成エラー: {str(e)}"
