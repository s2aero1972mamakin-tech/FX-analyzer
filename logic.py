import yfinance as yf
import pandas as pd
import google.generativeai as genai
import pytz
import requests
import time
from datetime import datetime
import re
import json

TOKYO = pytz.timezone("Asia/Tokyo")

# 取得失敗時の理由をここに残す
LAST_FETCH_ERROR = ""

# -----------------------------
# ✅ 【追加】通貨ペア設定 (全通貨対応の基盤)
# -----------------------------
PAIR_MAP = {
    "USD/JPY (ドル円)": "JPY=X",
    "EUR/USD (ユーロドル)": "EURUSD=X",
    "AUD/JPY (豪ドル円)": "AUDJPY=X",
    "GBP/JPY (ポンド円)": "GBPJPY=X",
    "EUR/JPY (ユーロ円)": "EURJPY=X",
    "AUD/USD (豪ドル米ドル)": "AUDUSD=X"
}

# -----------------------------
# AI予想レンジ 自動取得キャッシュ
# -----------------------------
AI_RANGE_TTL_SEC = 60 * 60 * 72
_AI_RANGE_CACHE = {"expire": 0.0, "value": None}

def ensure_ai_range(api_key: str, context_data: dict, force: bool = False):
    now = time.time()
    if (not force) and _AI_RANGE_CACHE.get("value") and now <= float(_AI_RANGE_CACHE.get("expire", 0.0) or 0.0):
        return _AI_RANGE_CACHE["value"]

    getrng = globals().get("get_ai_range")
    if not callable(getrng):
        return None

    rng = getrng(api_key, context_data)
    if isinstance(rng, dict) and rng.get("low") is not None and rng.get("high") is not None:
        try:
            low = float(rng["low"]); high = float(rng["high"])
        except Exception:
            return None
        if low > high:
            low, high = high, low
        out = {"low": low, "high": high, "why": str(rng.get("why", ""))}
        _AI_RANGE_CACHE["value"] = out
        _AI_RANGE_CACHE["expire"] = now + AI_RANGE_TTL_SEC
        return out
    return None

# -----------------------------
# JSON固定出力: パース/検証ヘルパ
# -----------------------------
def _extract_json_block(text: str) -> str:
    if not text: return ""
    s = text.strip()
    if s.startswith("{") and s.endswith("}"): return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return ""

def _safe_float(x, default=None):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception: return default

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _validate_order_json(obj: dict, ctx: dict) -> (bool, list):
    reasons = []
    if not isinstance(obj, dict): return False, ["order_json_not_object"]

    decision = obj.get("decision")
    if decision not in ("TRADE", "NO_TRADE"): reasons.append("decision_invalid")
    if decision == "NO_TRADE": return True, reasons

    side = obj.get("side")
    if side not in ("LONG", "SHORT"): reasons.append("side_invalid")

    entry = _safe_float(obj.get("entry"))
    tp = _safe_float(obj.get("take_profit"))
    sl = _safe_float(obj.get("stop_loss"))
    if entry is None: reasons.append("entry_missing")
    if tp is None: reasons.append("take_profit_missing")
    if sl is None: reasons.append("stop_loss_missing")

    horizon = obj.get("horizon")
    if horizon not in ("WEEK", "MONTH"): reasons.append("horizon_invalid")

    conf = _safe_float(obj.get("confidence"), default=0.0)
    if conf is None: reasons.append("confidence_missing")
    else: obj["confidence"] = _clamp(conf, 0.0, 1.0)

    if entry is not None and tp is not None and sl is not None and side in ("LONG","SHORT"):
        if side == "LONG":
            if not (sl < entry < tp): reasons.append("levels_inconsistent_long")
        else:
            if not (tp < entry < sl): reasons.append("levels_inconsistent_short")

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0: reasons.append("risk_nonpositive")
        else:
            rr = reward / risk
            obj["rr_ratio"] = rr
            if rr < 1.1: reasons.append("rr_too_low")

        p = _safe_float(ctx.get("price"), default=entry)
        if p is not None:
            if abs(entry - p) / max(p, 1e-6) > 0.03:
                reasons.append("entry_too_far_from_price")

    if "why" not in obj: obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list): obj["notes"] = []
    return (len(reasons) == 0), reasons

def _validate_regime_json(obj: dict) -> (bool, list):
    reasons = []
    if not isinstance(obj, dict): return False, ["regime_json_not_object"]
    regime = obj.get("market_regime")
    if regime not in ("DEFENSIVE", "OPPORTUNITY"): reasons.append("market_regime_invalid")
    conf = _safe_float(obj.get("confidence"), default=0.0)
    obj["confidence"] = _clamp(conf, 0.0, 1.0)
    if "why" not in obj: obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list): obj["notes"] = []
    return (len(reasons) == 0), reasons

def _validate_weekend_json(obj: dict) -> (bool, list):
    reasons = []
    if not isinstance(obj, dict): return False, ["weekend_json_not_object"]
    action = obj.get("action")
    if action not in ("TAKE_PROFIT","CUT_LOSS","HOLD_WEEK","HOLD_MONTH","NO_POSITION"): reasons.append("action_invalid")
    if "why" not in obj: obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list): obj["notes"] = []
    if "levels" not in obj or not isinstance(obj.get("levels"), dict):
        obj["levels"] = {"take_profit": 0, "stop_loss": 0, "trail": 0}
    return (len(reasons) == 0), reasons

# -----------------------------
# NO_TRADEゲート
# -----------------------------
_NO_TRADE_THRESHOLDS = {
    "DEFENSIVE": {"sma_diff_pct": 0.20, "rsi_lo": 45.0, "rsi_hi": 55.0, "atr_mult": 1.6, "ma_close_pct": 0.10},
    "OPPORTUNITY": {"sma_diff_pct": 0.15, "rsi_lo": 48.0, "rsi_hi": 52.0, "atr_mult": 1.9, "ma_close_pct": 0.08},
}

def no_trade_gate(context_data: dict, market_regime: str, force_defensive: bool = False):
    reasons = []
    regime = "DEFENSIVE" if force_defensive else (market_regime if market_regime in _NO_TRADE_THRESHOLDS else "DEFENSIVE")
    th = _NO_TRADE_THRESHOLDS[regime]

    price = _safe_float(context_data.get("price"))
    
    # ✅ 【追加】スプレッド/窓開け検知 (月曜対策)
    # 始値と現在値が大きく乖離(0.5円=50pips)している場合は、スプレッド拡大中とみなす
    open_price = _safe_float(context_data.get("open_price"), default=price)
    if price and open_price and abs(price - open_price) > 0.5:
         reasons.append("HIGH_VOLATILITY_WARNING(Gap/Spread)")

    sma25 = _safe_float(context_data.get("sma25"))
    sma75 = _safe_float(context_data.get("sma75"))
    rsi = _safe_float(context_data.get("rsi"))
    atr = _safe_float(context_data.get("atr"))
    atr_avg60 = _safe_float(context_data.get("atr_avg60"))

    for k,v in [("price",price),("sma25",sma25),("sma75",sma75),("rsi",rsi),("atr",atr)]:
        if v is None or v != v: reasons.append(f"data_invalid_{k}")

    if reasons: return True, regime, reasons

    sma_diff_pct = abs(sma25 - sma75) / max(price, 1e-6) * 100.0
    if sma_diff_pct < th["sma_diff_pct"] and (th["rsi_lo"] <= rsi <= th["rsi_hi"]):
        reasons.append("no_direction_ma_converge_and_rsi_neutral")

    if sma_diff_pct < th["ma_close_pct"]:
        reasons.append("ma25_ma75_too_close")

    if atr_avg60 is not None and atr_avg60 > 0:
        if atr > atr_avg60 * th["atr_mult"]:
            reasons.append("volatility_too_high_atr_spike")

    return (len(reasons) > 0), regime, reasons

# -----------------------------
# 超軽量TTLキャッシュ
# -----------------------------
_TTL_CACHE = {}

def _cache_get(key):
    try:
        exp, val = _TTL_CACHE.get(key, (0, None))
        if time.time() <= exp: return val
    except Exception: pass
    return None

def _cache_set(key, val, ttl_sec):
    try: _TTL_CACHE[key] = (time.time() + float(ttl_sec), val)
    except Exception: pass

def _set_err(msg: str):
    global LAST_FETCH_ERROR
    LAST_FETCH_ERROR = msg

def _to_jst(ts):
    if ts is None: return None
    try:
        if getattr(ts, "tzinfo", None) is None: ts = ts.tz_localize("UTC")
        return ts.tz_convert(TOKYO)
    except Exception: return ts

def _requests_get_json(url, params=None, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        return r, r.json() if r.status_code == 200 else None
    except: return None, None

# =====================================================
# Yahoo Chart API
# =====================================================
def _yahoo_chart(symbol: str, rng: str = "1y", interval: str = "1d", ttl_sec: int = 900):
    cache_key = f"yahoo_chart::{symbol}::{rng}::{interval}"
    cached = _cache_get(cache_key)
    if cached is not None: return cached

    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": rng, "interval": interval}
        r, j = _requests_get_json(url, params=params, timeout=15)

        if r.status_code != 200 or j is None:
            _set_err(f"Yahoo chart HTTP {r.status_code}")
            _cache_set(cache_key, None, 30); return None

        res = j.get("chart", {}).get("result", None)
        if not res:
            _set_err("Yahoo chart no result"); _cache_set(cache_key, None, 30); return None

        res0 = res[0]
        ts = res0.get("timestamp", [])
        quote = res0.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not quote:
            _set_err("Yahoo chart missing timestamp/quote"); _cache_set(cache_key, None, 30); return None

        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(TOKYO).tz_localize(None)
        df = pd.DataFrame({
            "Open": quote.get("open", []), "High": quote.get("high", []),
            "Low": quote.get("low", []), "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=idx).dropna(subset=["Close"])
        
        if df.empty: _set_err("Empty df"); _cache_set(cache_key, None, 30); return None

        _cache_set(cache_key, df, ttl_sec)
        return df

    except Exception as e:
        _set_err(f"Exception: {e}"); _cache_set(cache_key, None, 30); return None

def get_latest_quote(symbol="JPY=X"):
    df = _yahoo_chart(symbol, rng="1d", interval="1m", ttl_sec=60)
    if df is not None and not df.empty:
        price = float(df["Close"].iloc[-1])
        qt = pd.Timestamp(df.index[-1]).tz_localize(TOKYO)
        return price, _to_jst(qt)
    return None, None

# =====================================================
# ✅ 【修正】市場データ取得 (汎用化: symbol引数追加)
# =====================================================
def get_market_data(period="1y", symbol="JPY=X"):
    main_df = None
    us10y_df = None

    # メイン通貨
    try:
        main_df = yf.Ticker(symbol).history(period=period)
        if getattr(main_df.index, "tz", None): main_df.index = main_df.index.tz_localize(None)
    except: pass

    # 米10年債
    try:
        us10y_df = yf.Ticker("^TNX").history(period=period)
        if getattr(us10y_df.index, "tz", None): us10y_df.index = us10y_df.index.tz_localize(None)
    except: pass

    if main_df is None or getattr(main_df, "empty", True):
        main_df = _yahoo_chart(symbol, rng=period, interval="1d", ttl_sec=900)
    if us10y_df is None or getattr(us10y_df, "empty", True):
        us10y_df = _yahoo_chart("^TNX", rng=period, interval="1d", ttl_sec=900)

    if main_df is None or getattr(main_df, "empty", True):
        if not LAST_FETCH_ERROR: _set_err(f"All sources failed for {symbol}")
        return None, None

    return main_df, us10y_df

# =====================================================
# 指標計算
# =====================================================
def calculate_indicators(df, us10y):
    if df is None or getattr(df, "empty", True): return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] for c in df.columns]
    except: pass

    new_df = df[["Open", "High", "Low", "Close"]].copy()
    for c in new_df.columns:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    new_df = new_df.dropna()

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

    if us10y is not None:
        try:
            s = us10y["Close"]
            if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
            new_df["US10Y"] = s.reindex(new_df.index).ffill()
        except: new_df["US10Y"] = float("nan")
    else: new_df["US10Y"] = float("nan")

    return new_df

# =====================================================
# 通貨強弱
# =====================================================
def get_currency_strength():
    # 簡易比較用
    pairs = {"JPY": "JPY=X", "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "AUD": "AUDUSD=X"}
    strength_data = pd.DataFrame()
    for name, sym in pairs.items():
        try:
            d = _yahoo_chart(sym, rng="1mo", interval="1d")
            if d is not None and not d.empty:
                val = d["Close"]
                if name == "JPY": strength_data[name] = (1/val).pct_change().cumsum()*100
                else: strength_data[name] = val.pct_change().cumsum()*100
        except: pass
    if not strength_data.empty:
        strength_data["USD"] = strength_data.mean(axis=1) * -1
        return strength_data.ffill().dropna()
    return strength_data

# =====================================================
# 判定ロジック
# =====================================================
def judge_condition(df):
    if df is None or len(df) < 2: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi, price = last["RSI"], last["Close"]
    sma5, sma25, sma75 = last["SMA_5"], last["SMA_25"], last["SMA_75"]

    if rsi > 70: mid_s, mid_c, mid_a = "‼️ 利益確定検討", "#ffeb3b", f"RSI({rsi:.1f})過熱。買われすぎ。"
    elif rsi < 30: mid_s, mid_c, mid_a = "押し目買い検討", "#00bcd4", f"RSI({rsi:.1f})底値圏。仕込み好機。"
    elif sma25 > sma75 and prev["SMA_25"] <= prev["SMA_75"]: mid_s, mid_c, mid_a = "強気・上昇開始", "#ccffcc", "ゴールデンクロス発生。"
    else: mid_s, mid_c, mid_a = "ステイ・静観", "#e0e0e0", "明確なシグナル待ち。"

    if price > sma5: short_s, short_c, short_a = "上昇継続（短期）", "#e3f2fd", f"5日線({sma5:.2f})の上を推移中。"
    else: short_s, short_c, short_a = "勢い鈍化・調整", "#fce4ec", f"5日線({sma5:.2f})割れ。調整注意。"

    return {
        "short": {"status": short_s, "color": short_c, "advice": short_a},
        "mid": {"status": mid_s, "color": mid_c, "advice": mid_a},
        "price": price,
    }

# =====================================================
# AI分析基盤
# =====================================================
def get_active_model(api_key):
    genai.configure(api_key=api_key)
    try:
        return "gemini-2.0-flash-exp" # 最新モデル優先
    except:
        return "gemini-1.5-flash"

# -----------------------------
# ✅ 【追加】AIスキャナー (全ペア分析)
# -----------------------------
def scan_best_pair(api_key):
    """全ペアをスキャンしてベストを推奨"""
    model_name = get_active_model(api_key)
    if not model_name: return None
    market_summary = ""
    for name, sym in PAIR_MAP.items():
        try:
            df = _yahoo_chart(sym, rng="5d", interval="1d")
            if df is None or df.empty: continue
            close = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            chg = (close - prev) / prev * 100
            trend = "上昇" if close > df["Close"].mean() else "下降"
            market_summary += f"- {name}: Price={close:.3f} / Chg={chg:+.2f}% / Trend={trend}\n"
        except: continue

    prompt = f"""
あなたはプロのFXファンドマネージャーです。
以下の市場データから「今週、最も利益チャンスがありそうなペア」を1つだけ選定し、推奨してください。
特に USD/JPY が政治イベントで膠着する可能性がある場合、代わりのペア(AUD/JPYなど)を優先してください。

【市場概況】
{market_summary}

【出力】JSONのみ
{{
  "best_pair_name": "通貨ペア名(例: AUD/JPY)",
  "reason": "選定理由(簡潔に)",
  "confidence": 0.0〜1.0
}}
"""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return _extract_json_block(resp.text)
    except: return None

# -----------------------------
# AI戦略 (汎用化対応)
# -----------------------------
def get_ai_market_regime(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get("price", 0.0)
        prompt = f"""
市場環境を DEFENSIVE (守り) か OPPORTUNITY (攻め) か判定せよ。
現在価格: {p}
RSI: {context_data.get('rsi', 50)}
ATR: {context_data.get('atr', 0)}

JSON出力: {{ "market_regime": "DEFENSIVE" or "OPPORTUNITY", "confidence": 0.8, "why": "..." }}
"""
        resp = model.generate_content(prompt)
        j = _extract_json_block(resp.text)
        return json.loads(j) if j else {}
    except: return {}

def get_ai_analysis(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get("price", 0.0)
        u = context_data.get("us10y", 0.0)
        r = context_data.get("rsi", 50.0)
        
        # ✅ 【修正】ペア名を含めた汎用プロンプト
        prompt = f"""
あなたはFP1級の金融ストラテジストです。
現在価格: {p}
10年債金利: {u}%
RSI: {r}

【指示】
1. 現在の重要ファンダメンタルズ（選挙、政策金利、経済指標など）を考慮し、市場環境を解説してください。
2. テクニカル面での割安/割高感を判定してください。
3. 向こう1週間の具体的なトレード戦略を提示してください。
"""
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e: return f"Error: {str(e)}"

# -----------------------------
# ✅ 【修正】注文命令 (pair_name引数追加)
# -----------------------------
def get_ai_order_strategy(api_key, context_data, override_mode="AUTO", override_reason="", pair_name="USD/JPY"):
    # 緊急停止
    if override_mode == "FORCE_NO_TRADE":
        return {"decision":"NO_TRADE", "why": "緊急停止", "notes":["manual_stop"]}

    regime_obj = get_ai_market_regime(api_key, context_data)
    market_regime = regime_obj.get("market_regime", "DEFENSIVE")
    
    is_no, regime, gate_reasons = no_trade_gate(context_data, market_regime, force_defensive=(override_mode=="FORCE_DEFENSIVE"))
    if is_no:
        return {"decision":"NO_TRADE", "why": "NO_TRADEゲート発動", "notes": gate_reasons}

    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        a = context_data.get('atr', 0.0)
        
        # ✅ 【修正】ペア名をプロンプトに反映
        prompt = f"""
あなたはFX投資ロボットの執行エンジンです。軍資金{context_data.get('capital')}円。
対象通貨ペア: 【 {pair_name} 】

現在の市場データ:
market_regime={regime}
現在価格={p}
ATR={a}

【重要指示】
1. {pair_name} の特性（クロス円なら日本の政治、ドルストなら米金利）を踏まえて判断せよ。
2. 直近の重要イベント（選挙や指標）のリスクを考慮せよ。
3. 出力はJSONオブジェクトのみ。
4. decision: "TRADE" or "NO_TRADE"
5. entry, stop_loss, take_profit を必ず含める。

JSON形式:
{{
  "decision": "TRADE",
  "side": "LONG",
  "entry": {p},
  "stop_loss": {p - a*2},
  "take_profit": {p + a*3},
  "horizon": "WEEK",
  "confidence": 0.85,
  "why": "日本語の理由",
  "notes": ["risk_factor_1"]
}}
"""
        resp = model.generate_content(prompt)
        j = _extract_json_block(resp.text)
        obj = json.loads(j) if j else {"decision":"NO_TRADE"}
        return obj
    except Exception as e:
        return {"decision":"NO_TRADE", "why": f"Error: {e}"}

# -----------------------------
# ✅ 【修正】週末判断 (数値ルール主導)
# -----------------------------
def get_ai_weekend_decision(api_key, context_data, symbol="USD/JPY"):
    """
    数値ルール（Month Hold Line）を主役に、AIは説明役に徹する設計。
    """
    model_name = get_active_model(api_key)
    if not model_name: return "API Key Error"

    price = float(context_data.get('price', 0.0))
    entry_price = float(context_data.get('entry_price', 0.0))
    trade_type = context_data.get('trade_type', "None")
    
    # 利益確保ライン (例: 2.0円 = 200pips)
    PROFIT_THRESHOLD = 2.0 
    
    month_hold_line = 0.0
    rule_judgment = "NO_POSITION"
    dist = 0.0
    
    # 数値判定
    if "Long" in trade_type or "Buy" in trade_type:
        month_hold_line = entry_price + PROFIT_THRESHOLD
        dist = price - month_hold_line
        if price >= month_hold_line:
            rule_judgment = "HOLD_MONTH (条件クリア)"
        else:
            rule_judgment = "TAKE_PROFIT (利益不足/リスク回避)"
            
    elif "Short" in trade_type or "Sell" in trade_type:
        month_hold_line = entry_price - PROFIT_THRESHOLD
        dist = month_hold_line - price
        if price <= month_hold_line:
            rule_judgment = "HOLD_MONTH (条件クリア)"
        else:
            rule_judgment = "TAKE_PROFIT (利益不足/リスク回避)"

    prompt = f"""
あなたは冷徹なFXファンドマネージャーです。
以下の「数値ルール」に基づき、週末の行動（来週も持ち越すか、手仕舞いするか）を決定します。
対象ペア: {symbol}

【数値ルール判定結果（絶対厳守）】
判定: {rule_judgment}
---------------------------
現在価格: {price:.3f}
取得単価: {entry_price:.3f} ({trade_type})
月越ホールド基準線(Month Hold Line): {month_hold_line:.3f}
現在値との距離: {dist:+.3f} (プラスならクリア)
---------------------------

【指示】
1. 上記の「判定」をそのまま採用すること。AIの独自判断で覆さないこと。
2. なぜその判定になったかを、トレンドやファンダメンタルズの観点から補足説明すること。
3. もし TAKE_PROFIT なら、月曜日の具体的な決済アクションを助言すること。

出力形式(Markdown):
## 判定: {rule_judgment}
**理由:** ...
**アドバイス:** ...
"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error: {e}"

def get_ai_portfolio(api_key, context_data):
    # 既存のまま
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        prompt = "ポートフォリオ分析をお願いします"
        resp = model.generate_content(prompt)
        return resp.text
    except: return "Error"

def get_ai_range(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        prompt = f"現在の価格は {p:.2f}です。今後1週間の[最高値, 最安値]を半角数字のみで返してください。"
        res = model.generate_content(prompt).text
        nums = re.findall(r"\d+\.\d+|\d+", res)
        return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None
    except: return None
