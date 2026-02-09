
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import pytz
import requests
import time
from datetime import datetime
import re  # ✅ 【必須】AI予想ラインの数値抽出用
import json  # ✅ JSON固定出力のため

TOKYO = pytz.timezone("Asia/Tokyo")

# 取得失敗時の理由をここに残す（main.pyで表示できる）
LAST_FETCH_ERROR = ""


# -----------------------------
# AI予想レンジ 自動取得キャッシュ（意思決定に必須で連携）
# -----------------------------
AI_RANGE_TTL_SEC = 60 * 60 * 72  # 72時間（週2回運用なら十分）
_AI_RANGE_CACHE = {"expire": 0.0, "value": None}

def ensure_ai_range(api_key: str, context_data: dict, force: bool = False):
    """意思決定（注文命令/週末判断）の直前に必ず呼ぶ。
    - ボタン不要: 取引判断の導線で自動取得する
    - TTL内はキャッシュを返す（429対策）
    - 失敗時は None を返す（後段で守りに倒す/ゲートで弾く）
    """
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
    """LLM出力から最初のJSONオブジェクト({..})を抽出。見つからなければ空文字。"""
    if not text:
        return ""
    s = text.strip()
    # そのままJSONの場合
    if s.startswith("{") and s.endswith("}"):
        return s
    # 文字列内の { ... } を最短〜最長で探索（簡易）
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return ""

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        # "155.20" / "155,20" など
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return default

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _validate_order_json(obj: dict, ctx: dict) -> (bool, list):
    """注文JSONの必須/整合性チェック。NGなら理由リストを返す。"""
    reasons = []
    if not isinstance(obj, dict):
        return False, ["order_json_not_object"]

    decision = obj.get("decision")
    if decision not in ("TRADE", "NO_TRADE"):
        reasons.append("decision_invalid")

    # NO_TRADEなら最低限でOK
    if decision == "NO_TRADE":
        return True, reasons

    side = obj.get("side")
    if side not in ("LONG", "SHORT"):
        reasons.append("side_invalid")

    entry = _safe_float(obj.get("entry"))
    tp = _safe_float(obj.get("take_profit"))
    sl = _safe_float(obj.get("stop_loss"))
    if entry is None: reasons.append("entry_missing")
    if tp is None: reasons.append("take_profit_missing")
    if sl is None: reasons.append("stop_loss_missing")

    horizon = obj.get("horizon")
    if horizon not in ("WEEK", "MONTH"):
        reasons.append("horizon_invalid")

    conf = _safe_float(obj.get("confidence"), default=0.0)
    if conf is None:
        reasons.append("confidence_missing")
    else:
        obj["confidence"] = _clamp(conf, 0.0, 1.0)

    if entry is not None and tp is not None and sl is not None and side in ("LONG","SHORT"):
        # 方向整合
        if side == "LONG":
            if not (sl < entry < tp):
                reasons.append("levels_inconsistent_long")
        else:
            if not (tp < entry < sl):
                reasons.append("levels_inconsistent_short")

        # RR最低ライン（例: 1.1）
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            reasons.append("risk_nonpositive")
        else:
            rr = reward / risk
            obj["rr_ratio"] = rr
            if rr < 1.1:
                reasons.append("rr_too_low")

        # 異常値ガード（現値から極端に遠い等）
        p = _safe_float(ctx.get("price"), default=entry)
        if p is not None:
            if abs(entry - p) / max(p, 1e-6) > 0.03:  # 3%超乖離は異常
                reasons.append("entry_too_far_from_price")

    # why/notes は任意（あると良い）
    if "why" not in obj:
        obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list):
        obj["notes"] = []

    return (len(reasons) == 0), reasons

def _validate_regime_json(obj: dict) -> (bool, list):
    reasons = []
    if not isinstance(obj, dict):
        return False, ["regime_json_not_object"]
    regime = obj.get("market_regime")
    if regime not in ("DEFENSIVE", "OPPORTUNITY"):
        reasons.append("market_regime_invalid")
    conf = _safe_float(obj.get("confidence"), default=0.0)
    obj["confidence"] = _clamp(conf, 0.0, 1.0)
    if "why" not in obj:
        obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list):
        obj["notes"] = []
    return (len(reasons) == 0), reasons

def _validate_weekend_json(obj: dict) -> (bool, list):
    reasons = []
    if not isinstance(obj, dict):
        return False, ["weekend_json_not_object"]
    action = obj.get("action")
    if action not in ("TAKE_PROFIT","CUT_LOSS","HOLD_WEEK","HOLD_MONTH","NO_POSITION"):
        reasons.append("action_invalid")
    if "why" not in obj:
        obj["why"] = ""
    if "notes" not in obj or not isinstance(obj.get("notes"), list):
        obj["notes"] = []
    if "levels" not in obj or not isinstance(obj.get("levels"), dict):
        obj["levels"] = {"take_profit": 0, "stop_loss": 0, "trail": 0}
    return (len(reasons) == 0), reasons

# -----------------------------
# NO_TRADEゲート（守り/攻め）
# -----------------------------
_NO_TRADE_THRESHOLDS = {
    # 守り型
    "DEFENSIVE": {
        "sma_diff_pct": 0.20,  # 0.20%
        "rsi_lo": 45.0,
        "rsi_hi": 55.0,
        "atr_mult": 1.6,
        "ma_close_pct": 0.10,  # MA25とMA75が0.10%以内
    },
    # 攻め型
    "OPPORTUNITY": {
        "sma_diff_pct": 0.15,  # 0.15%
        "rsi_lo": 48.0,
        "rsi_hi": 52.0,
        "atr_mult": 1.9,
        "ma_close_pct": 0.08,
    },
}

def no_trade_gate(context_data: dict, market_regime: str, force_defensive: bool = False):
    """数値条件でNO_TRADE判定。TrueならNO_TRADE理由リストを返す。"""
    reasons = []
    regime = "DEFENSIVE" if force_defensive else (market_regime if market_regime in _NO_TRADE_THRESHOLDS else "DEFENSIVE")
    th = _NO_TRADE_THRESHOLDS[regime]

    ps = str(context_data.get("panel_short",""))
    pm = str(context_data.get("panel_mid",""))
    mid_wait = ("静観" in pm)
    price = _safe_float(context_data.get("price"))
    sma25 = _safe_float(context_data.get("sma25"))
    sma75 = _safe_float(context_data.get("sma75"))
    rsi = _safe_float(context_data.get("rsi"))
    atr = _safe_float(context_data.get("atr"))
    atr_avg60 = _safe_float(context_data.get("atr_avg60"))

    # データ不備
    for k,v in [("price",price),("sma25",sma25),("sma75",sma75),("rsi",rsi),("atr",atr)]:
        if v is None or v != v:  # NaN
            reasons.append(f"data_invalid_{k}")

    if reasons:
        return True, regime, reasons

    # 方向感なし（MA収束＆RSI中立）
    sma_diff_pct = abs(sma25 - sma75) / max(price, 1e-6) * 100.0
    if sma_diff_pct < th["sma_diff_pct"] and (th["rsi_lo"] <= rsi <= th["rsi_hi"]):
        reasons.append("no_direction_ma_converge_and_rsi_neutral")

    # MA同士の接近（さらに厳しめ）
    if sma_diff_pct < th["ma_close_pct"]:
        reasons.append("ma25_ma75_too_close")

    # 荒れすぎ
    if atr_avg60 is not None and atr_avg60 > 0:
        if atr > atr_avg60 * th["atr_mult"]:
            reasons.append("volatility_too_high_atr_spike")

    return (len(reasons) > 0), regime, reasons


# -----------------------------
# トレンド週のみエントリー（週1放置運用のための最重要ルール）
# -----------------------------
_TREND_ONLY_RULES = {
    # トレンド強度（abs(SMA25-SMA75)/ATR）
    "trend_score_min": 1.0,
    # 過熱回避（RSIレンジ）
    "rsi_long_min": 45.0,
    "rsi_long_max": 70.0,
    "rsi_short_min": 30.0,
    "rsi_short_max": 55.0,
    # 荒すぎる週は避ける（ATR/ATR_avg60）
    "atr_spike_max": 1.7,
}

def trend_only_gate(context_data: dict):
    """週1運用向け：トレンド条件を満たさない週はエントリー禁止。
    Returns:
      allowed(bool), side_hint(str), trend_score(float|None), reasons(list[str])
    """
    reasons = []
    price = _safe_float(context_data.get("price"))
    sma25 = _safe_float(context_data.get("sma25"))
    sma75 = _safe_float(context_data.get("sma75"))
    atr = _safe_float(context_data.get("atr"))
    atr_avg60 = _safe_float(context_data.get("atr_avg60"))
    rsi = _safe_float(context_data.get("rsi"))

    # 必要データ
    for k, v in [("price", price), ("sma25", sma25), ("sma75", sma75), ("atr", atr), ("rsi", rsi)]:
        if v is None or v != v:
            reasons.append(f"trend_gate_data_invalid_{k}")

    if reasons:
        return False, "NONE", None, reasons

    # 方向一致（CloseとMAの並び）
    side_hint = "NONE"
    if (price > sma25) and (sma25 > sma75):
        side_hint = "LONG"
    elif (price < sma25) and (sma25 < sma75):
        side_hint = "SHORT"
    else:
        reasons.append("trend_gate_direction_not_aligned")

    # トレンド強度（MA差がATRに対して十分か）
    trend_score = None
    try:
        trend_score = abs(sma25 - sma75) / max(float(atr), 1e-9)
        if trend_score < float(_TREND_ONLY_RULES["trend_score_min"]):
            reasons.append(f"trend_gate_trend_score_low:{trend_score:.2f}")
    except Exception:
        reasons.append("trend_gate_trend_score_calc_failed")

    # RSI過熱回避
    if side_hint == "LONG":
        if not (float(_TREND_ONLY_RULES["rsi_long_min"]) <= rsi <= float(_TREND_ONLY_RULES["rsi_long_max"])):
            reasons.append("trend_gate_rsi_out_of_range_long")
    elif side_hint == "SHORT":
        if not (float(_TREND_ONLY_RULES["rsi_short_min"]) <= rsi <= float(_TREND_ONLY_RULES["rsi_short_max"])):
            reasons.append("trend_gate_rsi_out_of_range_short")

    # 荒すぎる週は避ける（週1放置で事故りやすい）
    if atr_avg60 is not None and atr_avg60 > 0:
        try:
            ratio = float(atr) / float(atr_avg60)
            if ratio > float(_TREND_ONLY_RULES["atr_spike_max"]):
                reasons.append(f"trend_gate_atr_spike:{ratio:.2f}")
        except Exception:
            pass

    allowed = (len(reasons) == 0)
    return allowed, side_hint, trend_score, reasons

def _recommended_stop_entry(context_data: dict, side_hint: str):
    """トレンド週の逆指値（ブレイク）用推奨エントリー価格を返す。無ければNone。"""
    price = _safe_float(context_data.get("price"))
    atr = _safe_float(context_data.get("atr"))
    rh = _safe_float(context_data.get("recent_high20"))
    rl = _safe_float(context_data.get("recent_low20"))
    buf = _safe_float(context_data.get("breakout_buffer"))

    # バッファのフォールバック（0.1円 or ATRの1/4）
    if buf is None:
        try:
            buf = max(0.10, (float(atr) * 0.25 if atr is not None else 0.10))
        except Exception:
            buf = 0.10

    if side_hint == "LONG":
        base = rh if rh is not None else price
        if base is None:
            return None, buf
        return float(base) + float(buf), float(buf)
    if side_hint == "SHORT":
        base = rl if rl is not None else price
        if base is None:
            return None, buf
        return float(base) - float(buf), float(buf)
    return None, float(buf)

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

    # ✅ 5日線との距離・関係性を重視したロジック
    if price > sma5:
        short_s = "上昇継続（短期）"
        short_c = "#e3f2fd"
        short_a = f"現在値は<b>5日線 ({sma5:.2f})</b> の上を推移中。<br>短期的な上昇圧力は維持されています。"
    else:
        short_s = "勢い鈍化・調整"
        short_c = "#fce4ec"
        short_a = f"現在値は<b>5日線 ({sma5:.2f})</b> を下回りました。<br>短期的な調整（下落）局面に注意。"

    return {
        "short": {"status": short_s, "color": short_c, "advice": short_a},
        "mid": {"status": mid_s, "color": mid_c, "advice": mid_a},
        "price": price,
    }


# =====================================================
# AI分析（FP1級・衆院選【後】・実戦運用版）
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


def get_ai_market_regime(api_key, context_data):
    """
    market_regime を AI に出させる（DEFENSIVE / OPPORTUNITY）。
    ここでの出力は「NO_TRADEゲートの厳しさ」を切替える目的（裁量介入を減らす）。
    JSONのみ返す。
    """
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get("price", 0.0)
        rsi = context_data.get("rsi", 0.0)
        sma25 = context_data.get("sma25", 0.0)
        sma75 = context_data.get("sma75", 0.0)
        atr = context_data.get("atr", 0.0)
        atr_avg60 = context_data.get("atr_avg60", 0.0)
        ps = context_data.get("panel_short", "不明")
        pm = context_data.get("panel_mid", "不明")
        report = context_data.get("last_report", "なし")

        prompt = f"""
あなたはFX運用の市場環境判定エンジンです。
目的：今週の市場環境を「守り(DEFENSIVE)」か「機会(OPPORTUNITY)」のどちらかに分類し、
NO_TRADEゲートの厳しさを切り替えるための判定を出してください。

【入力（USD/JPY）】
price={p}
rsi={rsi}
sma25={sma25}
sma75={sma75}
atr={atr}
atr_avg60={atr_avg60}
panel_short={ps}
panel_mid={pm}
last_report_summary={report[:700]}

【出力ルール】
- 出力はJSONオブジェクトのみ（前後に文章を付けない）
- 次のキーを必ず含める：
  market_regime: "DEFENSIVE" または "OPPORTUNITY"
  confidence: 0.0〜1.0
  why: 1〜3文の理由（日本語）
  notes: 箇条書き配列（0〜6個）

【判定の目安】
- DEFENSIVE: 方向感が弱い/レンジ/ボラが荒い/中期が静観など、期待値が低い
- OPPORTUNITY: 週足で方向感が比較的明確で、継続/伸びが期待できる
"""
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        j = _extract_json_block(raw)
        obj = json.loads(j) if j else {}
        ok, reasons = _validate_regime_json(obj)
        if not ok:
            return {
                "market_regime": "DEFENSIVE",
                "confidence": 0.0,
                "why": "market_regime JSONが不正のため保守的にDEFENSIVEへ。",
                "notes": ["parse_or_validation_failed", *reasons]
            }
        return obj
    except Exception as e:
        return {
            "market_regime": "DEFENSIVE",
            "confidence": 0.0,
            "why": f"market_regime 判定で例外。保守的にDEFENSIVEへ。({type(e).__name__})",
            "notes": []
        }

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
        capital = context_data.get("capital", 300000)
        
        is_gotobi = context_data.get("is_gotobi", False)
        gotobi_text = "今日は五十日(ゴトウビ)です。実需のドル買いフローに注意してください。" if is_gotobi else "今日は五十日ではありません。"

        ep = context_data.get("entry_price", 0.0)
        tt = context_data.get("trade_type", "なし")

        # ✅ プロンプト修正：選挙「後」の本格運用フェーズに対応
        base_prompt = f"""
あなたはFP1級を保持する、極めて優秀な為替戦略家です。
特に現在は「衆議院選挙の結果」を受けた直後の、極めて重要な局面であることを強く認識してください。

【市場データ】
- ドル円価格: {p:.3f}円
- 日米金利差(10年債): {u:.2f}%
- ボラティリティ(ATR): {a:.3f}
- SMA25乖離率: {s:.2f}%
- RSI(14日): {r:.1f}
- 五十日判定: {gotobi_text}

【分析依頼：以下の4項目に沿ってFPに分かりやすく回答してください】
1. 【ファンダメンタルズ】日米金利差の現状と、選挙結果（市場の織り込み状況）を踏まえて解説
2. 【地政学・外部要因】選挙後の政治的安定性や、インフレ・景気後退への影響を分析
   特に「選挙結果」が市場にサプライズを与えたか、安定をもたらしたかを判断してください。
3. 【テクニカル】乖離率とRSI({r:.1f})、および「窓開け」の状況から見て、今は「割安」か「割高」か。
4. 【具体的戦略】NISAや外貨建資産のバランスを考える際のアアドバイスのように、
   出口戦略（利確）を含めた今後1週間の戦略を提示

【レポート構成：必ず以下の4項目に沿って記述してください】
1. 現在の相場環境の要約（選挙結果の影響含む）
2. 上記データ（特に金利差とボラティリティ）から読み解くリスク
3. 具体的な戦略（エントリー・利確・損切の目安価格を具体的に提示）
4. 経済カレンダーを踏まえた、今週の警戒イベントへの助言

回答は親しみやすくも、プロの厳格さを感じる日本語でお願いします。
        """

        add_prompt = f"""
        【追加コンテキスト：ユーザーの実戦運用情報】
        ユーザーは現在、軍資金{capital}円（SBI FX/レバレッジ25倍）で運用中です。
        保有ポジション: {f"{ep}円で{tt}" if ep > 0 else "現在なし"}

        もしポジションがある場合、上記3の「具体的戦略」内で、
        この保有建玉に対する「選挙後の処理（ホールド/決済）」を具体的に助言してください。
        
        また、以下の2点を必ず追記してください。
        - **推奨スリップロス**: 現在のボラティリティ(ATR)に基づいた、注文を通すための許容値（pips）。
        - **資金管理**: 30万円を守りながら増やすためのリスク管理アドバイス。
        """

        full_prompt = base_prompt + "\n" + add_prompt

        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"AI分析エラー: {str(e)}"


def get_ai_weekend_decision(api_key, context_data, override_mode="AUTO", override_reason=""):
    """
    週末判断（利確/損切/継続/1か月継続）をJSON命令で返す。
    """
    # 緊急停止：週末も「取引しない/新規しない」に固定
    if override_mode == "FORCE_NO_TRADE":
        why = "緊急停止（FORCE_NO_TRADE）。"
        if override_reason and override_reason.strip():
            why += f" 理由: {override_reason.strip()}"
        return {
            "action": "NO_POSITION",
            "why": why,
            "levels": {"take_profit": 0, "stop_loss": 0, "trail": 0},
            "notes": ["human_override=true"],
            "override": {"mode": override_mode, "reason": override_reason.strip()}
        }

    try:
        model = genai.GenerativeModel(get_active_model(api_key))

        p = context_data.get("price", 0.0)
        ps = context_data.get("panel_short", "不明")
        pm = context_data.get("panel_mid", "不明")
        report = context_data.get("last_report", "なし")

        ep = context_data.get("entry_price", 0.0)
        tt = context_data.get("trade_type", "なし")
        pos = f"現在ポジション: entry_price={ep}, trade_type={tt}" if ep and ep > 0 else "現在ポジション: なし"

        prompt = f"""
あなたはFX運用の週末判断エンジンです。出力はJSON命令のみ。

【前提（運用ルール）】
- 月曜にエントリー、週末に「利確/損切/継続/1か月継続」を判断する。
- 人間は判断せず、数値入力のみ行う。
- あいまい表現は禁止。必ず action を選ぶ。

【入力】
price={p}
panel_short={ps}
panel_mid={pm}
{pos}
last_report_summary={report[:900]}

【出力(JSONのみ)】
- action: "TAKE_PROFIT" | "CUT_LOSS" | "HOLD_WEEK" | "HOLD_MONTH" | "NO_POSITION"
- why: 1〜3文の理由（日本語）
- levels: {{ take_profit: number, stop_loss: number, trail: number }}  (該当が無ければ0)
- notes: string配列（0〜6）

【判定のガイド】
- 週末時点で構造が壊れている/損切基準に抵触 -> CUT_LOSS
- 週内目標達成/上限到達 -> TAKE_PROFIT（必要ならtrailも）
- 構造維持で週跨ぎ -> HOLD_WEEK
- 月足方向が明確で伸びしろあり -> HOLD_MONTH
"""
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        j = _extract_json_block(raw)
        obj = json.loads(j) if j else {}
        ok, reasons = _validate_weekend_json(obj)
        if not ok:
            return {
                "action": "NO_POSITION",
                "why": "週末判断JSONが不正のため保守的にNO_POSITIONへ。",
                "levels": {"take_profit": 0, "stop_loss": 0, "trail": 0},
                "notes": ["parse_or_validation_failed", *reasons],
                "override": {"mode": override_mode, "reason": override_reason.strip()} if override_mode != "AUTO" else {"mode":"AUTO","reason":""}
            }
        if override_mode != "AUTO":
            obj["override"] = {"mode": override_mode, "reason": override_reason.strip()}
            obj.setdefault("notes", []).append("human_override=true")
        return obj
    except Exception as e:
        return {
            "action": "NO_POSITION",
            "why": f"週末判断で例外。保守的にNO_POSITIONへ。({type(e).__name__})",
            "levels": {"take_profit": 0, "stop_loss": 0, "trail": 0},
            "notes": [],
            "override": {"mode": override_mode, "reason": override_reason.strip()} if override_mode != "AUTO" else {"mode":"AUTO","reason":""}
        }

def get_ai_portfolio(api_key, context_data):
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        ep = context_data.get("entry_price", 0.0)
        tt = context_data.get("trade_type", "なし")
        
        pos_str = f"現在 {ep}円で{tt}保有中。" if ep > 0 else "現在ノーポジション。"

        prompt = f"""
        あなたはFP1級技能士です。ユーザー状況: {pos_str}
        
        1. ユーザーの既存ポジションに対する「週末/月末の持ち越し診断」を行ってください。
        2. 日本円, 米ドル, ユーロ, 豪ドル, 英ポンド, メキシコペソの
           最適配分（合計100%）を提示してください。
        
        特に「メキシコペソ/円」や「豪ドル/円」などの高金利通貨をポートフォリオに組み込むメリット・デメリットを
        現在の市場環境（衆院選・米金利）を踏まえて解説してください。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "ポートフォリオ分析に失敗しました。"


def get_ai_order_strategy(api_key, context_data, override_mode="AUTO", override_reason=""):
    """
    注文命令書をJSON固定で返す（命令 + why/notes解説）。
    さらに market_regime をAIが判定し、守り/攻めのNO_TRADEゲートを自動切替する。
    """
    # --- 緊急停止 ---
    if override_mode == "FORCE_NO_TRADE":
        why = "緊急停止（FORCE_NO_TRADE）。"
        if override_reason and override_reason.strip():
            why += f" 理由: {override_reason.strip()}"
        return {
            "decision": "NO_TRADE",
            "side": "NONE",
            "entry": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "horizon": "WEEK",
            "confidence": 0.0,
            "why": why,
            "notes": ["human_override=true"],
            "market_regime": "DEFENSIVE",
            "regime_why": "FORCE_NO_TRADEのため市場判定は省略。",
            "override": {"mode": override_mode, "reason": override_reason.strip()}
        }

    # --- market_regime（AUTO/縮退） ---
    if override_mode == "FORCE_DEFENSIVE":
        regime_obj = {
            "market_regime": "DEFENSIVE",
            "confidence": 0.0,
            "why": "緊急縮退（FORCE_DEFENSIVE）のため、守り型として判定。",
            "notes": ["human_override=true"]
        }
        force_def = True
    else:
        regime_obj = get_ai_market_regime(api_key, context_data)
        force_def = False

    market_regime = regime_obj.get("market_regime", "DEFENSIVE")
    regime_why = regime_obj.get("why", "")

    # --- NO_TRADEゲート（先に数値ルールで弾く） ---
    is_no, regime_used, gate_reasons = no_trade_gate(context_data, market_regime, force_defensive=force_def)
    if is_no:
        why = "NO_TRADEゲートにより取引停止。"
        if gate_reasons:
            why += " / " + ", ".join(gate_reasons[:6])
        out = {
            "decision": "NO_TRADE",
            "side": "NONE",
            "entry": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "horizon": "WEEK",
            "confidence": 0.0,
            "why": why,
            "notes": gate_reasons[:12],
            "market_regime": regime_used,
            "regime_why": regime_why,
        }
        if override_mode != "AUTO":
            out["override"] = {"mode": override_mode, "reason": override_reason.strip()}
            out["notes"].append("human_override=true")
        return out


    # --- トレンド週のみ（週1放置運用の中核ルール） ---
    allowed_trend, side_hint, trend_score, trend_reasons = trend_only_gate(context_data)
    context_data["trend_side_hint"] = side_hint
    if trend_score is not None:
        context_data["trend_score"] = float(trend_score)

    # トレンド条件を満たさない週は無条件で見送り（レンジ週は原則やらない）
    if not allowed_trend:
        why = "トレンド条件を満たさないため今週は見送り（トレンド週のみ運用ルール）。"
        if trend_reasons:
            why += " / " + ", ".join(trend_reasons[:6])
        out = {
            "decision": "NO_TRADE",
            "side": "NONE",
            "entry": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "horizon": "WEEK",
            "confidence": 0.0,
            "why": why,
            "notes": trend_reasons[:12],
            "market_regime": market_regime,
            "regime_why": regime_why,
            "rule": {"trend_only": True, "trend_side_hint": side_hint, "trend_score": trend_score},
        }
        if override_mode != "AUTO":
            out["override"] = {"mode": override_mode, "reason": override_reason.strip()}
            out["notes"].append("human_override=true")
        return out

    # 推奨：ブレイク逆指値のエントリー価格（IFD-OCO前提）
    rec_entry, rec_buf = _recommended_stop_entry(context_data, side_hint)
    if rec_entry is not None:
        context_data["recommended_entry"] = float(rec_entry)
        context_data["breakout_buffer"] = float(rec_buf)

    # --- AI注文生成（JSON固定） ---
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = context_data.get('price', 0.0)
        a = context_data.get('atr', 0.0)
        report = context_data.get('last_report', "なし")
        ps = context_data.get("panel_short", "不明")
        pm = context_data.get("panel_mid", "不明")
        capital = context_data.get("capital", 300000)

        ep = context_data.get("entry_price", 0.0)
        tt = context_data.get("trade_type", "なし")
        pos_instr = f"ユーザーは既に {ep}円で{tt} を保有中。新規/増し玉/決済も含め最適な1つの行動に統合せよ。" if ep and ep > 0 else "ユーザーの保有ポジションはなし（新規判断）。"

        prompt = f"""
あなたはFX投資ロボットの執行エンジンです。軍資金{capital}円、レバレッジ25倍。
上部パネル診断とレポートを尊重し、今週の具体的な「注文命令」を1つだけ出してください。

【市場環境モード（自動判定）】
market_regime={market_regime}
regime_why={regime_why}

【入力】
price={p}
atr={a}
panel_short={ps}
panel_mid={pm}
{pos_instr}
trend_side_hint={context_data.get('trend_side_hint','NONE')}
trend_score={context_data.get('trend_score','')}
recent_high20={context_data.get('recent_high20','')}
recent_low20={context_data.get('recent_low20','')}
recommended_entry={context_data.get('recommended_entry','')}
breakout_buffer={context_data.get('breakout_buffer','')}
last_report_summary={report[:1200]}

【出力ルール（最重要）】
- 出力はJSONオブジェクトのみ（前後に文章を付けない）
- 必須キー：
  decision: "TRADE" または "NO_TRADE"
  side: "LONG" | "SHORT" | "NONE"
  entry: number
  take_profit: number
  stop_loss: number
  horizon: "WEEK" | "MONTH"
  confidence: 0.0〜1.0
  why: 1〜3文（日本語）
  notes: 0〜6個の配列（日本語）
- 追加必須キー（注文方式の自動化）：
  order_bundle: "IFD_OCO" | "OCO" | "NONE"
  entry_type: "STOP" | "LIMIT" | "MARKET" | "NONE"
  entry_price_kind_jp: "逆指値" | "指値" | "成行" | "なし"
  bundle_hint_jp: 1行（例: "SBI: 逆指値(IFD-OCO)で放置運用"）
- トレンド週のみ運用：side は trend_side_hint に必ず合わせる（LONG/SHORT）。
- decision="TRADE" のとき entry は recommended_entry を優先して使い、entry_type="STOP" とする。
- decision="TRADE" の場合、必ず stop_loss を含める（欠落禁止）
- 数値は小数OK、USD/JPYなので 2〜3桁小数で良い
- あいまい表現で行動を濁さない。TRADE/NO_TRADEどちらかに決める。
"""
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        j = _extract_json_block(raw)
        obj = json.loads(j) if j else {"decision":"NO_TRADE","side":"NONE","why":"JSON抽出失敗","notes":["json_extract_failed"]}
        ok, reasons = _validate_order_json(obj, context_data)

        # バリデーションNG → 強制NO_TRADE
        if not ok:
            out = {
                "decision": "NO_TRADE",
                "side": "NONE",
                "entry": 0,
                "take_profit": 0,
                "stop_loss": 0,
                "horizon": "WEEK",
                "confidence": 0.0,
                "why": "注文JSONの検証に失敗したため取引停止。",
                "notes": ["parse_or_validation_failed", *reasons][:12],
                "market_regime": market_regime,
                "regime_why": regime_why,
            }
            if override_mode != "AUTO":
                out["override"] = {"mode": override_mode, "reason": override_reason.strip()}
                out["notes"].append("human_override=true")
            return out

        # 正常時：regimeを付与
        obj["market_regime"] = market_regime
        obj["regime_why"] = regime_why

        # --- 注文方式の確定（週1放置運用：トレンド週のみ + 逆指値IFD-OCO固定） ---
        try:
            decision = obj.get("decision")
            side = obj.get("side")
            # トレンド週のみ運用：sideは必ずヒントに一致させる（不一致は事故防止で停止）
            if decision == "TRADE":
                if side_hint in ("LONG","SHORT") and side not in (side_hint,):
                    obj = {
                        "decision": "NO_TRADE",
                        "side": "NONE",
                        "entry": 0,
                        "take_profit": 0,
                        "stop_loss": 0,
                        "horizon": "WEEK",
                        "confidence": 0.0,
                        "why": f"AIのside({side})がトレンド判定({side_hint})と不一致のため取引停止（事故防止）。",
                        "notes": ["side_mismatch_to_trend_gate"],
                        "market_regime": market_regime,
                        "regime_why": regime_why,
                        "rule": {"trend_only": True, "trend_side_hint": side_hint}
                    }
                else:
                    # 推奨エントリー（ブレイク逆指値）に揃える
                    rec_entry = _safe_float(context_data.get("recommended_entry"))
                    if rec_entry is not None:
                        old_entry = _safe_float(obj.get("entry"))
                        tp = _safe_float(obj.get("take_profit"))
                        sl = _safe_float(obj.get("stop_loss"))
                        if old_entry is not None and tp is not None and sl is not None:
                            delta = float(rec_entry) - float(old_entry)
                            obj["entry"] = float(rec_entry)
                            obj["take_profit"] = float(tp) + delta
                            obj["stop_loss"] = float(sl) + delta
                            obj.setdefault("notes", []).append("entry_aligned_to_recommended_stop")

                    # 注文方式キーを必ず付与
                    obj["order_bundle"] = "IFD_OCO"
                    obj["entry_type"] = "STOP"
                    obj["entry_price_kind_jp"] = "逆指値"
                    obj["bundle_hint_jp"] = "SBI: 逆指値(IFD-OCO)で放置運用（TP/SL同時）"
                    obj.setdefault("notes", []).append("order_method_fixed_ifd_oco_stop")
        except Exception:
            pass
        if override_mode != "AUTO":
            obj["override"] = {"mode": override_mode, "reason": override_reason.strip()}
            obj.setdefault("notes", []).append("human_override=true")
        return obj
    except Exception as e:
        out = {
            "decision": "NO_TRADE",
            "side": "NONE",
            "entry": 0,
            "take_profit": 0,
            "stop_loss": 0,
            "horizon": "WEEK",
            "confidence": 0.0,
            "why": f"注文生成で例外。保守的に取引停止。({type(e).__name__})",
            "notes": [],
            "market_regime": market_regime,
            "regime_why": regime_why,
        }
        if override_mode != "AUTO":
            out["override"] = {"mode": override_mode, "reason": override_reason.strip()}
            out["notes"].append("human_override=true")
        return out

def get_ai_range(api_key, context_data):
    """
    AI予想レンジ（1週間の高値/安値）を取得して返す。
    返り値は [high, low] のリスト（main.pyの既存実装と互換）。

    ※不具合対策:
    - LLMが「1週間」などの文字を含めて返すと、正規表現が「1」を拾ってしまうことがある。
      → 数値を正規化（全角→半角）し、妥当レンジでフィルタし、最大/最小で決定する。
    """
    try:
        model_name = get_active_model(api_key)
        if not model_name:
            return None
        model = genai.GenerativeModel(model_name)

        # 現在値（妥当レンジ推定に使用）
        try:
            p = float(context_data.get("price", 0.0) or 0.0)
        except Exception:
            p = 0.0

        # できるだけ「数字2つだけ」を返させる（文章/単位/番号を禁止）
        prompt = (
            f"現在のドル円は {p:.3f}円です。"
            "今後1週間の『予想最高値,予想最安値』を、半角数字の小数で2つ、カンマ区切りで返してください。"
            "文章・単位（円）・括弧・改行・番号・追加説明は禁止。"
            "例: 161.250,159.800"
        )

        resp = model.generate_content(prompt)
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw:
            return None

        # 全角数字/句読点の正規化（Geminiが全角で返す場合に備える）
        trans = str.maketrans({
            "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
            "．":".","，":",","－":"-","ー":"-",
        })
        raw_norm = raw.translate(trans)

        # 数字抽出（小数優先）
        nums = re.findall(r"-?\d+\.\d+|-?\d+", raw_norm)
        vals = []
        for s in nums:
            try:
                vals.append(float(s))
            except Exception:
                continue

        if not vals:
            return None

        # 妥当レンジフィルタ（USD/JPYの想定レンジ + 現在値近傍を優先）
        # まず「極端な値（例: 1, 2026）」を落とす
        plausible = [v for v in vals if 50.0 <= v <= 300.0]

        # 現在値が取れているなら、さらに近傍（±30円）を優先
        if p >= 50.0:
            near = [v for v in plausible if (p - 30.0) <= v <= (p + 30.0)]
        else:
            near = plausible

        candidates = near if len(near) >= 2 else plausible

        if len(candidates) >= 2:
            high = float(max(candidates))
            low = float(min(candidates))
            # あり得ない並び（高値=安値）も許容だが、low>highはここでは起きない
            return [high, low]

        # ここまで来るのは「妥当な数値が1つしか取れない」ケース。
        # もう一度だけ強制フォーマットで再試行（429対策のため1回だけ）
        prompt2 = (
            f"USD/JPYの現在値は {p:.3f}。"
            "次の形式で『最高値,最安値』を必ず2つ返してください: 161.250,159.800"
        )
        resp2 = model.generate_content(prompt2)
        raw2 = (getattr(resp2, "text", None) or "").strip()
        raw2 = raw2.translate(trans)
        nums2 = re.findall(r"-?\d+\.\d+|-?\d+", raw2)
        vals2 = []
        for s in nums2:
            try:
                vals2.append(float(s))
            except Exception:
                continue
        plausible2 = [v for v in vals2 if 50.0 <= v <= 300.0]
        if len(plausible2) >= 2:
            high = float(max(plausible2))
            low = float(min(plausible2))
            return [high, low]

        return None
    except Exception:
        return None


# === PORTFOLIO_AUTOMATION_EXTENSIONS v1 ===

from typing import Any, Tuple

def _pair_label_to_currencies(pair_label: str) -> Tuple[str, str]:
    """Extract base/quote currencies from label like 'USD/JPY (ドル円)' or 'EUR/USD (ユーロドル)'."""
    # label begins with 'AAA/BBB'
    head = pair_label.split()[0]
    if "/" in head and len(head) >= 7:
        base, quote = head.split("/")[:2]
        return base.strip(), quote.strip()
    # fallback: attempt from ticker map key format
    if "/" in pair_label:
        base, quote = pair_label.split("/")[:2]
        return base.strip()[:3], quote.strip()[:3]
    return "UNK", "UNK"

def portfolio_weekly_risk_percent(active_positions: list) -> float:
    """Sum of risk_percent across active positions."""
    total = 0.0
    for p in active_positions or []:
        try:
            total += float(p.get("risk_percent", p.get("risk", 0.0)))
        except Exception:
            continue
    return float(total)

def portfolio_currency_counts(active_positions: list) -> dict:
    counts = {}
    for p in active_positions or []:
        pair = p.get("pair") or p.get("pair_label") or p.get("pair_name") or ""
        if not pair:
            continue
        b, q = _pair_label_to_currencies(pair)
        counts[b] = counts.get(b, 0) + 1
        counts[q] = counts.get(q, 0) + 1
    return counts

def violates_currency_concentration(candidate_pair_label: str, active_positions: list, max_positions_per_currency: int = 1) -> bool:
    """
    Simple correlation proxy:
    - If any currency (USD/JPY/EUR/AUD/GBP...) would be held in more than max_positions_per_currency positions, block.
    Default max_positions_per_currency=1 prevents stacking multiple JPY-crosses etc.
    """
    counts = portfolio_currency_counts(active_positions)
    b, q = _pair_label_to_currencies(candidate_pair_label)
    # +1 exposure if opened
    return (counts.get(b, 0) + 1 > max_positions_per_currency) or (counts.get(q, 0) + 1 > max_positions_per_currency)

def can_open_under_weekly_cap(active_positions: list, new_risk_percent: float, weekly_dd_cap_percent: float) -> bool:
    try:
        new_risk = float(new_risk_percent)
        cap = float(weekly_dd_cap_percent)
    except Exception:
        return True
    return (portfolio_weekly_risk_percent(active_positions) + new_risk) <= cap

def _build_market_summary_for_pairs() -> str:
    market_summary = ""
    for name, sym in PAIR_MAP.items():
        try:
            df = _yahoo_chart(sym, rng="5d", interval="1d")
            if df is None or df.empty:
                continue
            close = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
            chg = (close - prev) / prev * 100 if prev else 0.0
            trend = "上昇" if close > float(df["Close"].mean()) else "下降"
            market_summary += f"- {name}: Price={close:.3f} / Chg={chg:+.2f}% / Trend={trend}\n"
        except Exception:
            continue
    return market_summary

def suggest_alternative_pair_if_usdjpy_stay(
    api_key: str,
    active_positions: list,
    risk_percent_per_trade: float,
    weekly_dd_cap_percent: float = 2.0,
    max_positions_per_currency: int = 1,
    exclude_pair_label: str = "USD/JPY (ドル円)",
    display_top_k: int = 3
) -> dict:
    """
    If USD/JPY is NO_TRADE (STAY), suggest an alternative pair.

    Returns:
      {
        "best_pair_name": str,
        "reason": str,
        "confidence": float,
        "blocked": bool,
        "blocked_by": str,
        "source": "ai" | "numeric_scan" | "fallback",
        "candidates": [
            {"pair": str, "confidence": float, "status": "SELECTED"|"REJECTED", "rejected_by": [str], "reason": str}
        ]
      }

    Notes:
    - We keep the original AI prompt and filtering logic.
    - We additionally expose up to `display_top_k` evaluated candidates with rejection reasons
      so main.py can explain "なぜそれしか出ないのか" without changing trading rules.
    """
    model_name = get_active_model(api_key)
    if not model_name:
        return {}

    # Weekly cap gate first
    if not can_open_under_weekly_cap(active_positions, risk_percent_per_trade, weekly_dd_cap_percent):
        return {
            "best_pair_name": "",
            "reason": "週単位DDキャップを超えるため今週は新規不可",
            "confidence": 1.0,
            "blocked": True,
            "blocked_by": "weekly_dd_cap",
            "source": "filters",
            "candidates": []
        }

    market_summary = _build_market_summary_for_pairs()

    prompt = f"""
あなたはプロのFXファンドマネージャーです。
以下の市場データから「今週、USD/JPYが見送りのときに代替として最も利益チャンスがありそうなペア」を最大5つ、優先順で提案してください。
ただし同じ通貨への偏り（例: JPY絡みを複数）を避ける観点も考慮してください。

【市場概況】
{market_summary}

【必須制約】
- 可能なら {exclude_pair_label} 以外から選ぶ
- 出力はJSONのみ

【出力JSON】
{{
  "candidates": [
    {{"pair": "EUR/USD (ユーロドル)", "reason": "...", "confidence": 0.0}},
    {{"pair": "AUD/JPY (豪ドル円)", "reason": "...", "confidence": 0.0}}
  ]
}}
"""

    candidates = []
    source = "ai"
    ai_error = ""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = (resp.text or "").strip()
        data = safe_json_loads(txt) if 'safe_json_loads' in globals() else json.loads(_extract_json(txt))
        candidates = data.get("candidates", []) or []
    except Exception as e:
        ai_error = str(e)
        # fallback: reuse existing scan_best_pair single answer
        fn = globals().get('scan_best_pair')
        single = fn(api_key) if callable(fn) else {}
        candidates = []
        source = "fallback"
        if isinstance(single, dict) and single.get("best_pair_name"):
            candidates.append({
                "pair": single.get("best_pair_name"),
                "reason": single.get("reason", ""),
                "confidence": single.get("confidence", 0.5)
            })

    # Normalize list shape
    norm = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        pair = (c.get("pair") or "").strip()
        if not pair:
            continue
        norm.append({
            "pair": pair,
            "reason": c.get("reason", ""),
            "confidence": _fx_safe_float(c.get("confidence", 0.5), 0.5)
        })
    candidates = norm

    # Evaluate (explainable) filtering
    evals = []
    selected = None

    # If portfolio is empty, do NOT block on currency concentration (first position is allowed)
    active_nonempty = bool(active_positions) and len(active_positions) > 0

    for c in candidates:
        pair = c["pair"]
        rejected_by = []
        if pair == exclude_pair_label:
            rejected_by.append("excluded_pair")
        if active_nonempty and violates_currency_concentration(pair, active_positions, max_positions_per_currency=max_positions_per_currency):
            rejected_by.append("currency_concentration")
        status = "REJECTED" if rejected_by else "CANDIDATE"
        if (selected is None) and (status == "CANDIDATE"):
            status = "SELECTED"
            selected = c
        evals.append({
            "pair": pair,
            "confidence": c.get("confidence", 0.5),
            "status": status,
            "rejected_by": rejected_by,
            "reason": c.get("reason", "")
        })
        if len(evals) >= max(1, int(display_top_k or 3)):
            # we still want to show why top candidates are rejected/selected
            continue

    # If none selected from AI list, try numeric scan as a last resort (kept conservative)
    if selected is None:
        try:
            fn = globals().get("scan_best_pair_numeric")
            # scan_best_pair_numeric may exist in some builds; if not, fall back to scan_best_pair
            if callable(fn):
                numeric = fn(api_key, topn=5)  # expected list of dicts
            else:
                numeric = []
        except Exception:
            numeric = []

        # Numeric scan: pick first that passes currency concentration (if any)
        if isinstance(numeric, list):
            source = "numeric_scan"
            for item in numeric:
                try:
                    pair = (item.get("pair") or item.get("best_pair_name") or "").strip()
                    if not pair:
                        continue
                    if pair == exclude_pair_label:
                        continue
                    rejected_by = []
                    if active_nonempty and violates_currency_concentration(pair, active_positions, max_positions_per_currency=max_positions_per_currency):
                        rejected_by.append("currency_concentration")
                    if not rejected_by:
                        selected = {"pair": pair, "reason": item.get("reason", ""), "confidence": _fx_safe_float(item.get("confidence", 0.65), 0.65)}
                        break
                except Exception:
                    continue

    # Compose return
    if selected is not None:
        return {
            "best_pair_name": selected["pair"],
            "reason": selected.get("reason", ""),
            "confidence": selected.get("confidence", 0.5),
            "blocked": False,
            "blocked_by": "",
            "source": source,
            "candidates": evals[: max(1, int(display_top_k or 3))],
            "debug": {"ai_candidate_error": ai_error} if ai_error else {}
        }

    return {
        "best_pair_name": "",
        "reason": "条件（DDキャップ/通貨分散）を満たす代替ペアが見つかりませんでした",
        "confidence": 0.0,
        "blocked": True,
        "blocked_by": "filters",
        "source": source,
        "candidates": evals[: max(1, int(display_top_k or 3))],
        "debug": {"ai_candidate_error": ai_error} if ai_error else {}
    }


# =============================
# 100% TOOL ADDITIONS
# - explicit weekly structure_ok rules
# - numeric HOLD_MONTH enforcement (AI HOLD_MONTH is downgraded if numeric conditions not met)
# - fix function signatures used by main_auto (pair_name/portfolio args)
# NOTE: Existing algorithms/prompts are NOT deleted or modified above.
# =============================

# --- Ensure PAIR_MAP exists at module scope (some upstream edits may have nested it accidentally) ---
if "PAIR_MAP" not in globals():
    PAIR_MAP = {
        "USD/JPY (ドル円)": "JPY=X",
        "EUR/USD (ユーロドル)": "EURUSD=X",
        "GBP/USD (ポンドドル)": "GBPUSD=X",
        "AUD/USD (豪ドル米ドル)": "AUDUSD=X",
        "EUR/JPY (ユーロ円)": "EURJPY=X",
        "GBP/JPY (ポンド円)": "GBPJPY=X",
        "AUD/JPY (豪ドル円)": "AUDJPY=X",
    }

# --- Lightweight helpers (safe float / json extract might already exist) ---
def _fx_safe_float(x, default=None):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _pair_label_to_symbol(pair_label: str) -> str:
    # Prefer explicit ticker in PAIR_MAP
    if pair_label in PAIR_MAP:
        return PAIR_MAP[pair_label]
    # If label already looks like a ticker, return as-is
    if isinstance(pair_label, str) and pair_label.endswith("=X"):
        return pair_label
    # Fallback: try to build e.g. "EURJPY=X" from "EUR/JPY ..."
    try:
        head = pair_label.split()[0]  # "EUR/JPY"
        base, quote = head.split("/")
        return f"{base}{quote}=X"
    except Exception:
        return ""

def _fetch_ohlc(symbol: str, period: str, interval: str):
    """
    Robust OHLC fetch. Uses existing _yahoo_chart if present; else falls back to yfinance.
    """
    try:
        if "_yahoo_chart" in globals():
            df = _yahoo_chart(symbol, rng=period, interval=interval)
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        # Normalize columns
        for c in ["Open","High","Low","Close","Adj Close","Volume"]:
            if c in df.columns and hasattr(df[c], "iloc"):
                # yfinance sometimes returns multi-index columns
                if isinstance(df[c].iloc[0], (pd.Series, pd.DataFrame)):
                    df[c] = df[c].iloc[:,0]
        return df
    except Exception:
        return None

def compute_month_hold_line(pair_label: str, direction: str, buffer_atr_mult: float = 0.2):
    """
    Numeric 1-month hold line based on previous month's high/low (+ optional buffer).
    - LONG: prev_month_high + buffer
    - SHORT: prev_month_low  - buffer
    Buffer is derived from weekly average range as a lightweight ATR proxy.
    """
    sym = _pair_label_to_symbol(pair_label)
    if not sym:
        return None

    mdf = _fetch_ohlc(sym, period="1y", interval="1mo")
    if mdf is None or mdf.empty or len(mdf) < 2:
        return None

    # previous completed month is -2 (last row may be current month in progress)
    prev = mdf.iloc[-2]
    prev_high = _fx_safe_float(prev.get("High"))
    prev_low  = _fx_safe_float(prev.get("Low"))
    if prev_high is None or prev_low is None:
        return None

    # weekly ATR proxy
    wdf = _fetch_ohlc(sym, period="6mo", interval="1wk")
    buf = 0.0
    if wdf is not None and not wdf.empty and len(wdf) >= 8:
        rng = (wdf["High"] - wdf["Low"]).dropna()
        if not rng.empty:
            atr_proxy = float(rng.tail(14).mean())
            buf = atr_proxy * float(buffer_atr_mult)

    d = str(direction).upper()
    if d in ("BUY","LONG"):
        return float(prev_high + buf)
    if d in ("SELL","SHORT"):
        return float(prev_low - buf)
    return None

def compute_weekly_structure_ok(pair_label: str, direction: str, lookback_weeks: int = 4, close_rule: str = "CLOSE_BREAK"):
    """
    Explicit numeric structure_ok rules (weekly timeframe):
    - '週足高値安値更新' : LONG -> higher high, SHORT -> lower low vs lookback window
    - '週足終値が○○以上' : default CLOSE_BREAK:
        LONG -> weekly close >= prior window max high (breakout close)
        SHORT-> weekly close <= prior window min low  (breakdown close)
    Returns (ok: bool, details: dict)
    """
    sym = _pair_label_to_symbol(pair_label)
    if not sym:
        return False, {"reason": "no_symbol"}

    wdf = _fetch_ohlc(sym, period="1y", interval="1wk")
    if wdf is None or wdf.empty or len(wdf) < (lookback_weeks + 2):
        return False, {"reason": "weekly_data_insufficient"}

    # last completed week: use -2 to avoid partial current week
    cur = wdf.iloc[-2]
    hist = wdf.iloc[-(lookback_weeks+2):-2]  # prior lookback_weeks
    cur_high = _fx_safe_float(cur.get("High"))
    cur_low  = _fx_safe_float(cur.get("Low"))
    cur_close= _fx_safe_float(cur.get("Close"))

    if cur_high is None or cur_low is None or cur_close is None:
        return False, {"reason": "weekly_nan"}

    prior_high_max = float(hist["High"].max())
    prior_low_min  = float(hist["Low"].min())
    prior_close_max= float(hist["Close"].max())
    prior_close_min= float(hist["Close"].min())

    d = str(direction).upper()
    if d in ("BUY","LONG"):
        hh = cur_high > prior_high_max
        if close_rule == "CLOSE_BREAK":
            cc = cur_close >= prior_high_max
        else:
            cc = cur_close >= prior_close_max
        ok = bool(hh and cc)
        return ok, {
            "direction": "LONG",
            "higher_high": hh,
            "close_confirm": cc,
            "cur_high": cur_high,
            "cur_close": cur_close,
            "prior_high_max": prior_high_max,
            "prior_close_max": prior_close_max,
        }

    if d in ("SELL","SHORT"):
        ll = cur_low < prior_low_min
        if close_rule == "CLOSE_BREAK":
            cc = cur_close <= prior_low_min
        else:
            cc = cur_close <= prior_close_min
        ok = bool(ll and cc)
        return ok, {
            "direction": "SHORT",
            "lower_low": ll,
            "close_confirm": cc,
            "cur_low": cur_low,
            "cur_close": cur_close,
            "prior_low_min": prior_low_min,
            "prior_close_min": prior_close_min,
        }

    return False, {"reason": "direction_unknown"}

def numeric_hold_month_ok(context_data: dict, buffer_atr_mult: float = 0.2):
    """
    Numeric-only HOLD_MONTH condition:
    - month_hold_line reached (prev month high/low +/- buffer)
    - weekly structure_ok is True (explicit weekly rules)
    Direction is derived from trade_type (BUY/SELL) or decision.
    """
    pair_label = context_data.get("pair_label") or context_data.get("pair") or "USD/JPY (ドル円)"
    price = _fx_safe_float(context_data.get("price"))
    direction = context_data.get("trade_type") or context_data.get("decision") or context_data.get("trend") or ""
    direction = str(direction).upper()

    if price is None:
        return False, {"reason": "no_price"}

    # Normalize direction tokens
    if direction == "BUY":
        d = "LONG"
    elif direction == "SELL":
        d = "SHORT"
    elif direction in ("LONG","SHORT"):
        d = direction
    else:
        return False, {"reason": "no_direction"}

    month_line = compute_month_hold_line(pair_label, d, buffer_atr_mult=buffer_atr_mult)
    if month_line is None:
        return False, {"reason": "no_month_hold_line"}

    structure_ok, sdetail = compute_weekly_structure_ok(pair_label, d)

    if d == "LONG":
        reached = price >= month_line
    else:
        reached = price <= month_line

    ok = bool(reached and structure_ok)
    return ok, {
        "pair": pair_label,
        "direction": d,
        "price": price,
        "month_hold_line": month_line,
        "reached": reached,
        "structure_ok": structure_ok,
        "structure_detail": sdetail,
    }

# --- Wrapper: enforce numeric-only HOLD_MONTH without deleting existing prompt logic ---
if "get_ai_weekend_decision" in globals():
    _old_get_ai_weekend_decision = get_ai_weekend_decision

    def get_ai_weekend_decision(api_key, context_data, override_mode="AUTO", override_reason="", **kwargs):
        obj = _old_get_ai_weekend_decision(api_key, context_data, override_mode=override_mode, override_reason=override_reason)

        # If no position, keep as-is
        try:
            ep = float(context_data.get("entry_price", 0.0) or 0.0)
        except Exception:
            ep = 0.0

        if ep <= 0:
            return obj

        ok, detail = numeric_hold_month_ok(context_data)

        # Always annotate notes with numeric checks (for audit)
        try:
            if isinstance(obj, dict):
                notes = obj.get("notes")
                if not isinstance(notes, list):
                    notes = []
                notes.append(f"numeric_hold_month_ok={ok}")
                notes.append(f"month_hold_line={detail.get('month_hold_line','')}")
                notes.append(f"weekly_structure_ok={detail.get('structure_ok','')}")
                obj["notes"] = notes
                # expose computed levels for UI/log (non-breaking)
                obj.setdefault("levels", {})
                obj["levels"].setdefault("month_hold_line", detail.get("month_hold_line", 0))
        except Exception:
            pass

        # Enforce: HOLD_MONTH only when numeric rules pass
        if isinstance(obj, dict) and obj.get("action") == "HOLD_MONTH" and not ok:
            obj["action"] = "HOLD_WEEK"
            why = (obj.get("why") or "").strip()
            addon = "（数値ルール: month_hold_line未達 or 週足構造未確認のためHOLD_MONTHをHOLD_WEEKに降格）"
            obj["why"] = (why + " " + addon).strip() if why else addon

        # If numeric rules pass, promote to HOLD_MONTH only when AI intends to hold (safety-first)
        if isinstance(obj, dict) and ok:
            a = obj.get("action")
            if a in ("HOLD_WEEK", "HOLD_MONTH"):
                obj["action"] = "HOLD_MONTH"
                why = (obj.get("why") or "").strip()
                addon = "（数値ルールで1か月継続条件を満たしたためHOLD_MONTH）"
                obj["why"] = (why + " " + addon).strip() if why else addon

        return obj

# --- Wrapper: make get_ai_order_strategy accept pair_name/portfolio args used by main_auto (no deletions) ---
if "get_ai_order_strategy" in globals():
    _old_get_ai_order_strategy = get_ai_order_strategy

    def get_ai_order_strategy(api_key, context_data, override_mode="AUTO", override_reason="", pair_name=None,
                              portfolio_positions=None, weekly_dd_cap_percent=2.0, risk_percent_per_trade=2.0,
                              max_positions_per_currency=1, **kwargs):
        """
        Backward/forward compatible wrapper:
        - Keeps original prompt/logic (calls old function)
        - Adds weekly DD cap + currency concentration gates if portfolio_positions is provided
        """
        # If portfolio provided, gate before calling AI (fast fail)
        if portfolio_positions is not None:
            if not can_open_under_weekly_cap(portfolio_positions, risk_percent_per_trade, weekly_dd_cap_percent):
                return {
                    "decision": "NO_TRADE",
                    "confidence": 1.0,
                    "why": "週単位DDキャップ超過のため新規不可",
                    "notes": ["blocked_by_weekly_dd_cap"]
                }
            if pair_name:
                if violates_currency_concentration(pair_name, portfolio_positions, max_positions_per_currency=max_positions_per_currency):
                    return {
                        "decision": "NO_TRADE",
                        "confidence": 1.0,
                        "why": "通貨集中（簡易相関）フィルタにより新規不可",
                        "notes": ["blocked_by_currency_concentration"]
                    }

        # call original function unmodified
        return _old_get_ai_order_strategy(api_key, context_data, override_mode=override_mode, override_reason=override_reason)

# --- Promote nested helper defs to module scope if missing (fix for earlier indentation issues) ---
if "portfolio_weekly_risk_percent" not in globals():
    def portfolio_weekly_risk_percent(active_positions: list) -> float:
        total = 0.0
        for p in active_positions or []:
            try:
                total += float(p.get("risk_percent", p.get("risk", 0.0)))
            except Exception:
                continue
        return float(total)

if "portfolio_currency_counts" not in globals():
    def portfolio_currency_counts(active_positions: list) -> dict:
        counts = {}
        for p in active_positions or []:
            pair = p.get("pair") or p.get("pair_label") or p.get("pair_name") or ""
            if not pair:
                continue
            b, q = _pair_label_to_currencies(pair) if "_pair_label_to_currencies" in globals() else (pair[:3], pair[4:7])
            counts[b] = counts.get(b, 0) + 1
            counts[q] = counts.get(q, 0) + 1
        return counts

if "violates_currency_concentration" not in globals():
    def violates_currency_concentration(candidate_pair_label: str, active_positions: list, max_positions_per_currency: int = 1) -> bool:
        counts = portfolio_currency_counts(active_positions)
        b, q = _pair_label_to_currencies(candidate_pair_label) if "_pair_label_to_currencies" in globals() else (candidate_pair_label[:3], candidate_pair_label[4:7])
        return (counts.get(b, 0) + 1 > max_positions_per_currency) or (counts.get(q, 0) + 1 > max_positions_per_currency)

if "can_open_under_weekly_cap" not in globals():
    def can_open_under_weekly_cap(active_positions: list, new_risk_percent: float, weekly_dd_cap_percent: float) -> bool:
        try:
            new_risk = float(new_risk_percent)
            cap = float(weekly_dd_cap_percent)
        except Exception:
            return True
        return (portfolio_weekly_risk_percent(active_positions) + new_risk) <= cap

if "suggest_alternative_pair_if_usdjpy_stay" not in globals():
    def _build_market_summary_for_pairs() -> str:
        market_summary = ""
        for name, sym in PAIR_MAP.items():
            try:
                df = _fetch_ohlc(sym, period="5d", interval="1d")
                if df is None or df.empty:
                    continue
                close = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
                chg = (close - prev) / prev * 100 if prev else 0.0
                trend = "上昇" if close > float(df["Close"].mean()) else "下降"
                market_summary += f"- {name}: Price={close:.3f} / Chg={chg:+.2f}% / Trend={trend}\n"
            except Exception:
                continue
        return market_summary

    def suggest_alternative_pair_if_usdjpy_stay(
        api_key: str,
        active_positions: list,
        risk_percent_per_trade: float,
        weekly_dd_cap_percent: float = 2.0,
        max_positions_per_currency: int = 1,
        exclude_pair_label: str = "USD/JPY (ドル円)"
    ) -> dict:
        model_name = get_active_model(api_key)
        if not model_name:
            return {}

        if not can_open_under_weekly_cap(active_positions, risk_percent_per_trade, weekly_dd_cap_percent):
            return {"best_pair_name": "", "reason": "週単位DDキャップ超過", "confidence": 1.0, "blocked": True, "blocked_by": "weekly_dd_cap"}

        market_summary = _build_market_summary_for_pairs()

        prompt = f"""
あなたはプロのFXファンドマネージャーです。
以下の市場データから「今週、USD/JPYが見送りのときに代替として最も利益チャンスがありそうなペア」を最大5つ、優先順で提案してください。
ただし同じ通貨への偏り（例: JPY絡みを複数）を避ける観点も考慮してください。

【市場概況】
{market_summary}

【必須制約】
- 可能なら {exclude_pair_label} 以外から選ぶ
- 出力はJSONのみ

【出力JSON】
{{
  "candidates": [
    {{"pair": "EUR/USD (ユーロドル)", "reason": "...", "confidence": 0.0}},
    {{"pair": "AUD/JPY (豪ドル円)", "reason": "...", "confidence": 0.0}}
  ]
}}
"""
        candidates = []
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            txt = (getattr(resp, "text", "") or "").strip()
            # try to parse using existing helpers
            if "safe_json_loads" in globals():
                data = safe_json_loads(txt)
            else:
                data = json.loads(_extract_json(txt)) if "_extract_json" in globals() else json.loads(txt)
            candidates = data.get("candidates", []) or []
        except Exception:
            pass

        for c in candidates:
            pair = c.get("pair", "")
            if not pair or pair == exclude_pair_label:
                continue
            if violates_currency_concentration(pair, active_positions, max_positions_per_currency=max_positions_per_currency):
                continue
            return {"best_pair_name": pair, "reason": c.get("reason", ""), "confidence": c.get("confidence", 0.5), "blocked": False}

        return {"best_pair_name": "", "reason": "条件を満たす代替ペアなし", "confidence": 0.0, "blocked": True, "blocked_by": "filters"}



# =====================================================
# ALT_PAIR_FIXES v2 (2026-02-09)
# - Fix: suggest_alternative_pair_if_usdjpy_stay() JSON parse NameError by providing
#        _extract_json and safe_json_loads.
# - Add: numeric_scan_best_pair() fallback that scans PAIR_MAP with the SAME trend_only_gate
#        (so "USD/JPY見送りでも他ペアがトレンドならTRADE" を確実に拾う)
# - Override: suggest_alternative_pair_if_usdjpy_stay() to use AI candidates first, then numeric scan fallback.
# NOTE: Existing algorithms/prompts above are NOT deleted.
# =====================================================

def _extract_json(text: str) -> str:
    """Backward-compat alias (older code referenced _extract_json)."""
    return _extract_json_block(text)

def safe_json_loads(text: str) -> dict:
    """Robust JSON loader for LLM outputs (supports code fences / extra text)."""
    j = _extract_json_block(text)
    if not j:
        # try stripping ```json fences
        s = (text or "").strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        j = _extract_json_block(s)
    if not j:
        raise ValueError("json_not_found")
    return json.loads(j)

def _build_ctx_from_indicator_df(df_ind: pd.DataFrame) -> dict:
    """Build minimal context required by trend_only_gate from an indicator DataFrame."""
    lr = df_ind.iloc[-1]
    ctx = {
        "price": float(lr["Close"]) if pd.notna(lr.get("Close", None)) else None,
        "atr": float(lr["ATR"]) if pd.notna(lr.get("ATR", None)) else None,
        "rsi": float(lr["RSI"]) if pd.notna(lr.get("RSI", None)) else None,
        "sma25": float(lr["SMA_25"]) if pd.notna(lr.get("SMA_25", None)) else None,
        "sma75": float(lr["SMA_75"]) if pd.notna(lr.get("SMA_75", None)) else None,
    }
    try:
        if "ATR" in df_ind.columns and df_ind["ATR"].tail(60).notna().any():
            ctx["atr_avg60"] = float(df_ind["ATR"].tail(60).mean())
        else:
            ctx["atr_avg60"] = ctx.get("atr")
    except Exception:
        ctx["atr_avg60"] = ctx.get("atr")
    return ctx

def numeric_scan_best_pair(
    active_positions: list = None,
    exclude_pair_label: str = "USD/JPY (ドル円)",
    max_positions_per_currency: int = 1,
) -> dict:
    """Deterministic fallback: scan PAIR_MAP and pick the strongest pair that passes trend_only_gate."""
    active_positions = active_positions or []
    exclude_head = (exclude_pair_label or "").split()[0]

    best_pair = ""
    best_score = -1e9
    best_meta = {}

    for pair_label, sym in (PAIR_MAP or {}).items():
        try:
            head = (pair_label or "").split()[0]
            if head and head == exclude_head:
                continue

            # If there are already positions, apply currency concentration filter.
            # If no positions (ノーポジ), concentration check is meaningless, so we keep it permissive.
            if active_positions:
                try:
                    if violates_currency_concentration(pair_label, active_positions, max_positions_per_currency=max_positions_per_currency):
                        continue
                except Exception:
                    pass

            # Fetch OHLC
            raw = None
            try:
                if "_fetch_ohlc" in globals():
                    raw = _fetch_ohlc(sym, period="1y", interval="1d")
                else:
                    raw = _yahoo_chart(sym, rng="1y", interval="1d")
            except Exception:
                raw = None

            df_ind = calculate_indicators(raw, None) if raw is not None else None
            # Need enough length for SMA75 and ATR
            if df_ind is None or getattr(df_ind, "empty", True) or len(df_ind) < 80:
                continue

            ctx = _build_ctx_from_indicator_df(df_ind)
            allowed, side_hint, trend_score, reasons = trend_only_gate(ctx)
            if not allowed or trend_score is None:
                continue

            score = float(trend_score)
            if score > best_score:
                best_score = score
                best_pair = pair_label
                best_meta = {
                    "side_hint": side_hint,
                    "trend_score": score,
                    "price": ctx.get("price"),
                    "rsi": ctx.get("rsi"),
                    "atr": ctx.get("atr"),
                }
        except Exception:
            continue

    if not best_pair:
        return {
            "best_pair_name": "",
            "reason": "数値スキャンでもトレンド合格ペアが見つかりませんでした",
            "confidence": 0.0,
            "blocked": True,
            "blocked_by": "numeric_scan_no_candidate",
        }

    return {
        "best_pair_name": best_pair,
        "reason": f"数値スキャンでトレンド合格（trend_score={best_meta.get('trend_score'):.2f}, RSI={best_meta.get('rsi'):.1f}）",
        "confidence": 0.65,
        "blocked": False,
        "source": "numeric_scan",
        "meta": best_meta,
    }

# Provide scan_best_pair for older fallback paths (used by earlier suggest_alternative impl)
def scan_best_pair(api_key: str = None) -> dict:
    try:
        return numeric_scan_best_pair(active_positions=[])
    except Exception:
        return {"best_pair_name": "", "reason": "scan_best_pair_failed", "confidence": 0.0}

# Override alternative suggestion to be robust and transparent
def suggest_alternative_pair_if_usdjpy_stay(
    api_key: str,
    active_positions: list,
    risk_percent_per_trade: float,
    weekly_dd_cap_percent: float = 2.0,
    max_positions_per_currency: int = 1,
    exclude_pair_label: str = "USD/JPY (ドル円)",
) -> dict:
    """
    代替ペア提案（安全 + 取りこぼし防止）:
      1) 週DDキャップを先にチェック
      2) AI候補(JSON) → フィルタ
      3) AIがコケる/空なら、数値スキャン(trend_only_gate合格)で必ず拾いに行く
    """
    # Weekly cap gate first
    if not can_open_under_weekly_cap(active_positions, risk_percent_per_trade, weekly_dd_cap_percent):
        return {
            "best_pair_name": "",
            "reason": "週単位DDキャップを超えるため今週は新規不可",
            "confidence": 1.0,
            "blocked": True,
            "blocked_by": "weekly_dd_cap",
        }

    # If model is unavailable, still try numeric scan
    model_name = get_active_model(api_key)
    if not model_name:
        return numeric_scan_best_pair(
            active_positions=active_positions,
            exclude_pair_label=exclude_pair_label,
            max_positions_per_currency=max_positions_per_currency,
        )

    market_summary = _build_market_summary_for_pairs()

    # Prompt: do not over-constrain when portfolio is empty
    bias_note = "（現在ノーポジのため、JPYクロスも候補に含めてOK）" if not (active_positions or []) else "（既存ポジとの通貨重複を避ける）"
    allowed_labels = list(PAIR_MAP.keys()) if "PAIR_MAP" in globals() else []

    prompt = f"""
あなたはプロのFXファンドマネージャーです。
以下の市場データから「今週、USD/JPYが見送りのときに代替として最も利益チャンスがありそうなペア」を最大5つ、優先順で提案してください。
{bias_note}

【市場概況】
{market_summary}

【必須制約】
- 可能なら {exclude_pair_label} 以外から選ぶ
- 出力はJSONのみ
- pair は必ず次の候補のいずれかから選ぶ（表記は完全一致）:
{allowed_labels}

【出力JSON】
{{
  "candidates": [
    {{"pair": "EUR/USD (ユーロドル)", "reason": "...", "confidence": 0.0}},
    {{"pair": "AUD/JPY (豪ドル円)", "reason": "...", "confidence": 0.0}}
  ]
}}
""".strip()

    candidates = []
    ai_error = ""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = (resp.text or "").strip()
        data = safe_json_loads(txt)
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
    except Exception as e:
        ai_error = f"{type(e).__name__}: {e}"
        candidates = []

    exclude_head = (exclude_pair_label or "").split()[0]

    # Filter + pick from AI candidates
    for c in candidates or []:
        try:
            pair = (c or {}).get("pair", "")
            if not pair:
                continue

            # Exclude by head match (more robust than exact Japanese label)
            if (pair.split()[0] == exclude_head):
                continue

            # If already have positions, apply concentration
            if active_positions:
                if violates_currency_concentration(pair, active_positions, max_positions_per_currency=max_positions_per_currency):
                    continue

            return {
                "best_pair_name": pair,
                "reason": (c or {}).get("reason", ""),
                "confidence": float((c or {}).get("confidence", 0.5) or 0.5),
                "blocked": False,
                "source": "ai",
            }
        except Exception:
            continue

    # Fallback: numeric scan (guarantees "他ペアがトレンドなら拾う")
    fb = numeric_scan_best_pair(
        active_positions=active_positions,
        exclude_pair_label=exclude_pair_label,
        max_positions_per_currency=max_positions_per_currency,
    )
    if ai_error:
        fb.setdefault("debug", {})
        fb["debug"]["ai_candidate_error"] = ai_error
    return fb


# =====================================================
# ALT_PAIR_FIXES v3 (2026-02-09)
# - Make alternative suggestions "TRADE-able":
#   * After AI proposes candidates, we fetch that pair's OHLC and compute indicators,
#     then apply the SAME numeric gates used for weekly trade safety:
#       - trend_only_gate (trend week only)
#       - no_trade_gate (volatility/MA-convergence gate) [conservative: force_defensive]
#   * If candidate fails gates, we skip to next candidate.
#   * numeric_scan_best_pair() is also tightened to require passing both gates.
#   This prevents: "代替ペアは出たのに、注文書がNO_TRADEで0だらけ" confusion.
# NOTE: Existing algorithms/prompts above are NOT deleted.
# =====================================================

def _alt_pair_build_indicator_df(pair_label: str):
    """Fetch 1y daily OHLC for the given PAIR_MAP label and return indicator df.
    Returns None on failure."""
    try:
        sym = (PAIR_MAP or {}).get(pair_label)
        if not sym:
            return None
        # Prefer _fetch_ohlc if available (stable), else fallback to _yahoo_chart
        raw = None
        try:
            if "_fetch_ohlc" in globals():
                raw = _fetch_ohlc(sym, period="1y", interval="1d")
            else:
                raw = _yahoo_chart(sym, rng="1y", interval="1d")
        except Exception:
            raw = None
        if raw is None or getattr(raw, "empty", True):
            return None
        # Reuse existing indicator pipeline
        df_ind = calculate_indicators(raw, None)
        if df_ind is None or getattr(df_ind, "empty", True):
            return None
        # Need enough bars for SMA75/ATR avg
        if len(df_ind) < 90:
            return None
        return df_ind
    except Exception:
        return None


def _alt_pair_tradeable_precheck(pair_label: str):
    """Return (ok, debug_dict). ok=True means the pair is eligible to produce a TRADE plan.

    We intentionally use force_defensive=True for no_trade_gate to avoid 'too wild' weeks.
    """
    dbg = {"pair": pair_label, "ok": False, "why": "", "notes": []}

    df_ind = _alt_pair_build_indicator_df(pair_label)
    if df_ind is None:
        dbg["why"] = "indicator_df_unavailable"
        return False, dbg

    ctx = _build_ctx_from_indicator_df(df_ind) if "_build_ctx_from_indicator_df" in globals() else {}

    # Add fields used by no_trade_gate (optional but helps)
    try:
        # atr_avg60 is already set by _build_ctx_from_indicator_df; keep
        # include latest SMA values explicitly if present
        lr = df_ind.iloc[-1]
        if "SMA_25" in df_ind.columns and pd.notna(lr.get("SMA_25")):
            ctx["sma25"] = float(lr["SMA_25"])
        if "SMA_75" in df_ind.columns and pd.notna(lr.get("SMA_75")):
            ctx["sma75"] = float(lr["SMA_75"])
        # also include price
        if "Close" in df_ind.columns and pd.notna(lr.get("Close")):
            ctx["price"] = float(lr["Close"])
    except Exception:
        pass

    # 1) trend-only gate
    ok_trend, side_hint, trend_score, trend_reasons = trend_only_gate(ctx)
    if not ok_trend:
        dbg["why"] = "trend_only_gate_block"
        dbg["notes"].extend(trend_reasons or [])
        dbg["trend_score"] = trend_score
        dbg["side_hint"] = side_hint
        return False, dbg

    # 2) no-trade gate (conservative)
    try:
        nt, regime, nt_reasons = no_trade_gate(ctx, "DEFENSIVE", force_defensive=True)
    except Exception as e:
        nt, regime, nt_reasons = True, "DEFENSIVE", [f"no_trade_gate_error:{type(e).__name__}"]

    # enrich debug with volatility ratio
    try:
        atr = _safe_float(ctx.get("atr"))
        atr_avg60 = _safe_float(ctx.get("atr_avg60"))
        if atr is not None and atr_avg60 not in (None, 0):
            dbg["atr_ratio"] = float(atr) / float(atr_avg60)
    except Exception:
        pass

    if nt:
        dbg["why"] = "no_trade_gate_block"
        dbg["notes"].extend(nt_reasons or [])
        dbg["regime"] = regime
        dbg["trend_score"] = trend_score
        dbg["side_hint"] = side_hint
        return False, dbg

    dbg["ok"] = True
    dbg["why"] = "tradeable"
    dbg["trend_score"] = trend_score
    dbg["side_hint"] = side_hint
    return True, dbg


# Tighten numeric scan to require both gates
try:
    _old_numeric_scan_best_pair_v2 = numeric_scan_best_pair
except Exception:
    _old_numeric_scan_best_pair_v2 = None


def numeric_scan_best_pair(
    active_positions: list = None,
    exclude_pair_label: str = "USD/JPY (ドル円)",
    max_positions_per_currency: int = 1,
) -> dict:
    active_positions = active_positions or []
    exclude_head = (exclude_pair_label or "").split()[0]

    best_pair = ""
    best_score = -1e9
    best_meta = {}

    for pair_label in (PAIR_MAP or {}).keys():
        try:
            head = (pair_label or "").split()[0]
            if head and head == exclude_head:
                continue

            # Concentration filter only when positions exist
            if active_positions:
                try:
                    if violates_currency_concentration(pair_label, active_positions, max_positions_per_currency=max_positions_per_currency):
                        continue
                except Exception:
                    pass

            ok, dbg = _alt_pair_tradeable_precheck(pair_label)
            if not ok:
                continue

            score = float(dbg.get("trend_score") or -1e9)
            if score > best_score:
                best_score = score
                best_pair = pair_label
                best_meta = dbg
        except Exception:
            continue

    if not best_pair:
        return {
            "best_pair_name": "",
            "reason": "数値スキャンでもTRADE可能な代替ペアが見つかりませんでした（トレンド条件/荒れ相場ゲートで全落ち）",
            "confidence": 0.0,
            "blocked": True,
            "blocked_by": "numeric_scan_no_tradeable_candidate",
        }

    return {
        "best_pair_name": best_pair,
        "reason": f"数値スキャンでTRADE可能（trend_score={best_meta.get('trend_score'):.2f}, ATR比={best_meta.get('atr_ratio','?')}）",
        "confidence": 0.70,
        "blocked": False,
        "source": "numeric_scan_tradeable",
        "meta": best_meta,
    }


# Override: choose only tradeable AI candidates; else fallback numeric scan (tradeable)
try:
    _old_suggest_alternative_v2 = suggest_alternative_pair_if_usdjpy_stay
except Exception:
    _old_suggest_alternative_v2 = None


def suggest_alternative_pair_if_usdjpy_stay(
    api_key: str,
    active_positions: list,
    risk_percent_per_trade: float,
    weekly_dd_cap_percent: float = 2.0,
    max_positions_per_currency: int = 1,
    exclude_pair_label: str = "USD/JPY (ドル円)",
) -> dict:
    # Weekly cap first
    if not can_open_under_weekly_cap(active_positions, risk_percent_per_trade, weekly_dd_cap_percent):
        return {
            "best_pair_name": "",
            "reason": "週単位DDキャップを超えるため今週は新規不可",
            "confidence": 1.0,
            "blocked": True,
            "blocked_by": "weekly_dd_cap",
        }

    model_name = get_active_model(api_key)

    # If model unavailable, deterministic scan
    if not model_name:
        return numeric_scan_best_pair(
            active_positions=active_positions,
            exclude_pair_label=exclude_pair_label,
            max_positions_per_currency=max_positions_per_currency,
        )

    market_summary = _build_market_summary_for_pairs()
    bias_note = "（現在ノーポジのため、JPYクロスも候補に含めてOK）" if not (active_positions or []) else "（既存ポジとの通貨重複を避ける）"
    allowed_labels = list((PAIR_MAP or {}).keys())

    prompt = f"""
あなたはプロのFXファンドマネージャーです。
以下の市場データから「今週、USD/JPYが見送りのときに代替として最も利益チャンスがありそうなペア」を最大5つ、優先順で提案してください。
{bias_note}

【市場概況】
{market_summary}

【必須制約】
- 可能なら {exclude_pair_label} 以外から選ぶ
- 出力はJSONのみ
- pair は必ず次の候補のいずれかから選ぶ（表記は完全一致）:
{allowed_labels}

【出力JSON】
{{
  "candidates": [
    {{"pair": "EUR/USD (ユーロドル)", "reason": "...", "confidence": 0.0}},
    {{"pair": "AUD/JPY (豪ドル円)", "reason": "...", "confidence": 0.0}}
  ]
}}
""".strip()

    candidates = []
    ai_error = ""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = (resp.text or "").strip()
        data = safe_json_loads(txt)
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
    except Exception as e:
        ai_error = f"{type(e).__name__}: {e}"
        candidates = []

    exclude_head = (exclude_pair_label or "").split()[0]

    # AI candidates -> apply portfolio filters -> then tradeable precheck
    for c in candidates or []:
        try:
            pair = (c or {}).get("pair", "")
            if not pair:
                continue
            if pair.split()[0] == exclude_head:
                continue

            # concentration only when positions exist
            if active_positions:
                if violates_currency_concentration(pair, active_positions, max_positions_per_currency=max_positions_per_currency):
                    continue

            ok, dbg = _alt_pair_tradeable_precheck(pair)
            if not ok:
                # skip non-tradeable candidates
                continue

            return {
                "best_pair_name": pair,
                "reason": (c or {}).get("reason", ""),
                "confidence": float((c or {}).get("confidence", 0.5) or 0.5),
                "blocked": False,
                "source": "ai_tradeable",
                "meta": dbg,
            }
        except Exception:
            continue

    # fallback numeric scan (tradeable)
    fb = numeric_scan_best_pair(
        active_positions=active_positions,
        exclude_pair_label=exclude_pair_label,
        max_positions_per_currency=max_positions_per_currency,
    )
    if ai_error:
        fb.setdefault("debug", {})
        fb["debug"]["ai_candidate_error"] = ai_error
    return fb



# ==========================================================
# AUTO_HIERARCHY_ORDER_GENERATION v1
#   - 迷わない運用：AI →（失敗時）AI再生成 →（さらに失敗時）数値フォールバック
#   - 「AI厳格」も残す（generation_policy="AI_STRICT"）
# ==========================================================

# 既存の get_ai_order_strategy（厳格1回生成）を退避
try:
    _STRICT_GET_AI_ORDER_STRATEGY = get_ai_order_strategy
except Exception:
    _STRICT_GET_AI_ORDER_STRATEGY = None

def _is_technical_failure_order(result: dict) -> bool:
    """AI出力の整形/検証失敗など『本来は取引可能性があるのに止まった』ケースだけ True。"""
    if not isinstance(result, dict):
        return True

    decision = result.get("decision")
    why = str(result.get("why", "") or "")
    notes = result.get("notes", [])
    notes_s = " ".join([str(x) for x in notes]) if isinstance(notes, list) else str(notes)

    # TRADEなら当然OK
    if decision == "TRADE":
        return False

    # ここは「正しい見送り」なので技術失敗扱いしない
    legit_markers = [
        "トレンド条件を満たさない",
        "trend_gate_",          # トレンド週のみ運用
        "NO_TRADEゲート",       # 数値ゲートで止めた
        "volatility_too_high",  # 数値ゲート系
        "data_invalid_",        # 指標欠損（AIの再生成では直らない）
        "weekly_dd_cap",        # DDキャップ
        "currency_concentration",
    ]
    if any(m in why for m in legit_markers) or any(m in notes_s for m in legit_markers):
        return False

    # 『AIの出力が壊れてる/遠すぎる/JSON不正』系は再生成・数値フォールバック対象
    tech_markers = [
        "parse_or_validation_failed",
        "json_extract_failed",
        "order_json_not_object",
        "decision_invalid",
        "side_invalid",
        "horizon_invalid",
        "entry_missing",
        "take_profit_missing",
        "stop_loss_missing",
        "levels_inconsistent_",
        "rr_too_low",
        "entry_too_far_from_price",
        "注文JSONの検証に失敗",
        "注文生成で例外",
        "(ResourceExhausted)",
        "ResourceExhausted",
        "quota",
        "429",
    ]
    return any(m in why for m in tech_markers) or any(m in notes_s for m in tech_markers)

def _is_quota_error(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    why = str(result.get("why", "") or "")
    return ("ResourceExhausted" in why) or ("quota" in why) or ("429" in why)

def _ensure_trend_fields(ctx: dict) -> dict:
    """trend_only_gate で使う補助フィールドを確実に作る（既にあれば上書きしない）。"""
    if not isinstance(ctx, dict):
        return {}
    # base側でセットされる想定だが、保険で補完
    try:
        price = _safe_float(ctx.get("price"))
        atr = _safe_float(ctx.get("atr"))
        if price is None or atr is None:
            return ctx
        if "breakout_buffer" not in ctx:
            ctx["breakout_buffer"] = max(atr * 0.35, price * 0.0025)  # 0.25% or 0.35ATR
        if "recommended_entry" not in ctx:
            side_hint = ctx.get("trend_side_hint")
            if side_hint not in ("LONG", "SHORT"):
                side_hint = "LONG"
            try:
                ctx["recommended_entry"] = _recommended_stop_entry(price, ctx["breakout_buffer"], side_hint)
            except Exception:
                ctx["recommended_entry"] = price
    except Exception:
        pass
    return ctx

def _build_numeric_fallback_order(ctx: dict, market_regime: str, regime_why: str, pair_name: str = "") -> dict:
    """AI不調時の『止まらない』最終手段（数値ルール）。"""
    ctx = _ensure_trend_fields(ctx or {})
    price = _safe_float(ctx.get("price"), default=0.0) or 0.0
    atr = _safe_float(ctx.get("atr"), default=0.0) or 0.0

    side = ctx.get("trend_side_hint")
    if side not in ("LONG", "SHORT"):
        # side_hintが無いなら安全側で見送り
        return {
            "decision": "NO_TRADE", "side": "NONE", "entry": 0, "take_profit": 0, "stop_loss": 0,
            "horizon": "WEEK", "confidence": 0.0,
            "why": "数値フォールバック: side_hint不明のため取引停止。",
            "notes": ["numeric_fallback_side_unknown"],
            "market_regime": market_regime, "regime_why": regime_why,
            "generator_path": "numeric_fallback_blocked"
        }

    # 価格が取れない/ATRが取れないなら停止
    if price <= 0 or atr <= 0:
        return {
            "decision": "NO_TRADE", "side": "NONE", "entry": 0, "take_profit": 0, "stop_loss": 0,
            "horizon": "WEEK", "confidence": 0.0,
            "why": "数値フォールバック: price/ATR不足のため取引停止。",
            "notes": ["numeric_fallback_missing_price_or_atr"],
            "market_regime": market_regime, "regime_why": regime_why,
            "generator_path": "numeric_fallback_blocked"
        }

    # entryは推奨逆指値（ブレイクアウト）に合わせる
    entry = _safe_float(ctx.get("recommended_entry"), default=price)
    # リスク幅（ATRベース）
    risk = max(atr * 2.0, price * 0.004)  # 0.4% or 2ATR
    reward = risk * 2.0  # RR=2.0 目安

    if side == "LONG":
        sl = entry - risk
        tp = entry + reward
        entry_kind = "STOP"
        entry_kind_jp = "逆指値"
        bundle = "IFD_OCO"
        bundle_hint = "SBI: 逆指値(IFD-OCO)で放置運用（TP/SL同時）"
    else:
        sl = entry + risk
        tp = entry - reward
        entry_kind = "STOP"
        entry_kind_jp = "逆指値"
        bundle = "IFD_OCO"
        bundle_hint = "SBI: 逆指値(IFD-OCO)で放置運用（TP/SL同時）"

    obj = {
        "decision": "TRADE",
        "side": side,
        "entry": float(entry),
        "take_profit": float(tp),
        "stop_loss": float(sl),
        "horizon": "WEEK",
        "confidence": 0.60,
        "why": "AIの注文JSONが不正/失敗したため、数値ルールでフォールバック生成（RR=2.0 / ATR基準）。",
        "notes": ["numeric_fallback_rr2_atr", "order_method_fixed_ifd_oco_stop"],
        "order_bundle": bundle,
        "entry_type": entry_kind,
        "entry_price_kind_jp": entry_kind_jp,
        "bundle_hint_jp": bundle_hint,
        "market_regime": market_regime,
        "regime_why": regime_why,
        "generator_path": "numeric_fallback"
    }

    ok, reasons = _validate_order_json(obj, ctx)
    if not ok:
        return {
            "decision": "NO_TRADE", "side": "NONE", "entry": 0, "take_profit": 0, "stop_loss": 0,
            "horizon": "WEEK", "confidence": 0.0,
            "why": "数値フォールバック生成は試みたが、検証に失敗したため取引停止。",
            "notes": ["numeric_fallback_validation_failed"] + reasons,
            "market_regime": market_regime, "regime_why": regime_why,
            "generator_path": "numeric_fallback_failed"
        }
    return obj

def _ai_retry_order(api_key: str, ctx: dict, market_regime: str, regime_why: str, pair_name: str = "") -> dict:
    """AIの再生成（失敗時のみ）。JSON仕様を厳しくしてもう一度だけ作る。"""
    try:
        model_name = get_active_model(api_key)
        if not model_name:
            return {}
        ctx = _ensure_trend_fields(ctx or {})
        price = float(_safe_float(ctx.get("price"), default=0.0) or 0.0)
        atr = float(_safe_float(ctx.get("atr"), default=0.0) or 0.0)
        rsi = float(_safe_float(ctx.get("rsi"), default=50.0) or 50.0)
        sma5 = _safe_float(ctx.get("sma5"))
        sma25 = _safe_float(ctx.get("sma25"))
        sma75 = _safe_float(ctx.get("sma75"))
        side_hint = ctx.get("trend_side_hint", "LONG")
        recommended_entry = float(_safe_float(ctx.get("recommended_entry"), default=price) or price)

        prompt = f"""
あなたはFXの運用担当です。以下の条件で、**必ずJSONのみ**を返してください。装飾や説明文は禁止です。

【ペア】{pair_name or ctx.get("pair_label","")}
【現値】{price:.6f}
【ATR】{atr:.6f}
【RSI】{rsi:.2f}
【SMA5】{sma5}
【SMA25】{sma25}
【SMA75】{sma75}
【相場モード】{market_regime}
【相場理由】{regime_why}

【運用ルール（厳守）】
- 今週はトレンド週のみ。方向ヒント: {side_hint}
- エントリーはブレイクアウト用の**逆指値**（STOP）で作る
- 推奨エントリー: {recommended_entry:.6f} を必ず基準にする
- entryは現値から3%以内（entry_too_farを回避）
- RRは1.2以上（できれば1.6〜2.4）
- 数値はすべて number（文字列禁止）

【出力JSONスキーマ】
{{
  "decision": "TRADE" or "NO_TRADE",
  "side": "LONG" or "SHORT" or "NONE",
  "entry": number,
  "take_profit": number,
  "stop_loss": number,
  "horizon": "WEEK",
  "confidence": number,
  "why": "string",
  "notes": ["string", ...],
  "order_bundle": "IFD_OCO",
  "entry_type": "STOP",
  "entry_price_kind_jp": "逆指値",
  "bundle_hint_jp": "SBI: 逆指値(IFD-OCO)で放置運用（TP/SL同時）"
}}
"""

        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        txt = (getattr(resp, "text", "") or "").strip()
        data = safe_json_loads(txt) if 'safe_json_loads' in globals() else json.loads(_extract_json(txt))
        if not isinstance(data, dict):
            return {}

        # トレンド側ヒントと矛盾するなら見送り（運用ルール）
        if data.get("decision") == "TRADE":
            if side_hint in ("LONG","SHORT") and data.get("side") not in (side_hint,):
                data["decision"] = "NO_TRADE"
                data["side"] = "NONE"
                data["entry"] = 0
                data["take_profit"] = 0
                data["stop_loss"] = 0
                data["why"] = "AI再生成のsideがトレンド方向ヒントと不一致のため取引停止。"
                data["notes"] = ["retry_side_mismatch"]

        # entryは推奨に寄せる（暴発防止）
        if data.get("decision") == "TRADE":
            try:
                data["entry"] = float(recommended_entry)
            except Exception:
                pass

        ok, reasons = _validate_order_json(data, ctx)
        if not ok:
            return {
                "decision": "NO_TRADE", "side": "NONE", "entry": 0, "take_profit": 0, "stop_loss": 0,
                "horizon": "WEEK", "confidence": 0.0,
                "why": "AI再生成でも注文JSONの検証に失敗したため取引停止。",
                "notes": ["retry_parse_or_validation_failed"] + reasons,
                "market_regime": market_regime, "regime_why": regime_why,
                "generator_path": "ai_retry_failed"
            }

        data["market_regime"] = market_regime
        data["regime_why"] = regime_why
        data["generator_path"] = "ai_retry"
        return data
    except Exception as e:
        return {
            "decision": "NO_TRADE", "side": "NONE", "entry": 0, "take_profit": 0, "stop_loss": 0,
            "horizon": "WEEK", "confidence": 0.0,
            "why": f"AI再生成で例外。保守的に取引停止。({type(e).__name__})",
            "notes": ["retry_exception"],
            "market_regime": market_regime, "regime_why": regime_why,
            "generator_path": "ai_retry_exception"
        }

def get_ai_order_strategy(
    api_key: str,
    context_data: dict,
    override_mode: str = "AUTO",
    override_reason: str = "",
    pair_name: str = None,
    portfolio_positions: list = None,
    weekly_dd_cap_percent: float = 2.0,
    risk_percent_per_trade: float = 2.0,
    max_positions_per_currency: int = 1,
    generation_policy: str = "AUTO_HIERARCHY",
):
    """
    generation_policy:
      - "AUTO_HIERARCHY"（推奨）: AI →（技術失敗時）AI再生成 →（さらに失敗時）数値フォールバック
      - "AI_STRICT": 従来通り。AIが壊れたら見送りで止める（安全最優先）
    """
    if _STRICT_GET_AI_ORDER_STRATEGY is None:
        return {"decision":"NO_TRADE","side":"NONE","entry":0,"take_profit":0,"stop_loss":0,"horizon":"WEEK","confidence":0.0,
                "why":"内部エラー: strict generator未定義","notes":["strict_generator_missing"],"market_regime":"DEFENSIVE","regime_why":"",
                "generator_path":"error"}

    # まずは従来の厳格生成（1回）
    strict_res = _STRICT_GET_AI_ORDER_STRATEGY(
        api_key=api_key,
        context_data=context_data,
        override_mode=override_mode,
        override_reason=override_reason,
        pair_name=pair_name,
        portfolio_positions=portfolio_positions,
        weekly_dd_cap_percent=weekly_dd_cap_percent,
        risk_percent_per_trade=risk_percent_per_trade,
        max_positions_per_currency=max_positions_per_currency,
    )
    if isinstance(strict_res, dict) and "generator_path" not in strict_res:
        strict_res["generator_path"] = "ai_strict"

    # AI厳格モードならここで終わり
    if str(generation_policy).upper() in ("AI_STRICT","STRICT","AI100"):
        return strict_res

    # 自動階層化：『技術失敗だけ』救済する
    if not _is_technical_failure_order(strict_res):
        return strict_res

    market_regime = strict_res.get("market_regime", "DEFENSIVE")
    regime_why = strict_res.get("regime_why", "")
    ctx = context_data or {}

    # quota系は再生成せず、数値フォールバックに直行
    if not _is_quota_error(strict_res):
        retry_res = _ai_retry_order(api_key, ctx, market_regime, regime_why, pair_name=pair_name or "")
        if isinstance(retry_res, dict) and retry_res.get("decision") == "TRADE":
            return retry_res
        # retryがNO_TRADEでも、技術失敗ではないならそれを返す
        if isinstance(retry_res, dict) and not _is_technical_failure_order(retry_res):
            return retry_res

    # 最終手段：数値フォールバック
    fb = _build_numeric_fallback_order(ctx, market_regime, regime_why, pair_name=pair_name or "")
    return fb
