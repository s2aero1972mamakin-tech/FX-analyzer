
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

    rng = get_ai_range(api_key, context_data)
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
    if "静観" in pm:
        reasons.append("panel_mid_says_wait")

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

    # --- AI予想レンジ（任意連携）---
    # main.py側で ai_range_* が ctx に入っていれば活用する
    ai_w = _safe_float(context_data.get("ai_range_width_pct"))  # 予想レンジ幅（%）
    ai_pos = _safe_float(context_data.get("ai_range_pos"))      # 予想レンジ内の位置（0〜1）
    if ai_w is not None:
        narrow_th = 0.55 if regime == "DEFENSIVE" else 0.35
        if ai_w < narrow_th:
            reasons.append("ai_range_too_narrow_range_market")
    if ai_pos is not None:
        mid_band = 0.18 if regime == "DEFENSIVE" else 0.12
        if abs(ai_pos - 0.5) < mid_band:
            reasons.append("ai_range_middle_no_edge")

    return (len(reasons) > 0), regime, reasons

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
    """AI予想レンジ（1週間の高値/安値）を取得して {low, high, why} を返す。
    返り値:
      - 成功: {"low": float, "high": float, "why": str}
      - 失敗: None
    """
    try:
        model = genai.GenerativeModel(get_active_model(api_key))
        p = float(context_data.get("price", 0.0) or 0.0)

        prompt = (
            "あなたはFXレンジ予測エンジンです。\n"
            f"現在のUSD/JPYは {p:.3f} です。\n"
            "今後1週間の想定レンジの『高値(high)』と『安値(low)』を、次のJSONだけで返してください。\n"
            "余計な文章は不要です。\n"
            "{\n"
            "  \"high\": 0.0,\n"
            "  \"low\": 0.0,\n"
            "  \"why\": \"短い理由（日本語）\"\n"
            "}\n"
        )
        res = (model.generate_content(prompt).text or "").strip()

        # JSON抽出（最初の {...}）
        m = re.search(r"\{[\s\S]*\}", res)
        if m:
            obj = json.loads(m.group(0))
            high = float(obj.get("high"))
            low = float(obj.get("low"))
            if low > high:
                low, high = high, low
            return {"low": low, "high": high, "why": str(obj.get("why", ""))}

        # 数字だけ返ってきた場合の救済（高値→安値の順を期待）
        nums = re.findall(r"\d+\.\d+|\d+", res)
        if len(nums) >= 2:
            high = float(nums[0]); low = float(nums[1])
            if low > high:
                low, high = high, low
            return {"low": low, "high": high, "why": "JSON以外で返答された場合の救済抽出"}

        return None
    except Exception:
        return None


