# data_layer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import os
import time
import math
import requests

# -------------------------
# TTL Cache
# -------------------------
@dataclass
class _CacheEntry:
    expire: float
    value: Any
    meta: Dict[str, Any]

class TTLCache:
    def __init__(self):
        self._d: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[_CacheEntry]:
        e = self._d.get(key)
        if not e:
            return None
        if time.time() <= e.expire:
            return e
        self._d.pop(key, None)
        return None

    def set(self, key: str, value: Any, ttl_sec: int, meta: Optional[Dict[str, Any]] = None):
        self._d[key] = _CacheEntry(expire=time.time() + float(ttl_sec), value=value, meta=meta or {})

_CACHE = TTLCache()

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return None
    s = str(v).strip()
    if s == "":
        return None
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def _http_get(url: str, params: Optional[dict] = None, timeout: int = 12, retries: int = 2) -> Tuple[Optional[requests.Response], Dict[str, Any]]:
    meta: Dict[str, Any] = {"url": url, "ok": False, "status": None, "error": None}
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            meta["status"] = getattr(r, "status_code", None)
            if r.status_code == 200:
                meta["ok"] = True
                return r, meta
            last_err = f"HTTP_{r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}:{e}"
        time.sleep(0.4 * (i + 1))
    meta["error"] = last_err
    return None, meta

# -------------------------
# News: GDELT DOC API (tone)
# -------------------------
_GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

_PAIR_NEWS_QUERY = {
    "USD/JPY": '(USDJPY OR "USD JPY" OR "dollar yen" OR BOJ OR "Bank of Japan" OR Fed OR "US yields")',
    "EUR/USD": '(EURUSD OR "EUR USD" OR "euro dollar" OR ECB OR Fed OR "US yields")',
    "GBP/USD": '(GBPUSD OR "GBP USD" OR "pound dollar" OR BoE OR Fed OR "US yields")',
    "AUD/USD": '(AUDUSD OR "AUD USD" OR "aussie dollar" OR RBA OR Fed OR "China data")',
    "EUR/JPY": '(EURJPY OR "EUR JPY" OR ECB OR BOJ OR "Bank of Japan")',
    "GBP/JPY": '(GBPJPY OR "GBP JPY" OR BoE OR BOJ OR "Bank of Japan")',
    "AUD/JPY": '(AUDJPY OR "AUD JPY" OR RBA OR BOJ OR "risk sentiment")',
}

def fetch_news_sentiment_gdelt(pair_label: str, timespan: str = "3d", ttl_sec: int = 30 * 60) -> Tuple[float, Dict[str, Any]]:
    pair_key = (pair_label or "").split()[0].strip()
    q = _PAIR_NEWS_QUERY.get(pair_key, f'("{pair_key}" OR FX OR forex)')
    cache_key = f"gdelt_tone::{pair_key}::{timespan}"
    ce = _CACHE.get(cache_key)
    if ce:
        return float(ce.value), dict(ce.meta)

    params = {"query": q, "mode": "timelinetone", "format": "json", "timespan": timespan}
    r, meta = _http_get(_GDELT_DOC_ENDPOINT, params=params, timeout=14, retries=2)
    tone = 0.0
    if r is None or not meta.get("ok"):
        out_meta = {"source": "GDELT", **meta, "tone": None, "sentiment": 0.0}
        _CACHE.set(cache_key, 0.0, ttl_sec, out_meta)
        return 0.0, out_meta

    try:
        js = r.json()
        tl = js.get("timeline") or js.get("data") or []
        tones: List[float] = []
        weights: List[float] = []
        for row in tl:
            t = _to_float(row.get("tone") if isinstance(row, dict) else None)
            v = _to_float(row.get("value") if isinstance(row, dict) else None)
            if t is None:
                continue
            tones.append(float(t))
            weights.append(float(v) if (v is not None and v > 0) else 1.0)
        if tones:
            sw = sum(weights) if sum(weights) > 0 else float(len(weights))
            tone = sum(t*w for t, w in zip(tones, weights)) / sw
    except Exception as e:
        meta["ok"] = False
        meta["error"] = f"parse_fail:{type(e).__name__}"
        out_meta = {"source": "GDELT", **meta, "tone": None, "sentiment": 0.0}
        _CACHE.set(cache_key, 0.0, ttl_sec, out_meta)
        return 0.0, out_meta

    sentiment = _clamp(float(tone) / 10.0, -1.0, 1.0)
    out_meta = {"source": "GDELT", **meta, "tone": float(tone), "sentiment": float(sentiment)}
    _CACHE.set(cache_key, float(sentiment), ttl_sec, out_meta)
    return float(sentiment), out_meta

# -------------------------
# Economic calendar: TradingEconomics
# -------------------------
_TE_CAL_ENDPOINT = "https://api.tradingeconomics.com/calendar"

def _te_key_from_any(keys: Dict[str, str]) -> str:
    k = (keys or {}).get("TRADING_ECONOMICS_KEY") or os.getenv("TRADING_ECONOMICS_KEY", "")
    return str(k).strip()

def fetch_te_surprises(keys: Dict[str, str], lookback_days: int = 14, ttl_sec: int = 30 * 60) -> Tuple[Dict[str, float], Dict[str, Any]]:
    te_key = _te_key_from_any(keys)
    cache_key = f"te_surprises::{te_key[-6:]}::{lookback_days}"
    ce = _CACHE.get(cache_key)
    if ce:
        return dict(ce.value), dict(ce.meta)

    if not te_key:
        out = {"cpi_surprise": 0.0, "nfp_surprise": 0.0}
        meta = {"source": "TradingEconomics", "ok": False, "error": "missing_key"}
        _CACHE.set(cache_key, out, ttl_sec, meta)
        return out, meta

    params = {"c": te_key, "f": "json"}
    r, meta = _http_get(_TE_CAL_ENDPOINT, params=params, timeout=18, retries=2)
    if r is None or not meta.get("ok"):
        out = {"cpi_surprise": 0.0, "nfp_surprise": 0.0}
        out_meta = {"source": "TradingEconomics", **meta}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta

    try:
        data = r.json()
        now = time.time()
        cutoff = now - lookback_days * 86400

        from datetime import datetime
        def _ts(s: Any) -> Optional[float]:
            if not s:
                return None
            try:
                dt = datetime.fromisoformat(str(s).replace("Z",""))
                return dt.timestamp()
            except Exception:
                return None

        cpi_candidates = []
        nfp_candidates = []
        for row in data if isinstance(data, list) else []:
            ev = str(row.get("Event","") or "")
            t = _ts(row.get("Date"))
            if t is not None and t < cutoff:
                continue
            if "CPI" in ev.upper():
                cpi_candidates.append(row)
            if ("NON FARM" in ev.upper()) or ("NONFARM" in ev.upper()) or ("PAYROLL" in ev.upper()):
                nfp_candidates.append(row)

        def _best_surprise(rows: list) -> float:
            best = None
            best_ts = -1.0
            for row in rows:
                a = _to_float(row.get("Actual"))
                f = _to_float(row.get("Forecast"))
                t = _ts(row.get("Date")) or 0.0
                if a is None or f is None:
                    continue
                if t > best_ts:
                    best_ts = t
                    best = (a, f)
            if not best:
                return 0.0
            a, f = best
            denom = abs(f) if abs(f) > 1e-9 else 1.0
            s = (a - f) / denom * 10.0
            return float(_clamp(s, -10.0, 10.0))

        out = {"cpi_surprise": _best_surprise(cpi_candidates), "nfp_surprise": _best_surprise(nfp_candidates)}
        out_meta = {"source": "TradingEconomics", **meta, "items": len(data) if isinstance(data, list) else 0}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta
    except Exception as e:
        out = {"cpi_surprise": 0.0, "nfp_surprise": 0.0}
        out_meta = {"source": "TradingEconomics", **meta, "ok": False, "error": f"parse_fail:{type(e).__name__}"}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta

# -------------------------
# Rate diff: FRED
# -------------------------
_FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

def _fred_key_from_any(keys: Dict[str, str]) -> str:
    k = (keys or {}).get("FRED_API_KEY") or os.getenv("FRED_API_KEY", "")
    return str(k).strip()

def fetch_rate_diff_change_fred(keys: Dict[str, str], ttl_sec: int = 60 * 60) -> Tuple[float, Dict[str, Any]]:
    fred_key = _fred_key_from_any(keys)
    cache_key = f"fred_spread_change::{fred_key[-6:]}"
    ce = _CACHE.get(cache_key)
    if ce:
        return float(ce.value), dict(ce.meta)

    if not fred_key:
        meta = {"source": "FRED", "ok": False, "error": "missing_key"}
        _CACHE.set(cache_key, 0.0, ttl_sec, meta)
        return 0.0, meta

    def _fetch_series(series_id: str) -> Tuple[List[float], Dict[str, Any]]:
        params = {"series_id": series_id, "api_key": fred_key, "file_type": "json", "sort_order": "desc", "limit": 60}
        r, meta = _http_get(_FRED_ENDPOINT, params=params, timeout=16, retries=2)
        if r is None or not meta.get("ok"):
            return [], meta
        try:
            js = r.json()
            obs = js.get("observations") or []
            vals=[]
            for o in obs:
                v = _to_float(o.get("value"))
                if v is None:
                    continue
                vals.append(float(v))
            return vals, meta
        except Exception as e:
            meta["ok"]=False
            meta["error"]=f"parse_fail:{type(e).__name__}"
            return [], meta

    us_vals, us_meta = _fetch_series("DGS10")
    jp_vals, jp_meta = _fetch_series("IRLTLT01JPM156N")

    if (not us_vals) or (not jp_vals):
        meta = {"source":"FRED","ok":False,"error":"series_empty","us":us_meta,"jp":jp_meta}
        _CACHE.set(cache_key, 0.0, ttl_sec, meta)
        return 0.0, meta

    try:
        spread_now = float(us_vals[0]) - float(jp_vals[0])
        idx = 5 if len(us_vals) > 6 and len(jp_vals) > 6 else min(len(us_vals), len(jp_vals)) - 1
        spread_prev = float(us_vals[idx]) - float(jp_vals[idx])
        change = spread_now - spread_prev
    except Exception as e:
        meta = {"source":"FRED","ok":False,"error":f"calc_fail:{type(e).__name__}"}
        _CACHE.set(cache_key, 0.0, ttl_sec, meta)
        return 0.0, meta

    change = float(_clamp(change, -2.0, 2.0))
    meta = {"source":"FRED","ok":True,"spread_change":change,"us_meta":us_meta,"jp_meta":jp_meta}
    _CACHE.set(cache_key, change, ttl_sec, meta)
    return change, meta

# -------------------------
# COT: CFTC FinFutWk
# -------------------------
_COT_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

_CFTC_MARKET_NAME_MAP = {
    "USD/JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "EUR/USD": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP/USD": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "AUD/USD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "EUR/JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "GBP/JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "AUD/JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
}

def fetch_cot_features(pair_label: str, ttl_sec: int = 6 * 60 * 60) -> Tuple[Dict[str, float], Dict[str, Any]]:
    pair_key = (pair_label or "").split()[0].strip()
    target = _CFTC_MARKET_NAME_MAP.get(pair_key, "")
    cache_key = f"cot::{pair_key}"
    ce = _CACHE.get(cache_key)
    if ce:
        return dict(ce.value), dict(ce.meta)

    r, meta = _http_get(_COT_URL, params=None, timeout=18, retries=2)
    if r is None or not meta.get("ok"):
        out = {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}
        out_meta = {"source":"CFTC","ok":False, **meta, "target": target}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta

    try:
        text = r.text
        lines = [ln.strip("\n\r") for ln in text.splitlines() if ln.strip()]
        cols = [c.strip() for c in lines[0].split(",")]
        col_idx = {c:i for i,c in enumerate(cols)}

        def idx(name: str) -> Optional[int]:
            return col_idx.get(name)

        i_market = idx("Market_and_Exchange_Names")
        i_oi = idx("Open_Interest_All")
        i_lev_long = idx("Lev_Money_Positions_Long_All")
        i_lev_short = idx("Lev_Money_Positions_Short_All")
        i_ast_long = idx("Asset_Mgr_Positions_Long_All")
        i_ast_short = idx("Asset_Mgr_Positions_Short_All")
        i_asof = idx("As_of_Date_In_Form_YYMMDD")

        if None in (i_market, i_oi, i_lev_long, i_lev_short, i_ast_long, i_ast_short, i_asof):
            out = {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}
            out_meta = {"source":"CFTC","ok":False, "error":"column_missing"}
            _CACHE.set(cache_key, out, ttl_sec, out_meta)
            return out, out_meta

        best_row = None
        best_asof = ""
        need_i = max(i_market, i_oi, i_lev_long, i_lev_short, i_ast_long, i_ast_short, i_asof)

        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) <= need_i:
                continue
            mkt = parts[i_market].strip('"')
            if target and mkt != target:
                continue
            asof = parts[i_asof].strip('"')
            if asof > best_asof:
                best_asof = asof
                best_row = parts

        if not best_row:
            out = {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}
            out_meta = {"source":"CFTC","ok":False, "error":"market_not_found", "target": target}
            _CACHE.set(cache_key, out, ttl_sec, out_meta)
            return out, out_meta

        oi = _to_float(best_row[i_oi]) or 0.0
        lev_net = (_to_float(best_row[i_lev_long]) or 0.0) - (_to_float(best_row[i_lev_short]) or 0.0)
        ast_net = (_to_float(best_row[i_ast_long]) or 0.0) - (_to_float(best_row[i_ast_short]) or 0.0)

        if oi <= 0:
            lev_pct = 0.0
            ast_pct = 0.0
        else:
            lev_pct = float(_clamp(lev_net / oi, -1.0, 1.0))
            ast_pct = float(_clamp(ast_net / oi, -1.0, 1.0))

        out = {"cot_leveraged_net_pctoi": lev_pct, "cot_asset_net_pctoi": ast_pct}
        out_meta = {"source":"CFTC","ok":True, "asof": best_asof, "target": target, "open_interest": oi}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta

    except Exception as e:
        out = {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}
        out_meta = {"source":"CFTC","ok":False, **meta, "error": f"parse_fail:{type(e).__name__}"}
        _CACHE.set(cache_key, out, ttl_sec, out_meta)
        return out, out_meta

# -------------------------
# ✅ Unified: fetch all (main.py が呼ぶ入口)
# -------------------------
def fetch_external_features(pair_label: str, keys: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    keys = keys or {}
    feats: Dict[str, float] = {}
    meta: Dict[str, Any] = {"ok": True, "errors": []}

    s_news, m_news = fetch_news_sentiment_gdelt(pair_label)
    feats["news_sentiment"] = float(s_news)
    meta["news"] = m_news
    if not m_news.get("ok", False):
        meta["ok"] = False
        meta["errors"].append(f"news:{m_news.get('error') or m_news.get('status')}")

    te_sur, m_te = fetch_te_surprises(keys)
    feats.update({k: float(v) for k, v in te_sur.items()})
    meta["economic"] = m_te
    if not m_te.get("ok", False):
        meta["ok"] = False
        meta["errors"].append(f"economic:{m_te.get('error') or m_te.get('status')}")

    spread_chg, m_sp = fetch_rate_diff_change_fred(keys)
    feats["rate_diff_change"] = float(spread_chg)
    meta["rate_diff"] = m_sp
    if not m_sp.get("ok", False):
        meta["ok"] = False
        meta["errors"].append(f"rate_diff:{m_sp.get('error') or m_sp.get('status')}")

    cot, m_cot = fetch_cot_features(pair_label)
    feats.update({k: float(v) for k, v in cot.items()})
    meta["cot"] = m_cot
    if not m_cot.get("ok", False):
        meta["ok"] = False
        meta["errors"].append(f"cot:{m_cot.get('error') or m_cot.get('status')}")

    return feats, meta
