# data_layer.py (Integrated v1)
from __future__ import annotations

import os
import time
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import requests

# Streamlit caching (optional when running outside Streamlit)
try:
    import streamlit as st
    _cache = st.cache_data
except Exception:
    def _cache(ttl: int = 0):
        def deco(fn):
            return fn
        return deco

# -------------------------
# Helpers
# -------------------------
DEFAULT_HEADERS = {"User-Agent": "fx-ev-engine/1.0"}

def _now_ts() -> float:
    return time.time()

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _env_or(keys: Dict[str, str], name: str, default: str = "") -> str:
    """
    Read from (1) keys dict passed from main, (2) Streamlit secrets if available, (3) env vars.
    """
    v = ""
    try:
        v = (keys or {}).get(name, "") or ""
    except Exception:
        v = ""
    if not v:
        try:
            import streamlit as st  # type: ignore
            v = str(st.secrets.get(name, "") or "")
        except Exception:
            pass
    if not v:
        v = os.getenv(name, default) or default
    return str(v).strip()

def _env_any(keys: Dict[str, str], names: List[str], default: str = "") -> Tuple[str, Optional[str]]:
    """
    Try multiple key names. Returns (value, used_name).
    """
    for n in names:
        v = _env_or(keys, n, "")
        if v:
            return v, n
    return default, None

def _http_get_json(url: str, params: Dict[str, Any] | None = None, timeout: Any = 12) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}:{r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"

def _http_get_json_retry(
    url: str,
    params: Dict[str, Any] | None = None,
    timeout: Any = (8, 25),
    retries: int = 2,
    backoff_s: float = 1.2,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retry GET JSON with exponential backoff. Returns (json, err_str)."""
    last_err: Optional[str] = None
    for i in range(retries + 1):
        j, err = _http_get_json(url, params=params, timeout=timeout)
        if not err:
            return j, None
        last_err = err
        # 429 or transient network issues -> retry
        if i < retries:
            sleep = backoff_s * (2 ** i) + random.random() * 0.25
            time.sleep(min(6.0, sleep))
    return None, last_err


# -------------------------
# Throttles (to avoid 429)
# -------------------------
_GDELT_LAST_TS = 0.0

def _gdelt_throttle(min_interval_s: float = 5.2) -> None:
    global _GDELT_LAST_TS
    now = time.time()
    wait = (_GDELT_LAST_TS + float(min_interval_s)) - now
    if wait > 0:
        time.sleep(wait)
    _GDELT_LAST_TS = time.time()

def _http_get_text(url: str, params: Dict[str, Any] | None = None, timeout: int = 12) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}:{r.text[:200]}"
        return r.text, None
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"

# -------------------------
# FRED
# -------------------------
@_cache(ttl=60*60)
def fetch_fred_series(series_id: str, api_key: str, limit: int = 10) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    Returns list of (date, value) ascending. If api_key missing, returns empty.
    """
    meta: Dict[str, Any] = {"ok": False, "source": "FRED", "series_id": series_id, "error": None}
    if not api_key:
        meta["error"] = "missing_api_key"
        return [], meta
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": int(limit),
    }
    j, err = _http_get_json(url, params=params, timeout=12)
    if err:
        meta["error"] = err
        return [], meta
    obs = (j or {}).get("observations", []) if isinstance(j, dict) else []
    out: List[Tuple[str, float]] = []
    for o in reversed(obs):
        d = o.get("date")
        v = o.get("value")
        if d is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        out.append((str(d), fv))
    meta["ok"] = True
    meta["n"] = len(out)
    return out, meta

@_cache(ttl=60*60)
def fetch_fred_latest(series_id: str, api_key: str) -> Tuple[Optional[float], Dict[str, Any]]:
    vals, meta = fetch_fred_series(series_id, api_key, limit=2)
    if not vals:
        return None, meta
    return float(vals[-1][1]), meta

# -------------------------
# TradingEconomics (optional)
# Note: TE has multiple auth styles. We support the simplest:
# - If TRADING_ECONOMICS_KEY is present, we call as ?c=<key>
# - This should be a "guest:guest" or paid key.
# -------------------------
@_cache(ttl=60*30)
def fetch_te_calendar(country: str, api_key: str, limit: int = 50) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = {"ok": False, "source": "TradingEconomics", "error": None}
    if not api_key:
        meta["error"] = "missing_api_key"
        return [], meta
    # Calendar endpoint (JSON). Country can be like 'united states' or 'japan'
    url = f"https://api.tradingeconomics.com/calendar/country/{country}"
    params = {"c": api_key, "format": "json"}
    j, err = _http_get_json(url, params=params, timeout=15)
    if err:
        meta["error"] = err
        return [], meta
    if not isinstance(j, list):
        meta["error"] = "unexpected_response"
        return [], meta
    # keep most recent items
    meta["ok"] = True
    return j[:limit], meta

def _calc_surprise_from_te(events: List[Dict[str, Any]], indicator_contains: str) -> Optional[float]:
    """
    Find latest event whose Event/Indicator contains substring and compute (Actual - Forecast) / abs(Forecast).
    """
    needle = (indicator_contains or "").lower()
    for e in events:
        name = str(e.get("Event") or e.get("Category") or e.get("Indicator") or "").lower()
        if needle and needle not in name:
            continue
        actual = e.get("Actual")
        forecast = e.get("Forecast")
        a = _safe_float(actual, None)
        f = _safe_float(forecast, None)
        if a is None or f is None or f == 0:
            continue
        return float((a - f) / abs(f))
    return None

# -------------------------
# GDELT (free)
# We'll query 2.1 doc API "doc" for counts via timeline search.
# -------------------------
@_cache(ttl=60*10)
def gdelt_doc_count(query: str, mode: str = "artlist", timespan: str = "1d") -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Returns integer count estimate for query over last timespan (e.g. 12h, 1d, 7d).

    Notes:
    - Streamlit CloudなどでGDELT応答が遅い/不安定な場合があるため、
      まず指定timespanで取得し、ReadTimeout系なら短いtimespan(12h)へフォールバックします。
    - 12hフォールバック時は 1d相当にスケール（×2）して返します（推定）。
    """
    meta: Dict[str, Any] = {
        "ok": False,
        "source": "GDELT",
        "error": None,
        "query": query,
        "timespan": timespan,
        "used_timespan": timespan,
        "scaled": False,
    }

    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    def _fetch(ts: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        params = {
            "query": query,
            "mode": "timelinevolraw",
            "format": "json",
            "timelinesmooth": "0",
            # timeline系はmaxrecordsが効かない場合もあるが、過大にならないよう保守的に
            "maxrecords": "120",
            "timespan": ts,
        }
        # GDELTは429もあるので軽くスロットル
        _gdelt_throttle(min_interval_s=5.2)
        # 読み取りが遅い環境対策：readを長めに
        return _http_get_json_retry(base, params=params, timeout=(8, 75), retries=2, backoff_s=1.6)

    j, err = _fetch(timespan)
    scale = 1.0

    # フォールバック：ReadTimeout/Timeoutなら短いtimespanで再試行
    if err and str(timespan).lower() in ("1d", "24h") and ("ReadTimeout" in err or "Timeout" in err):
        j2, err2 = _fetch("12h")
        if not err2:
            j, err = j2, None
            meta["used_timespan"] = "12h"
            meta["scaled"] = True
            scale = 2.0
        else:
            # 併記して返す（診断用）
            meta["error"] = (str(err) + " | fallback12h:" + str(err2))[:380]
            return None, meta

    if err:
        meta["error"] = str(err)[:380]
        return None, meta

    try:
        timeline = (j or {}).get("timeline", []) if isinstance(j, dict) else []
        if not timeline:
            meta["ok"] = True
            return 0, meta

        s = int(sum(int(t.get("value", 0)) for t in timeline))
        if scale != 1.0:
            s = int(round(float(s) * float(scale)))

        meta["ok"] = True
        return s, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return None, meta

def newsapi_headlines(query: str, api_key: str, page_size: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = {"ok": False, "source": "NewsAPI", "error": None}
    if not api_key:
        meta["error"] = "missing_api_key"
        return [], meta
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": int(page_size), "sortBy": "publishedAt", "language": "en"}
    try:
        r = requests.get(url, params=params, headers={"X-Api-Key": api_key, **DEFAULT_HEADERS}, timeout=15)
        if r.status_code >= 400:
            meta["error"] = f"http_{r.status_code}:{r.text[:200]}"
            return [], meta
        j = r.json()
        arts = j.get("articles", []) if isinstance(j, dict) else []
        meta["ok"] = True
        return arts, meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return [], meta

def simple_sentiment_from_articles(articles: List[Dict[str, Any]]) -> float:
    """
    Very conservative sentiment in [-1,1] based on headline keywords.
    """
    if not articles:
        return 0.0
    score = 0.0
    n = 0
    for a in articles[:30]:
        txt = (a.get("title") or "") + " " + (a.get("description") or "")
        t = txt.lower()
        s = 0.0
        for w in POS_WORDS:
            if w in t:
                s += 1.0
        for w in NEG_WORDS:
            if w in t:
                s -= 1.2
        if s != 0.0:
            score += s
            n += 1
    if n == 0:
        return 0.0
    # squash
    score = score / max(n, 1)
    return float(max(-1.0, min(1.0, score / 5.0)))

# -------------------------

# -------------------------
# OpenAI (optional)
# -------------------------
@_cache(ttl=60*30)
def sentiment_from_news(articles: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """Return (sentiment_score, meta). Safe and cheap."""
    try:
        n = len(articles or [])
        s = float(simple_sentiment_from_articles(articles or []))
        return s, {"ok": True, "n": n}
    except Exception as e:
        return 0.0, {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}

def openai_geo_score(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    If key provided, asks OpenAI Responses API to return JSON with risk scores.
    Falls back to empty dict. Never raises.
    """
    meta = {"ok": False, "source": "OpenAI", "error": None, "model": model}
    if not api_key:
        meta["error"] = "missing_api_key"
        return {}, meta

    url = "https://api.openai.com/v1/responses"
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {api_key}",
    }

    # Ask for strict JSON (best-effort; we still robust-parse).
    instructions = (
        "You are a geopolitics & macro risk analyst for FX. "
        "Return ONLY valid JSON, no markdown, no commentary. "
        "Schema: {"
        "\n  \"war_probability\": number (0..1),"
        "\n  \"financial_crisis_probability\": number (0..1),"
        "\n  \"pandemic_risk\": number (0..1),"
        "\n  \"policy_shift_risk\": number (0..1),"
        "\n  \"geopolitical_stress\": number (0..1),"
        "\n  \"summary\": string (<=200 chars)"
        "\n}"
    )

    payload = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "temperature": 0.2,
        "max_output_tokens": 500,
        "text": {"format": {"type": "json_object"}},
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=25)
        if r.status_code >= 400:
            meta["error"] = f"http_{r.status_code}:{r.text[:200]}"
            return {}, meta
        j = r.json()

        # Prefer output_text if present
        text = ""
        if isinstance(j, dict) and isinstance(j.get("output_text"), str):
            text = j["output_text"]
        else:
            # Fallback: walk output array
            try:
                out = j.get("output") or []
                parts = []
                for item in out:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content") or []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                            t = c.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                text = "\n".join(parts)
            except Exception:
                text = ""

        js = None
        t = (text or "").strip()
        if t:
            # direct parse
            try:
                js = json.loads(t)
            except Exception:
                # try extract first JSON object inside
                if "{" in t and "}" in t:
                    start = t.find("{")
                    end = t.rfind("}")
                    blob = t[start:end+1]
                    try:
                        js = json.loads(blob)
                    except Exception:
                        js = None

        meta["ok"] = True
        return (js or {}), meta
    except Exception as e:
        meta["error"] = f"{type(e).__name__}:{e}"
        return {}, meta
# -------------------------
# Integrated feature fetch
# -------------------------
def _pair_to_geo_query(pair_label: str) -> str:
    # simple mapping to relevant countries
    pl = (pair_label or "").upper()
    if "JPY" in pl:
        return "Japan OR Yen OR JPY"
    if "EUR" in pl:
        return "Europe OR Euro OR EUR"
    if "GBP" in pl:
        return "UK OR Britain OR GBP"
    if "AUD" in pl:
        return "Australia OR AUD"
    return "FX OR currency"

@_cache(ttl=60*15)
def fetch_external_features(pair_label: str, keys: Dict[str, str] | None = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns (features, meta). Never raises.
    This function is pair-agnostic for global risk sources to avoid rate limits in multi-pair scans.
    """
    keys = keys or {}

    # ---- Defaults (never missing) ----
    feats: Dict[str, float] = {
        "news_sentiment": 0.0,            # [-1,1]
        "cpi_surprise": 0.0,              # [-1,1]
        "nfp_surprise": 0.0,              # [-1,1]
        "rate_diff_change": 0.0,          # [-1,1] proxy
        "macro_risk_score": 0.0,          # [0,1]
        "global_risk_index": 0.0,         # [0,1]
        "war_probability": 0.0,           # [0,1]
        "financial_stress": 0.0,          # [0,1]
        "gdelt_war_count_1d": 0.0,
        "gdelt_finance_count_1d": 0.0,
        "vix": float("nan"),
        "dxy": float("nan"),
        "us10y": float("nan"),
        "jp10y": float("nan"),
    }

    fred_key, fred_used = _env_any(keys, ["FRED_API_KEY", "FRED_KEY", "FREDAPI_KEY"])
    te_key, te_used = _env_any(keys, ["TRADING_ECONOMICS_KEY", "TE_API_KEY", "TRADING_ECONOMICS_API_KEY"])
    news_key, news_used = _env_any(keys, ["NEWSAPI_KEY", "NEWS_API_KEY", "NEWSAPI_API_KEY"])
    openai_key, openai_used = _env_any(keys, ["OPENAI_API_KEY", "OPENAI_KEY"])

    meta: Dict[str, Any] = {"ok": True, "parts": {}}
    meta["parts"]["keys"] = {
        "ok": True,
        "detail": {
            "FRED": {"present": bool(fred_key), "used": fred_used},
            "TradingEconomics": {"present": bool(te_key), "used": te_used},
            "NewsAPI": {"present": bool(news_key), "used": news_used},
            "OpenAI": {"present": bool(openai_key), "used": openai_used},
        },
    }

    # ---- FRED macro ----
    try:
        vix, m_vix = fetch_fred_latest("VIXCLS", fred_key)
        dxy, m_dxy = fetch_fred_latest("DTWEXBGS", fred_key)  # broad dollar index
        us10y, m_us10 = fetch_fred_latest("DGS10", fred_key)
        jp10y, m_jp10 = fetch_fred_latest("IRLTLT01JPM156N", fred_key)  # OECD LT rate JP (monthly)
        meta["parts"]["fred"] = {"ok": True, "detail": {"vix": m_vix, "dxy": m_dxy, "us10y": m_us10, "jp10y": m_jp10}}

        if vix is not None: feats["vix"] = float(vix)
        if dxy is not None: feats["dxy"] = float(dxy)
        if us10y is not None: feats["us10y"] = float(us10y)
        if jp10y is not None: feats["jp10y"] = float(jp10y)

        # macro risk: map VIX 15->0, 40->1
        if vix is not None:
            feats["macro_risk_score"] = float(max(0.0, min(1.0, (float(vix) - 15.0) / 25.0)))

        # rate diff proxy
        if us10y is not None and jp10y is not None:
            rd = float(us10y - jp10y)
            feats["rate_diff_change"] = float(max(-1.0, min(1.0, rd / 10.0)))
    except Exception as e:
        meta["parts"]["fred"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}

    # ---- TradingEconomics surprises (optional) ----
    try:
        if te_key:
            ev_us, m_te = fetch_te_calendar("united states", te_key, limit=80)
            cpi = _calc_surprise_from_te(ev_us, "cpi")
            nfp = _calc_surprise_from_te(ev_us, "non-farm") or _calc_surprise_from_te(ev_us, "employment")
            if cpi is not None:
                feats["cpi_surprise"] = float(max(-1.0, min(1.0, cpi)))
            if nfp is not None:
                feats["nfp_surprise"] = float(max(-1.0, min(1.0, nfp)))
            meta["parts"]["te"] = m_te if isinstance(m_te, dict) else {"ok": True}
        else:
            meta["parts"]["te"] = {"ok": False, "error": "missing_api_key"}
    except Exception as e:
        meta["parts"]["te"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}

    # ---- GDELT counts (global; cached) ----
    try:
        war_q = '(war OR invasion OR missile OR attack OR sanction OR "state of emergency" OR mobilization)'
        fin_q = '(bank OR default OR crisis OR recession OR bailout OR "credit crunch" OR "liquidity crunch")'
        war_cnt, m_war = gdelt_doc_count(war_q, timespan="1d")
        fin_cnt, m_fin = gdelt_doc_count(fin_q, timespan="1d")
        gdelt_ok = bool((m_war or {}).get("ok")) and bool((m_fin or {}).get("ok"))
        gdelt_errs = []
        if (m_war or {}).get("error"): gdelt_errs.append(f"war:{(m_war or {}).get('error')}")
        if (m_fin or {}).get("error"): gdelt_errs.append(f"finance:{(m_fin or {}).get('error')}")
        meta["parts"]["gdelt"] = {"ok": gdelt_ok, "error": "; ".join(gdelt_errs)[:240] if gdelt_errs else None, "detail": {"war": m_war, "finance": m_fin}}

        war_cnt_i = int(war_cnt or 0)
        fin_cnt_i = int(fin_cnt or 0)
        feats["gdelt_war_count_1d"] = float(war_cnt_i)
        feats["gdelt_finance_count_1d"] = float(fin_cnt_i)

        war_prob = float(max(0.0, min(1.0, math.log1p(war_cnt_i) / 8.0)))
        fin_stress = float(max(0.0, min(1.0, math.log1p(fin_cnt_i) / 9.0)))
        feats["war_probability"] = war_prob
        feats["financial_stress"] = fin_stress
        feats["global_risk_index"] = float(max(0.0, min(1.0, 0.55 * war_prob + 0.45 * fin_stress)))
    except Exception as e:
        meta["parts"]["gdelt"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}

    # ---- NewsAPI sentiment (optional) ----
    try:
        if news_key:
            geo_scope = "forex OR (USDJPY OR EURUSD OR GBPUSD OR AUDUSD)"
            arts, m_news = newsapi_headlines(f"{geo_scope} AND (central bank OR inflation OR war OR crisis)", news_key, page_size=30)
            news_sent, s_meta = sentiment_from_news(arts)
            feats["news_sentiment"] = float(max(-1.0, min(1.0, news_sent)))
            meta["parts"]["newsapi"] = {"ok": True, "detail": {"newsapi": m_news, "sentiment": s_meta}}
        else:
            meta["parts"]["newsapi"] = {"ok": False, "error": "missing_api_key"}
    except Exception as e:
        meta["parts"]["newsapi"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}

    # ---- OpenAI risk overlay (optional) ----
    # Purpose: produce conservative global_risk_index override using headlines + gdelt counts + vix.
    try:
        if openai_key:
            ov, m_ov = openai_risk_overlay(openai_key, feats, meta)
            if isinstance(ov, dict):
                # Override if provided
                if "global_risk_index" in ov:
                    feats["global_risk_index"] = float(max(0.0, min(1.0, float(ov["global_risk_index"]))))
                if "war_probability" in ov:
                    feats["war_probability"] = float(max(0.0, min(1.0, float(ov["war_probability"]))))
                if "financial_stress" in ov:
                    feats["financial_stress"] = float(max(0.0, min(1.0, float(ov["financial_stress"]))))
            meta["parts"]["openai"] = m_ov if isinstance(m_ov, dict) else {"ok": True}
        else:
            meta["parts"]["openai"] = {"ok": False, "error": "missing_api_key"}
    except Exception as e:
        meta["parts"]["openai"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}


    # ---- Fail-safe guard (never leave risk scores pinned at 0 when externals are degraded) ----
    try:
        macro = float(max(0.0, min(1.0, _safe_float(feats.get("macro_risk_score"), 0.0))))
        news_mag = float(min(1.0, abs(_safe_float(feats.get("news_sentiment"), 0.0))))
        # If both GDELT and OpenAI are not OK, raise conservative baseline from macro/news.
        gdelt_ok = bool((meta.get("parts", {}).get("gdelt", {}) or {}).get("ok"))
        openai_ok = bool((meta.get("parts", {}).get("openai", {}) or {}).get("ok"))
        used = False

        if not gdelt_ok:
            # war/finance counts missing -> avoid false sense of safety
            if _safe_float(feats.get("war_probability"), 0.0) <= 0.0:
                feats["war_probability"] = float(max(0.0, min(1.0, 0.20 * macro + 0.10 * news_mag)))
                used = True
            if _safe_float(feats.get("financial_stress"), 0.0) <= 0.0:
                feats["financial_stress"] = float(max(0.0, min(1.0, 0.30 * macro + 0.10 * news_mag)))
                used = True

        if (not gdelt_ok) and (not openai_ok):
            base = float(max(0.0, min(1.0, 0.45 * macro + 0.15 * news_mag)))
            if _safe_float(feats.get("global_risk_index"), 0.0) <= 0.0:
                feats["global_risk_index"] = base
                used = True

        if used:
            meta["parts"]["failsafe"] = {"ok": True, "detail": {"macro": macro, "news_mag": news_mag}}
    except Exception as e:
        meta["parts"]["failsafe"] = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e)}
    # ---- Panel values (for Status/CSV visibility) ----
    try:
        meta["parts"]["risk_values"] = {
            "ok": True,
            "detail": {
                "global_risk_index": round(_safe_float(feats.get("global_risk_index"), 0.0), 3),
                "war_probability": round(_safe_float(feats.get("war_probability"), 0.0), 3),
                "financial_stress": round(_safe_float(feats.get("financial_stress"), 0.0), 3),
                "macro_risk_score": round(_safe_float(feats.get("macro_risk_score"), 0.0), 3),
                "news_sentiment": round(_safe_float(feats.get("news_sentiment"), 0.0), 3),
                "gdelt_war_count_1d": int(_safe_float(feats.get("gdelt_war_count_1d"), 0.0)),
                "gdelt_finance_count_1d": int(_safe_float(feats.get("gdelt_finance_count_1d"), 0.0)),
            },
        }
    except Exception:
        pass



    return feats, meta


def openai_risk_overlay(openai_key: str, feats: Dict[str, float], meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls OpenAI to estimate conservative risk scores from already-fetched numbers.
    Robust against API parameter differences (Responses vs ChatCompletions) and 400s.
    Returns (overlay_dict, meta_dict).
    """
    def _num(x: Any) -> Any:
        try:
            if x is None:
                return None
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    model = os.getenv("OPENAI_RISK_MODEL", "gpt-4o-mini")

    payload = {
        "vix": _num(feats.get("vix")),
        "macro_risk_score": _num(feats.get("macro_risk_score")),
        "gdelt_war_count_1d": _num(feats.get("gdelt_war_count_1d")),
        "gdelt_finance_count_1d": _num(feats.get("gdelt_finance_count_1d")),
        "news_sentiment": _num(feats.get("news_sentiment")),
    }

    prompt = (
        "You are a risk overlay engine for FX trading. "
        "Given the inputs, output STRICT JSON only (no prose) with keys: "
        "global_risk_index (0-1), war_probability (0-1), financial_stress (0-1). "
        "Be conservative: if inputs are missing/uncertain, return higher risk. "
        f"inputs={json.dumps(payload, ensure_ascii=False)}"
    )

    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

    def _extract_json_text(resp_json: Dict[str, Any]) -> str:
        # Responses API shape
        out_txt = ""
        try:
            for item in (resp_json or {}).get("output", []) or []:
                for c in item.get("content", []) or []:
                    t = c.get("type")
                    if t in ("output_text", "text"):
                        out_txt += c.get("text", "") or ""
        except Exception:
            pass
        if not out_txt:
            out_txt = (resp_json or {}).get("output_text", "") or ""
        return out_txt.strip()

    # --- 1) Try Responses API (preferred) with two format styles ---
    try:
        url = "https://api.openai.com/v1/responses"

        bodies = [
            # Newer style
            {"model": model, "input": prompt, "text": {"format": {"type": "json_object"}}, "temperature": 0},
            # Older/alt style sometimes accepted
            {"model": model, "input": prompt, "response_format": {"type": "json_object"}, "temperature": 0},
        ]

        last_detail = ""
        for b in bodies:
            r = requests.post(url, headers=headers, json=b, timeout=25)
            if r.status_code == 200:
                j = r.json()
                out_txt = _extract_json_text(j)
                ov = json.loads(out_txt) if out_txt else {}
                ov = ov if isinstance(ov, dict) else {}
                detail = {k: ov.get(k) for k in ("global_risk_index","war_probability","financial_stress") if k in ov}
                return ov, {"ok": True, "provider": "responses", "model": model, "detail": detail, "overlay": ov}
            last_detail = r.text[:800]
            # non-400 might be auth/perm -> no point retrying here
            if r.status_code in (401, 403, 404):
                return {}, {"ok": False, "provider": "responses", "model": model, "error": f"http_{r.status_code}", "detail": last_detail}
        # if we reach here, likely 400s due to schema mismatch -> fall through to chat completions
    except Exception as e:
        last_detail = f"{type(e).__name__}:{e}"

    # --- 2) Fallback: Chat Completions ---
    try:
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, json=body, timeout=25)
        if r.status_code != 200:
            return {}, {"ok": False, "provider": "chat_completions", "model": model, "error": f"http_{r.status_code}", "detail": r.text[:800]}
        j = r.json()
        out_txt = (((j.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        ov = json.loads(out_txt) if out_txt else {}
        ov = ov if isinstance(ov, dict) else {}
        detail = {k: ov.get(k) for k in ("global_risk_index","war_probability","financial_stress") if k in ov}
        return ov, {"ok": True, "provider": "chat_completions", "model": model, "detail": detail, "overlay": ov}
    except Exception as e:
        return {}, {"ok": False, "provider": "chat_completions", "model": model, "error": f"{type(e).__name__}", "detail": str(e)}
