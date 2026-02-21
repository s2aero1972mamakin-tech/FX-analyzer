# data_layer.py (Integrated v1)
from __future__ import annotations

import os
import time
import math
import json
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
# --- simple per-process throttles (Streamlit Cloud friendly) ---
# NOTE: Streamlit Cloud runs multiple processes sometimes; this is best-effort.
_GDELT_LAST_CALL_TS = 0.0
_GDELT_BACKOFF_UNTIL_TS = 0.0

def _gdelt_throttle(min_interval_s: float = 1.2) -> None:
    """Best-effort throttle to reduce GDELT 429 during multi-pair scans."""
    global _GDELT_LAST_CALL_TS, _GDELT_BACKOFF_UNTIL_TS
    now = time.time()
    if now < _GDELT_BACKOFF_UNTIL_TS:
        time.sleep(max(0.0, _GDELT_BACKOFF_UNTIL_TS - now))
    wait = (_GDELT_LAST_CALL_TS + float(min_interval_s)) - time.time()
    if wait > 0:
        time.sleep(wait)
    _GDELT_LAST_CALL_TS = time.time()


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

def _http_get_json(url: str, params: Dict[str, Any] | None = None, timeout: int = 12) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}:{r.text[:200]}"
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"

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
@_cache(ttl=60*60)
# -------------------------
# GDELT (free)
# We'll query 2.1 doc API "doc" for counts via timeline search.
# -------------------------
@_cache(ttl=60*10)
def gdelt_doc_count(query: str, mode: str = "artlist", timespan: str = "1d") -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Returns integer count estimate for query over last timespan (e.g. 1d, 7d).

    We use `timelinevolraw` which returns series with counts.
    Handles 429 with backoff + single retry.
    """
    meta: Dict[str, Any] = {"ok": False, "source": "GDELT", "error": None, "query": query, "timespan": timespan}
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "timelinesmooth": "0",
        "maxrecords": "250",
        "timespan": timespan,
    }

    # Throttle before request (best-effort).
    _gdelt_throttle()

    for attempt in range(2):
        j, err = _http_get_json(base, params=params, timeout=15)
        if not err:
            try:
                timeline = (j or {}).get("timeline", [])
                if not timeline:
                    meta["ok"] = True
                    return 0, meta
                s = int(sum(int(t.get("value", 0)) for t in timeline))
                meta["ok"] = True
                return s, meta
            except Exception as e:
                meta["error"] = f"{type(e).__name__}:{e}"
                return None, meta

        # 429 backoff
        if isinstance(err, str) and err.startswith("http_429"):
            # global backoff window for this process
            global _GDELT_BACKOFF_UNTIL_TS
            backoff = 3.0 * (attempt + 1)
            _GDELT_BACKOFF_UNTIL_TS = max(_GDELT_BACKOFF_UNTIL_TS, time.time() + backoff)
            meta["error"] = err
            if attempt == 0:
                # wait and retry once
                _gdelt_throttle()
                continue

        meta["error"] = err
        return None, meta

    return None, meta

# -------------------------
# NewsAPI (optional)
# -------------------------
POS_WORDS = {
    "deal","agreement","ceasefire","peace","dovish","easing","recovery","growth","stability","calm"
}
NEG_WORDS = {
    "war","attack","missile","invasion","sanction","crisis","collapse","default","panic","bankrun",
    "pandemic","outbreak","lockdown","emergency","martial","terror","nuclear","escalation","shooting"
}

@_cache(ttl=60*15)
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

def sentiment_from_news(articles: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """Conservative sentiment + small diagnostics."""
    meta: Dict[str, Any] = {"ok": True, "n_articles": int(len(articles or []))}
    try:
        score = float(simple_sentiment_from_articles(articles))
        meta["score"] = score
        return score, meta
    except Exception as e:
        meta["ok"] = False
        meta["error"] = f"{type(e).__name__}:{e}"
        return 0.0, meta


# -------------------------

# -------------------------
# OpenAI (optional)
# -------------------------
@_cache(ttl=60*30)
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

# -------------------------
# Integrated feature fetch
# -------------------------
@_cache(ttl=60*15)
def _fetch_global_sources(keys: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Fetches global (pair-agnostic) features & status meta.
    Cached to avoid rate limits during multi-pair scans.
    """
    keys = keys or {}

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
            # Some free keys have country restrictions; try US then JP fallback.
            ev_us, m_te = fetch_te_calendar("united states", te_key, limit=80)
            if isinstance(m_te, dict) and isinstance(m_te.get("error"), str) and str(m_te["error"]).startswith("http_403"):
                ev_us, m_te = fetch_te_calendar("japan", te_key, limit=80)

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
        meta["parts"]["gdelt"] = {"ok": True, "detail": {"war": m_war, "finance": m_fin}}

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
    try:
        if openai_key:
            ov, m_ov = openai_risk_overlay(openai_key, feats, meta)
            if isinstance(ov, dict):
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

    return feats, meta


@_cache(ttl=60*15)
def fetch_external_features(pair_label: str, keys: Dict[str, str] | None = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns (features, meta). Never raises.

    NOTE:
    - Global sources are cached via `_fetch_global_sources` to avoid 429 in multi-pair scans.
    - `pair_label` is kept for backward compatibility and future pair-specific hooks.
    """
    keys = keys or {}
    feats, meta = _fetch_global_sources(keys)
    out_feats = dict(feats) if isinstance(feats, dict) else {}
    out_meta = dict(meta) if isinstance(meta, dict) else {"ok": True, "parts": {}}
    out_meta["pair_label"] = pair_label
    return out_feats, out_meta



def _extract_responses_output_text(j: Any) -> str:
    if not isinstance(j, dict):
        return ""
    if isinstance(j.get("output_text"), str):
        return j.get("output_text", "") or ""
    out_txt = ""
    try:
        for item in j.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for c in item.get("content", []) or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str):
                        out_txt += t
    except Exception:
        return ""
    return out_txt


def openai_risk_overlay(openai_key: str, feats: Dict[str, float], meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls OpenAI to estimate conservative risk scores from already-fetched numbers.

    Output schema:
      { "global_risk_index": 0..1, "war_probability": 0..1, "financial_stress": 0..1 }

    Never raises.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json", **DEFAULT_HEADERS}

    # Choose a widely available default model; allow override via env.
    model = os.getenv("OPENAI_RISK_MODEL", "gpt-4o-mini")

    payload = {
        "vix": feats.get("vix"),
        "macro_risk_score": feats.get("macro_risk_score"),
        "gdelt_war_count_1d": feats.get("gdelt_war_count_1d"),
        "gdelt_finance_count_1d": feats.get("gdelt_finance_count_1d"),
        "news_sentiment": feats.get("news_sentiment"),
    }

    prompt = (
        "You are a risk overlay engine for FX trading. "
        "Given the numeric inputs, output strict JSON ONLY (no markdown) with keys: "
        "global_risk_index (0..1), war_probability (0..1), financial_stress (0..1). "
        "Be conservative: if uncertain, return higher risk. "
        f"inputs={json.dumps(payload, ensure_ascii=False)}"
    )

    # Try the modern Responses schema first.
    body1 = {
        "model": model,
        "input": prompt,
        "temperature": 0.1,
        "max_output_tokens": 250,
        "text": {"format": {"type": "json_object"}},
    }

    # Fallback variant (some SDKs/docs used response_format).
    body2 = {
        "model": model,
        "input": prompt,
        "temperature": 0.1,
        "max_output_tokens": 250,
        "response_format": {"type": "json_object"},
    }

    for body in (body1, body2):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=25)
            if r.status_code != 200:
                # If first attempt fails with 400, we retry with the fallback body.
                err_meta = {"ok": False, "error": f"http_{r.status_code}", "detail": r.text[:500], "model": model}
                if r.status_code == 400:
                    last_err = err_meta
                    continue
                return {}, err_meta

            j = r.json()
            out_txt = _extract_responses_output_text(j).strip()
            if not out_txt:
                # Some variants return "output_text"
                out_txt = (j.get("output_text") if isinstance(j, dict) else "") or ""
                out_txt = str(out_txt).strip()

            ov: Any = {}
            if out_txt:
                try:
                    ov = json.loads(out_txt)
                except Exception:
                    # try extract first JSON object inside
                    if "{" in out_txt and "}" in out_txt:
                        blob = out_txt[out_txt.find("{"):out_txt.rfind("}") + 1]
                        try:
                            ov = json.loads(blob)
                        except Exception:
                            ov = {}
            if not isinstance(ov, dict):
                ov = {}

            return ov, {"ok": True, "model": model}
        except Exception as e:
            last_err = {"ok": False, "error": f"{type(e).__name__}", "detail": str(e), "model": model}
            continue

    return {}, (locals().get("last_err") if "last_err" in locals() else {"ok": False, "error": "unknown", "model": model})
