"""data_layer.py

Ver1: 統合AI状態確率モデルへ直結する「外部データ取得層」

目的:
 - main.py が作る context_data に、ニュース/経済指標/金利差/COT の特徴量を追加する。
 - Streamlit Cloud / GitHub 運用を想定し、
   - API Key は st.secrets か環境変数、または main.py から渡される cfg に置く。
   - Key 未設定でも落ちない（0でフォールバック）。
 - APIへのアクセス頻度を抑えるため、簡易ディスクキャッシュ(TTL)を内蔵。

採用API（Ver1決め打ち）:
 - News:
     - GDELT 2.1 DOC API（無料・キー不要）: 直近数十時間の記事一覧
     - NewsAPI（任意・キー必要）: 記事一覧
 - Economic calendar (forecast/actual): TradingEconomics（キー推奨、guest:guest も可）
 - Rates: FRED（キー必要）
 - COT: CFTC TFF Futures Only（FinFutWk.txt, キー不要）

注意:
 - これは「勝てる」を保証するものではありません。特徴量を増やすための土台です。
 - APIの仕様変更・レート制限・一時障害は普通に起きるため、必ずフォールバックを用意。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# Secrets / config
# -------------------------

def _get_secret(name: str, default: str = "") -> str:
    """Try st.secrets then env var."""
    # streamlit secrets
    try:
        import streamlit as st  # type: ignore

        try:
            if name in st.secrets:
                v = st.secrets.get(name)
                return str(v) if v is not None else default
        except Exception:
            pass
        # nested common patterns
        try:
            if "api_keys" in st.secrets and name in st.secrets["api_keys"]:
                v = st.secrets["api_keys"].get(name)
                return str(v) if v is not None else default
        except Exception:
            pass
    except Exception:
        pass

    return str(os.environ.get(name, default) or default)


@dataclass
class ExternalConfig:
    enable: bool = True
    enable_news: bool = True
    enable_econ: bool = True
    enable_rates: bool = True
    enable_cot: bool = True

    # provider switches
    news_provider: str = "GDELT"  # GDELT or NEWSAPI

    # keys (optional)
    newsapi_key: str = ""
    tradingeconomics_key: str = ""  # can be 'guest:guest'
    fred_api_key: str = ""

    # lookbacks
    news_timespan: str = "24h"
    econ_days_back: int = 14


def _cfg_from_any(cfg: Optional[dict] = None) -> ExternalConfig:
    if not isinstance(cfg, dict):
        cfg = {}

    # allow env/secrets fallback
    return ExternalConfig(
        enable=bool(cfg.get("enable", True)),
        enable_news=bool(cfg.get("enable_news", True)),
        enable_econ=bool(cfg.get("enable_econ", True)),
        enable_rates=bool(cfg.get("enable_rates", True)),
        enable_cot=bool(cfg.get("enable_cot", True)),
        news_provider=str(cfg.get("news_provider", "GDELT") or "GDELT").upper(),
        newsapi_key=str(cfg.get("newsapi_key") or _get_secret("NEWSAPI_KEY", "")),
        tradingeconomics_key=str(cfg.get("tradingeconomics_key") or _get_secret("TRADING_ECONOMICS_KEY", "")),
        fred_api_key=str(cfg.get("fred_api_key") or _get_secret("FRED_API_KEY", "")),
        news_timespan=str(cfg.get("news_timespan", "24h") or "24h"),
        econ_days_back=int(cfg.get("econ_days_back", 14) or 14),
    )


# -------------------------
# Simple cache
# -------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_fx_external_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", key)
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _cache_get(key: str, ttl_s: int) -> Optional[dict]:
    path = _cache_path(key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ts = float(obj.get("_ts", 0.0) or 0.0)
        if time.time() - ts <= ttl_s:
            return obj.get("data")
    except Exception:
        return None
    return None


def _cache_set(key: str, data: dict) -> None:
    path = _cache_path(key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "data": data}, f, ensure_ascii=False)
    except Exception:
        pass


def _http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 12) -> Any:
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------------
# News
# -------------------------

_POS_WORDS = {
    "surge", "soar", "rally", "beats", "strong", "hawkish", "tighten", "rate hike",
    "risk-on", "optimism", "upgrade", "boom", "growth",
}
_NEG_WORDS = {
    "plunge", "slump", "crash", "miss", "weak", "dovish", "cut", "rate cut",
    "risk-off", "fear", "recession", "downgrade", "crisis", "default",
}


def _simple_sentiment_score(text: str) -> float:
    """Very small lexicon score in [-1, 1]."""
    if not text:
        return 0.0
    t = (text or "").lower()
    pos = 0
    neg = 0
    for w in _POS_WORDS:
        if w in t:
            pos += 1
    for w in _NEG_WORDS:
        if w in t:
            neg += 1
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / float(pos + neg)
    # soften
    return max(-1.0, min(1.0, 0.75 * raw))


def _pair_query_terms(pair_label: str) -> str:
    # "USD/JPY (ドル円)" -> base,quote
    head = (pair_label or "").split()[0]
    base = "USD"; quote = "JPY"
    if "/" in head:
        try:
            base, quote = head.split("/")[:2]
            base = base.strip()[:3].upper()
            quote = quote.strip()[:3].upper()
        except Exception:
            base, quote = "USD", "JPY"
    # add central bank hints
    cb = []
    if base == "USD" or quote == "USD":
        cb += ["Fed", "FOMC", "Federal Reserve"]
    if base == "JPY" or quote == "JPY":
        cb += ["BOJ", "Bank of Japan"]
    if base == "EUR" or quote == "EUR":
        cb += ["ECB", "European Central Bank"]
    if base == "GBP" or quote == "GBP":
        cb += ["BoE", "Bank of England"]
    if base == "AUD" or quote == "AUD":
        cb += ["RBA", "Reserve Bank of Australia"]
    if base == "CAD" or quote == "CAD":
        cb += ["BoC", "Bank of Canada"]

    pair_terms = [f"{base}{quote}", f"{base}/{quote}"]
    if base == "USD" and quote == "JPY":
        pair_terms += ["dollar yen", "ドル円"]
    if base == "EUR" and quote == "USD":
        pair_terms += ["euro dollar", "ユーロドル"]
    if base == "GBP" and quote == "USD":
        pair_terms += ["pound dollar", "ポンドドル"]

    # GDELT query language supports OR with " OR "
    q = " OR ".join([f'"{x}"' for x in (pair_terms + [base, quote] + cb)])
    return q


def fetch_news_sentiment(pair_label: str, cfg: ExternalConfig) -> Tuple[float, dict]:
    """Return (sentiment[-1..1], meta)."""
    meta: dict = {"provider": cfg.news_provider}
    if not cfg.enable_news:
        return 0.0, {**meta, "disabled": True}

    cache_key = f"news_{cfg.news_provider}_{pair_label}_{cfg.news_timespan}"
    cached = _cache_get(cache_key, ttl_s=10 * 60)
    if isinstance(cached, dict) and "sentiment" in cached:
        return float(cached.get("sentiment", 0.0) or 0.0), cached.get("meta", meta)

    try:
        if cfg.news_provider == "NEWSAPI":
            # https://newsapi.org
            if not cfg.newsapi_key:
                return 0.0, {**meta, "error": "NEWSAPI_KEY_missing"}
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": (pair_label.split()[0] if pair_label else "FX"),
                "pageSize": 50,
                "sortBy": "publishedAt",
                "language": "en",
            }
            headers = {"X-Api-Key": cfg.newsapi_key}
            js = _http_get_json(url, params=params, headers=headers)
            arts = js.get("articles") or []
            scores = []
            for a in arts[:50]:
                title = str(a.get("title") or "")
                desc = str(a.get("description") or "")
                scores.append(_simple_sentiment_score(title + " " + desc))
            sent = float(sum(scores) / len(scores)) if scores else 0.0
            meta.update({"n": int(len(scores)), "query": params.get("q")})
        else:
            # default: GDELT DOC 2.1
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            q = _pair_query_terms(pair_label)
            params = {
                "query": q,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": 75,
                "timespan": cfg.news_timespan,
                "sort": "hybridrel",
            }
            js = _http_get_json(url, params=params)
            arts = (js or {}).get("articles") or []
            scores = []
            for a in arts[:75]:
                title = str(a.get("title") or "")
                # some fields exist: seendate, sourceCountry, domain, url
                scores.append(_simple_sentiment_score(title))
            sent = float(sum(scores) / len(scores)) if scores else 0.0
            meta.update({"n": int(len(scores)), "query": q})

        out = {"sentiment": sent, "meta": meta}
        _cache_set(cache_key, out)
        return sent, meta

    except Exception as e:
        return 0.0, {**meta, "error": f"news_fetch_failed:{type(e).__name__}", "detail": str(e)[:200]}


# -------------------------
# Economic Calendar (TradingEconomics)
# -------------------------

def _parse_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    # remove % and commas
    s = s.replace(",", "").replace("%", "")
    # units like 'B', 'M'
    mul = 1.0
    if s.endswith("B"):
        mul = 1e9
        s = s[:-1]
    elif s.endswith("M"):
        mul = 1e6
        s = s[:-1]
    elif s.endswith("K"):
        mul = 1e3
        s = s[:-1]
    try:
        return float(s) * mul
    except Exception:
        return None


def fetch_economic_surprises(pair_label: str, cfg: ExternalConfig) -> Tuple[Dict[str, float], dict]:
    """Return dict: {cpi_surprise, nfp_surprise} (signed, roughly z-like)."""
    meta: dict = {"provider": "TRADINGECONOMICS"}
    if not cfg.enable_econ:
        return {"cpi_surprise": 0.0, "nfp_surprise": 0.0}, {**meta, "disabled": True}

    cache_key = f"econ_te_{pair_label}_{cfg.econ_days_back}"
    cached = _cache_get(cache_key, ttl_s=30 * 60)
    if isinstance(cached, dict) and "cpi_surprise" in cached:
        return {"cpi_surprise": float(cached.get("cpi_surprise", 0.0) or 0.0), "nfp_surprise": float(cached.get("nfp_surprise", 0.0) or 0.0)}, cached.get("meta", meta)

    key = cfg.tradingeconomics_key or ""
    if not key:
        # allow guest mode (works for some endpoints)
        key = "guest:guest"
        meta["using_guest"] = True

    try:
        url = "https://api.tradingeconomics.com/calendar"
        # We'll fetch latest calendar and filter (avoid complex date params for Ver1)
        params = {"c": key, "f": "json"}
        js = _http_get_json(url, params=params)
        items = js if isinstance(js, list) else []

        # Determine which countries matter (from pair)
        head = (pair_label or "").split()[0]
        base = "USD"; quote = "JPY"
        if "/" in head:
            try:
                base, quote = head.split("/")[:2]
                base = base.strip()[:3].upper()
                quote = quote.strip()[:3].upper()
            except Exception:
                base, quote = "USD", "JPY"
        wanted_countries = set()
        if base == "USD" or quote == "USD":
            wanted_countries.add("United States")
        if base == "JPY" or quote == "JPY":
            wanted_countries.add("Japan")
        if base == "EUR" or quote == "EUR":
            wanted_countries.add("Euro Area")
        if base == "GBP" or quote == "GBP":
            wanted_countries.add("United Kingdom")
        if base == "AUD" or quote == "AUD":
            wanted_countries.add("Australia")
        if base == "CAD" or quote == "CAD":
            wanted_countries.add("Canada")

        # target events
        def _is_cpi(ev: str) -> bool:
            e = (ev or "").lower()
            return ("inflation" in e and "rate" in e) or ("cpi" in e) or ("consumer price" in e)

        def _is_nfp(ev: str) -> bool:
            e = (ev or "").lower()
            return ("non farm" in e) or ("non-farm" in e) or ("payroll" in e)

        cpi_best = None
        nfp_best = None
        for it in items:
            try:
                country = str(it.get("Country") or "")
                if wanted_countries and country not in wanted_countries:
                    continue
                event = str(it.get("Event") or "")
                actual = _parse_num(it.get("Actual"))
                forecast = _parse_num(it.get("Forecast"))
                tef = _parse_num(it.get("TEForecast"))
                if forecast is None and tef is not None:
                    forecast = tef
                if actual is None or forecast is None:
                    continue

                row = {
                    "country": country,
                    "event": event,
                    "actual": float(actual),
                    "forecast": float(forecast),
                    "date": str(it.get("Date") or ""),
                    "importance": int(_parse_num(it.get("Importance")) or 0),
                }
                # pick most recent-ish by LastUpdate field if available
                if _is_cpi(event):
                    # prefer importance high
                    if cpi_best is None or row["importance"] >= cpi_best.get("importance", 0):
                        cpi_best = row
                if _is_nfp(event):
                    if nfp_best is None or row["importance"] >= nfp_best.get("importance", 0):
                        nfp_best = row
            except Exception:
                continue

        def _surprise(row: Optional[dict]) -> float:
            if not row:
                return 0.0
            a = float(row["actual"])
            f = float(row["forecast"])
            denom = max(1e-9, abs(f))
            s = (a - f) / denom
            # soften scaling (roughly "z-like")
            return float(max(-10.0, min(10.0, 3.0 * s)))

        out = {
            "cpi_surprise": _surprise(cpi_best),
            "nfp_surprise": _surprise(nfp_best),
        }
        meta.update({"picked_cpi": cpi_best, "picked_nfp": nfp_best})
        _cache_set(cache_key, {**out, "meta": meta})
        return out, meta

    except Exception as e:
        return {"cpi_surprise": 0.0, "nfp_surprise": 0.0}, {**meta, "error": f"econ_fetch_failed:{type(e).__name__}", "detail": str(e)[:200]}


# -------------------------
# Rates (FRED)
# -------------------------

def fetch_fred_series_last(series_id: str, api_key: str, n: int = 10) -> List[Tuple[str, float]]:
    """Return last N observations as (date, value)."""
    if not api_key:
        raise RuntimeError("FRED_API_KEY_missing")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": int(max(1, n)),
    }
    js = _http_get_json(url, params=params)
    obs = js.get("observations") or []
    out: List[Tuple[str, float]] = []
    for o in obs:
        d = str(o.get("date") or "")
        v = _parse_num(o.get("value"))
        if v is None:
            continue
        out.append((d, float(v)))
    return out


def fetch_rate_diff_change(pair_label: str, cfg: ExternalConfig) -> Tuple[float, dict]:
    """Return interest rate differential change (percentage points) over ~5 observations."""
    meta: dict = {"provider": "FRED"}
    if not cfg.enable_rates:
        return 0.0, {**meta, "disabled": True}

    cache_key = f"rates_fred_{pair_label}"
    cached = _cache_get(cache_key, ttl_s=6 * 60 * 60)
    if isinstance(cached, dict) and "rate_diff_change" in cached:
        return float(cached.get("rate_diff_change", 0.0) or 0.0), cached.get("meta", meta)

    # For Ver1, use 10Y yields:
    # - US: DGS10
    # - JP: IRLTLT01JPM156N (Long-Term Government Bond Yields: 10-year: Japan)
    api_key = cfg.fred_api_key
    if not api_key:
        return 0.0, {**meta, "error": "FRED_API_KEY_missing"}

    try:
        us = fetch_fred_series_last("DGS10", api_key, n=8)
        jp = fetch_fred_series_last("IRLTLT01JPM156N", api_key, n=8)

        # align by index (descending). Use latest available values.
        if not us or not jp:
            return 0.0, {**meta, "error": "rate_series_empty"}
        us_now = us[0][1]
        jp_now = jp[0][1]
        # 5th element in descending list approximates 5 trading days
        us_prev = us[5][1] if len(us) > 5 else us[-1][1]
        jp_prev = jp[5][1] if len(jp) > 5 else jp[-1][1]

        diff_now = float(us_now) - float(jp_now)
        diff_prev = float(us_prev) - float(jp_prev)
        change = float(diff_now - diff_prev)  # percentage points

        meta.update({
            "us_now": us_now,
            "jp_now": jp_now,
            "diff_now": diff_now,
            "diff_prev": diff_prev,
        })
        _cache_set(cache_key, {"rate_diff_change": change, "meta": meta})
        return change, meta
    except Exception as e:
        return 0.0, {**meta, "error": f"rate_fetch_failed:{type(e).__name__}", "detail": str(e)[:200]}


# -------------------------
# COT (CFTC TFF Futures Only)
# -------------------------

_COT_URL = "https://www.cftc.gov/dea/newcot/FinFutWk.txt"

_COT_MARKET_HINTS = {
    "JPY": "JAPANESE YEN",
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND",
    "AUD": "AUSTRALIAN DOLLAR",
    "CAD": "CANADIAN DOLLAR",
    "CHF": "SWISS FRANC",
}


def fetch_cot_features(pair_label: str, cfg: ExternalConfig) -> Tuple[Dict[str, float], dict]:
    """Return COT features (TFF Futures Only): leveraged_net_pctoi, asset_net_pctoi."""
    meta: dict = {"provider": "CFTC_TFF_FUTONLY", "url": _COT_URL}
    if not cfg.enable_cot:
        return {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}, {**meta, "disabled": True}

    cache_key = f"cot_fin_{pair_label}"
    cached = _cache_get(cache_key, ttl_s=24 * 60 * 60)
    if isinstance(cached, dict) and "cot_leveraged_net_pctoi" in cached:
        return {
            "cot_leveraged_net_pctoi": float(cached.get("cot_leveraged_net_pctoi", 0.0) or 0.0),
            "cot_asset_net_pctoi": float(cached.get("cot_asset_net_pctoi", 0.0) or 0.0),
        }, cached.get("meta", meta)

    # choose the quote currency future (practical for USD/JPY -> JPY futures)
    head = (pair_label or "").split()[0]
    base = "USD"; quote = "JPY"
    if "/" in head:
        try:
            base, quote = head.split("/")[:2]
            base = base.strip()[:3].upper()
            quote = quote.strip()[:3].upper()
        except Exception:
            base, quote = "USD", "JPY"
    ccy = quote if quote != "USD" else base
    hint = _COT_MARKET_HINTS.get(ccy)
    if not hint:
        return {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}, {**meta, "error": f"cot_ccy_unsupported:{ccy}"}

    try:
        # FinFutWk.txt is CSV-like with quoted market names. We'll parse via python csv splitting.
        txt = requests.get(_COT_URL, timeout=15).text
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        # find rows that match market name hint
        rows = []
        for ln in lines:
            if hint in ln:
                rows.append(ln)
        if not rows:
            return {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}, {**meta, "error": f"cot_market_not_found:{hint}"}

        # pick first (latest date) row by report date field (3rd column)
        # Example snippet shows: "NAME",260106,2026-01-06,....
        def _row_date(r: str) -> str:
            try:
                parts = [p.strip() for p in r.split(",")]
                return parts[2]
            except Exception:
                return ""
        rows.sort(key=_row_date, reverse=True)
        row = rows[0]
        parts = [p.strip() for p in row.split(",")]
        # Layout (TFF Futures Only) is long; we only need a few:
        # 0 Market_and_Exchange_Names
        # 1 As_of_Date_In_Form_YYMMDD
        # 2 As_of_Date_In_Form_YYYY-MM-DD
        # 7 Open_Interest_All
        # Dealer Positions: 8-11? (long/short/spreading?)
        # Asset Manager: ...
        # Leveraged Funds: ...
        # NOTE: The exact indices can vary across reports. We'll use a robust heuristic:
        # - Open interest appears early and is the first big integer after contract code fields.
        # - In practice, in FinFutWk.txt, Open interest all is at index 7.
        market = parts[0].strip('"')
        date_iso = parts[2]

        # try fixed indices first
        oi = _parse_num(parts[7]) if len(parts) > 7 else None
        # leveraged long/short indices guess: 12/13? but depends.
        # We'll scan for the first 2*5 blocks after OI:
        nums = [_parse_num(p) for p in parts]
        nums2 = [float(x) if x is not None else None for x in nums]

        def _safe(idx: int) -> Optional[float]:
            if idx < 0 or idx >= len(nums2):
                return None
            return nums2[idx]

        # Heuristic based on common FinFutWk layout:
        # index 7 = OI
        # Dealer: 8-11 (Long_All, Short_All, Spreading_All, ???)
        # Asset Manager: 12-15
        # Leveraged Funds: 16-19
        # Other: 20-23
        # Nonreportable: 24-25
        dealer_long = _safe(8)
        dealer_short = _safe(9)
        asset_long = _safe(12)
        asset_short = _safe(13)
        lev_long = _safe(16)
        lev_short = _safe(17)

        # fallback: if indices look empty, attempt shifted layout (some rows have 0 placeholders)
        if oi is None or oi == 0:
            # find first integer-like after idx 5
            for j in range(5, min(len(nums2), 25)):
                v = nums2[j]
                if v is not None and v > 1000:
                    oi = float(v)
                    # shift base so that dealer_long is next
                    base = j + 1
                    dealer_long = _safe(base)
                    dealer_short = _safe(base + 1)
                    asset_long = _safe(base + 4)
                    asset_short = _safe(base + 5)
                    lev_long = _safe(base + 8)
                    lev_short = _safe(base + 9)
                    break

        def _net_pctoi(lng: Optional[float], sht: Optional[float], oi_v: Optional[float]) -> float:
            if lng is None or sht is None or oi_v is None or oi_v <= 0:
                return 0.0
            return float(max(-1.0, min(1.0, (float(lng) - float(sht)) / float(oi_v))))

        lev_net = _net_pctoi(lev_long, lev_short, oi)
        asset_net = _net_pctoi(asset_long, asset_short, oi)
        out = {"cot_leveraged_net_pctoi": lev_net, "cot_asset_net_pctoi": asset_net}
        meta.update({
            "market": market,
            "date": date_iso,
            "open_interest": oi,
            "leveraged_long": lev_long,
            "leveraged_short": lev_short,
            "asset_long": asset_long,
            "asset_short": asset_short,
        })
        _cache_set(cache_key, {**out, "meta": meta})
        return out, meta

    except Exception as e:
        return {"cot_leveraged_net_pctoi": 0.0, "cot_asset_net_pctoi": 0.0}, {**meta, "error": f"cot_fetch_failed:{type(e).__name__}", "detail": str(e)[:200]}


# -------------------------
# Orchestrator
# -------------------------

def enrich_context_v1(context_data: dict, cfg: Optional[dict] = None) -> Tuple[dict, dict]:
    """Return (updated_context, meta). Adds:
    - news_sentiment
    - cpi_surprise, nfp_surprise
    - rate_diff_change
    - cot_leveraged_net_pctoi, cot_asset_net_pctoi
    """
    ctx = dict(context_data or {})
    pair_label = str(ctx.get("pair_label") or ctx.get("pair") or "USD/JPY (ドル円)")
    conf = _cfg_from_any(cfg)
    if not conf.enable:
        return ctx, {"disabled": True}

    meta: dict = {"pair_label": pair_label, "ts": time.time()}

    # news
    news_s, news_meta = fetch_news_sentiment(pair_label, conf)
    ctx["news_sentiment"] = float(max(-1.0, min(1.0, news_s)))
    meta["news"] = news_meta

    # econ
    econ, econ_meta = fetch_economic_surprises(pair_label, conf)
    ctx["cpi_surprise"] = float(econ.get("cpi_surprise", 0.0) or 0.0)
    ctx["nfp_surprise"] = float(econ.get("nfp_surprise", 0.0) or 0.0)
    meta["econ"] = econ_meta

    # rates
    rd, rd_meta = fetch_rate_diff_change(pair_label, conf)
    # clip to model expectation range
    ctx["rate_diff_change"] = float(max(-5.0, min(5.0, rd)))
    meta["rates"] = rd_meta

    # cot
    cot, cot_meta = fetch_cot_features(pair_label, conf)
    ctx["cot_leveraged_net_pctoi"] = float(max(-1.0, min(1.0, cot.get("cot_leveraged_net_pctoi", 0.0) or 0.0)))
    ctx["cot_asset_net_pctoi"] = float(max(-1.0, min(1.0, cot.get("cot_asset_net_pctoi", 0.0) or 0.0)))
    meta["cot"] = cot_meta

    ctx["external_meta"] = meta
    return ctx, meta
