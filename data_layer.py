
import requests
import math
import os
import json

def fetch_external_features(scope="GLOBAL"):
    feats = {
        "global_risk_index": 0.0,
        "war_probability": 0.0,
        "financial_stress": 0.0,
        "macro_risk_score": 0.0
    }

    meta = {"parts": {}}

    # --- GDELT (always attempt) ---
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": "war OR invasion OR missile OR crisis",
            "mode": "ArtList",
            "maxrecords": 10,
            "format": "json"
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            count = len(data.get("articles", []))
            feats["war_probability"] = min(1.0, math.log1p(count) / 5.0)
            feats["global_risk_index"] = feats["war_probability"]
            meta["parts"]["gdelt"] = {"ok": True, "detail": f"articles={count}"}
        else:
            meta["parts"]["gdelt"] = {"ok": False, "error": f"http_{r.status_code}"}
    except Exception as e:
        meta["parts"]["gdelt"] = {"ok": False, "error": str(e)}

    # --- NewsAPI ---
    news_key = os.getenv("NEWSAPI_KEY")
    if news_key:
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {"q": "war OR crisis", "apiKey": news_key}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                meta["parts"]["newsapi"] = {"ok": True}
            else:
                meta["parts"]["newsapi"] = {"ok": False, "error": f"http_{r.status_code}"}
        except Exception as e:
            meta["parts"]["newsapi"] = {"ok": False, "error": str(e)}
    else:
        meta["parts"]["newsapi"] = {"ok": False, "error": "missing_api_key"}

    # --- OpenAI ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            url = "https://api.openai.com/v1/responses"
            headers = {"Authorization": f"Bearer {openai_key}"}
            payload = {"model": "gpt-4.1-mini", "input": "Return JSON risk 0-1."}
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            if r.status_code == 200:
                meta["parts"]["openai"] = {"ok": True}
            else:
                meta["parts"]["openai"] = {"ok": False, "error": f"http_{r.status_code}"}
        except Exception as e:
            meta["parts"]["openai"] = {"ok": False, "error": str(e)}
    else:
        meta["parts"]["openai"] = {"ok": False, "error": "missing_api_key"}

    return feats, meta
