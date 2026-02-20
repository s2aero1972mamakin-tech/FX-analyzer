# ==========================
# ADD: compute_indicators (compat)
# main.py が呼ぶ compute_indicators(df) が無いと落ちるので後方互換で追加
# ==========================
def compute_indicators(df):
    """
    df: OHLC DataFrame with columns: Open, High, Low, Close
    returns: dict used by Ver1 (EV/state_probs)
      - sma25, sma75, rsi, atr
      - recent_high20, recent_low20
      - atr_ratio, trend_strength (補助)
    """
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return {
            "sma25": 0.0, "sma75": 0.0, "rsi": 50.0, "atr": 0.0,
            "recent_high20": 0.0, "recent_low20": 0.0,
            "atr_ratio": 0.0, "trend_strength": 0.0,
        }

    d = df.copy()
    # yfinanceで MultiIndex になるケース
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] for c in d.columns]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in d.columns:
            return {
                "sma25": 0.0, "sma75": 0.0, "rsi": 50.0, "atr": 0.0,
                "recent_high20": 0.0, "recent_low20": 0.0,
                "atr_ratio": 0.0, "trend_strength": 0.0,
            }

    d = d[["Open", "High", "Low", "Close"]].dropna()
    if d.empty or len(d) < 80:
        # データ不足でも落ちないようにフォールバック
        last_close = float(d["Close"].iloc[-1]) if not d.empty else 0.0
        return {
            "sma25": last_close, "sma75": last_close, "rsi": 50.0, "atr": 0.0,
            "recent_high20": float(d["High"].tail(20).max()) if not d.empty else last_close,
            "recent_low20": float(d["Low"].tail(20).min()) if not d.empty else last_close,
            "atr_ratio": 0.0, "trend_strength": 0.0,
        }

    close = d["Close"]
    high = d["High"]
    low = d["Low"]

    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 最新値
    price = float(close.iloc[-1])
    atr_v = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    sma25_v = float(sma25.iloc[-1]) if pd.notna(sma25.iloc[-1]) else price
    sma75_v = float(sma75.iloc[-1]) if pd.notna(sma75.iloc[-1]) else price
    rsi_v = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    recent_high20 = float(high.tail(20).max())
    recent_low20 = float(low.tail(20).min())

    eps = 1e-9
    atr_ratio = atr_v / max(price, eps)
    trend_strength = abs(sma25_v - sma75_v) / max(atr_v, eps) if atr_v > 0 else 0.0

    return {
        "sma25": sma25_v,
        "sma75": sma75_v,
        "rsi": rsi_v,
        "atr": atr_v,
        "recent_high20": recent_high20,
        "recent_low20": recent_low20,
        "atr_ratio": float(atr_ratio),
        "trend_strength": float(trend_strength),
    }
