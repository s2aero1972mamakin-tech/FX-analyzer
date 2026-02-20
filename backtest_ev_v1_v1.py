
"""
backtest_ev_v1.py
Ver1: 状態確率(Softmax) → EV → 発注方式選択 の "簡易ウォークフォワード" 検証

目的:
- 「勝率」ではなく「期待値（AvgR）」がプラスの条件だけ撃つ設計が妥当かを、10年規模で粗く確認する。
- ここでOKになっても、実運用にはスプレッド/スリッページ/約定/指値到達率/ロール等の厳密化が必要。

使い方:
  python backtest_ev_v1.py --pair_label "USD/JPY" --pair_symbol "JPY=X" --min_expected_R 0.10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math
import pandas as pd
import yfinance as yf


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    mx = max(logits.values())
    exps = {k: math.exp(v - mx) for k, v in logits.items()}
    s = sum(exps.values()) or 1.0
    return {k: exps[k] / s for k in exps}

def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d[["Open","High","Low","Close"]].dropna()
    d["SMA_25"] = d["Close"].rolling(25).mean()
    d["SMA_75"] = d["Close"].rolling(75).mean()
    # RSI
    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + (gain / loss)))
    # ATR
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    d["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return d.dropna()

def state_probs_v1(row: pd.Series) -> Dict[str, float]:
    price = float(row["Close"])
    atr = float(row["ATR"])
    rsi = float(row["RSI"])
    sma25 = float(row["SMA_25"])
    sma75 = float(row["SMA_75"])

    eps = 1e-9
    trend_strength = abs(sma25 - sma75) / max(atr, eps)          # ATR単位のトレンド差
    slope = (sma25 - sma75) / max(price, eps)                    # 方向
    atr_ratio = atr / max(price, eps)

    # 外部データが無い前提の簡易版（本番は logic.py 側で外部データを加点）
    range_bias = _clamp((1.3 - trend_strength), -2.0, 2.0) + _clamp((0.010 - atr_ratio) * 50.0, -2.0, 2.0)

    logits = {
        "trend_up":  1.6 * slope + 0.25 * ((rsi - 50.0) / 10.0) + 0.10 * trend_strength,
        "trend_down": -1.6 * slope + 0.25 * ((50.0 - rsi) / 10.0) + 0.10 * trend_strength,
        "range":  0.9 * range_bias - 0.35 * trend_strength,
        "risk_off": 0.25 * _clamp((atr_ratio - 0.010) * 80.0, -2.0, 2.0),
    }
    return _softmax(logits)

def label_state(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)

def build_state_stats(df_ind: pd.DataFrame, horizon_days: int) -> Dict[str, Dict[str, float]]:
    d = df_ind.copy()
    d["fwd_ret"] = d["Close"].shift(-horizon_days) / d["Close"] - 1.0
    d["R"] = d["fwd_ret"] / (d["ATR"] / d["Close"]).replace(0, float("nan"))
    d = d.dropna(subset=["R"])
    # states
    states=[]
    for _, row in d.iterrows():
        p = state_probs_v1(row)
        states.append(label_state(p))
    d["state"] = states
    out: Dict[str, Dict[str, float]] = {}
    for st in ["trend_up","trend_down","range","risk_off"]:
        sub = d[d["state"] == st]
        if sub.empty:
            out[st] = {"mean_R": 0.0, "p_win": 0.0, "n": 0}
        else:
            out[st] = {
                "mean_R": float(sub["R"].mean()),
                "p_win": float((sub["R"] > 0).mean()),
                "n": int(len(sub)),
            }
    return out

def compute_ev(probs: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    ev = 0.0
    pwin = 0.0
    for st, pr in probs.items():
        ev += float(pr) * float(stats.get(st, {}).get("mean_R", 0.0))
        pwin += float(pr) * float(stats.get(st, {}).get("p_win", 0.0))
    return float(ev), float(pwin)

@dataclass
class WFResult:
    start: str
    end: str
    trades: int
    avg_R: float
    win_rate: float
    max_dd_R: float

def walk_forward(df_ind: pd.DataFrame, train_years: int, test_months: int, horizon_days: int, min_expected_R: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df_ind.copy()
    d = d.sort_index()
    # month boundaries
    start = d.index.min()
    end = d.index.max()
    if start is None or end is None:
        raise ValueError("no data")

    # iterate by test window
    results=[]
    equity_R=0.0
    peak=0.0
    max_dd=0.0
    trades=0
    wins=0
    R_list=[]

    cur_start = start
    # align start to month
    cur_start = pd.Timestamp(year=cur_start.year, month=cur_start.month, day=1, tz=cur_start.tz)
    while True:
        train_end = cur_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        train = d[(d.index >= cur_start) & (d.index < train_end)]
        test = d[(d.index >= train_end) & (d.index < test_end)]
        if len(train) < 120 or len(test) < 30:
            cur_start = cur_start + pd.DateOffset(months=test_months)
            continue

        stats = build_state_stats(train, horizon_days=horizon_days)

        # test simulate: 1bar=1 decision, take trade if EV >= threshold
        for ts, row in test.iterrows():
            probs = state_probs_v1(row)
            ev, pwin = compute_ev(probs, stats)
            if ev < min_expected_R:
                continue
            # realized R
            fwd = d.loc[ts:]["Close"].shift(-horizon_days).iloc[0] / row["Close"] - 1.0
            R = fwd / (row["ATR"] / row["Close"])
            if pd.isna(R):
                continue
            R = float(R)
            R_list.append(R)
            trades += 1
            if R > 0:
                wins += 1
            equity_R += R
            peak = max(peak, equity_R)
            dd = peak - equity_R
            max_dd = max(max_dd, dd)

        results.append(WFResult(
            start=str(cur_start.date()),
            end=str(test_end.date()),
            trades=trades,
            avg_R=float(sum(R_list)/len(R_list)) if R_list else 0.0,
            win_rate=float(wins/trades) if trades else 0.0,
            max_dd_R=float(max_dd),
        ))

        cur_start = cur_start + pd.DateOffset(months=test_months)

    df_out = pd.DataFrame([r.__dict__ for r in results])
    summary = {
        "trades": trades,
        "avg_R": float(sum(R_list)/len(R_list)) if R_list else 0.0,
        "win_rate": float(wins/trades) if trades else 0.0,
        "max_dd_R": float(max_dd),
    }
    return df_out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_label", type=str, default="USD/JPY")
    ap.add_argument("--pair_symbol", type=str, default="JPY=X")
    ap.add_argument("--period", type=str, default="10y")
    ap.add_argument("--horizon_days", type=int, default=5)
    ap.add_argument("--train_years", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=6)
    ap.add_argument("--min_expected_R", type=float, default=0.10)
    args = ap.parse_args()

    df = yf.Ticker(args.pair_symbol).history(period=args.period, interval="1d")
    if df is None or df.empty:
        raise SystemExit("no data from yfinance")

    ind = _indicators(df)
    wf, summ = walk_forward(ind, train_years=args.train_years, test_months=args.test_months, horizon_days=args.horizon_days, min_expected_R=args.min_expected_R)

    print("=== Walk-Forward (simplified) ===")
    if not wf.empty:
        print(wf.tail(10).to_string(index=False))
    print("\n=== Summary ===")
    for k, v in summ.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
