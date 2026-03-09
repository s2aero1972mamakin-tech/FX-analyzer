
# ---------------- PROP TP OPTIMIZER PATCH ----------------
def _liquidity_pool_tp(df, direction, lookback=40):
    try:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())
        if direction == "LONG":

    sl = min(last - 1.2 * atr14, recent_low - 0.15 * atr14)

    regime_mult = _regime_tp_multiple(phase_label)
    atr_tp = last + regime_mult * atr14

    liq_tp = _liquidity_pool_tp(df, "LONG")

    tp = max(atr_tp, liq_tp) if liq_tp else atr_tp

else:

    sl = max(last + 1.2 * atr14, recent_high + 0.15 * atr14)

    regime_mult = _regime_tp_multiple(phase_label)
    atr_tp = last - regime_mult * atr14

    liq_tp = _liquidity_pool_tp(df, "SHORT")

    tp = min(atr_tp, liq_tp) if liq_tp else atr_tp

    entry = last
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 1e-9:
        risk = atr14
    rr = reward / risk
    rr_min = float(ctx_in.get("rr_min_floor", 1.0) or 1.0)
    rr_floor_fail = bool(rr < rr_min)

    # -----------------------------------------------------------------
    # 5) 勝率 proxy（モデル）→ confidenceで縮退（p_eff）
    # -----------------------------------------------------------------
    cont_best = max(float(p_up), float(p_dn))
    if direction == "LONG":
        p_win_model = 0.46 + 0.42 * _clamp((float(p_up) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    else:
        p_win_model = 0.46 + 0.42 * _clamp((float(p_dn) - 0.5) * 2.0, -1.0, 1.0) + 0.10 * (float(strength) - 0.5)
    p_win_model = float(_clamp(p_win_model - _failure_features(df), 0.20, 0.80))

    # 信頼度（0..1）
    structure_ok_dir = (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))
    structure_flag = 1.0 if structure_ok_dir else 0.0
    confidence = float(_clamp(
        0.30
        + 0.40 * float(strength)
        + 0.18 * (float(cont_best) - 0.5)
        + 0.0
        + 0.04 * float(structure_flag),
        0.0, 1.0
    ))

    # p_eff: confidenceが低いほど0.5に寄せる（整合崩れ対策）
    conf_k = float(_clamp(confidence / 0.75, 0.0, 1.0))
    p_eff = float(_clamp(0.5 + (p_win_model - 0.5) * conf_k, 0.20, 0.80))

    # EV (R): EV = p*RR - (1-p)*1
    ev_raw = float(p_eff * float(rr) - (1.0 - p_eff) * 1.0)

    # -----------------------------------------------------------------
    # 6) 外部リスク（macro）: 弱いバイアスとして統合（NO連発の主因にしない）
    # -----------------------------------------------------------------
    gr = _safe_float(ext.get("global_risk_index", ext.get("global_risk", ext.get("risk_off", 0.35))), 0.35)
    war = _safe_float(ext.get("war_probability", ext.get("war", 0.0)), 0.0)
    macro_risk = _safe_float(ext.get("macro_risk_score", None), float("nan"))
    if not (isinstance(macro_risk, (int, float)) and math.isfinite(float(macro_risk))):
        macro_risk = _clamp(0.70 * gr + 0.30 * war, 0.0, 1.0)
    else:
        macro_risk = _clamp(float(macro_risk), 0.0, 1.0)

    # 表示用（ev_adj）は弱いペナルティに留める
    risk_penalty = 0.10 + 0.50 * float(macro_risk)   # 0.10..0.60
    ev_adj = float(ev_raw - 0.18 * float(risk_penalty))

    # -----------------------------------------------------------------
    # 6.5) 経済指標/イベント（直前の実行リスク + 直後の捕獲）
    # -----------------------------------------------------------------
    event_guard_enable = bool(ctx_in.get('event_guard_enable', True))
    event_block_window = bool(ctx_in.get('event_block_high_impact_window', True))
    event_horizon_hours = int(_safe_float(ctx_in.get("event_horizon_hours", 168), 168))
    event_past_lookback_hours = int(_safe_float(ctx_in.get("event_past_lookback_hours", 24), 24))
    event_window_minutes = int(_safe_float(ctx_in.get("event_window_minutes", 60), 60))
    event_impacts = ctx_in.get("event_impacts", None)
    if not isinstance(event_impacts, list) or not event_impacts:
        event_impacts = ["High", "Medium"]
    event_calendar_url = str(ctx_in.get("event_calendar_url", _FF_CAL_URL_DEFAULT) or _FF_CAL_URL_DEFAULT)

    ev_meta = {"ok": False, "status": "off", "err": None, "score": 0.0, "factor": 0.0,
               "window_high": False, "next_high_hours": None, "last_high_hours": None,
               "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}
    if event_guard_enable:
        try:
            ev_meta = _compute_event_risk(
                pair,
                now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"),
                horizon_hours=event_horizon_hours,
                past_lookback_hours=event_past_lookback_hours,
                hours_scale=float(ctx_in.get("event_hours_scale", 24.0) or 24.0),
                norm=float(ctx_in.get("event_norm", 3.0) or 3.0),
                impacts=[str(x) for x in event_impacts],
                high_window_minutes=event_window_minutes,
                url=event_calendar_url,
            )
        except Exception as e:
            ev_meta = {"ok": False, "status": "unknown", "err": f"{type(e).__name__}: {e}", "score": 0.0, "factor": 0.0,
                       "window_high": False, "next_high_hours": None, "last_high_hours": None,
                       "next_any_hours": None, "last_any_hours": None, "upcoming": [], "recent": [], "impact_ccys": {}}

    weekend_risk = float(_compute_weekend_risk(now_tz=str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo")))

    # Thu/Fri are special for swing entries (weekend gap approaches + event clusters).
    try:
        _tz = ZoneInfo(str(ctx_in.get("event_timezone", "Asia/Tokyo") or "Asia/Tokyo"))
        _now_local = datetime.now(tz=_tz)
        _wd = int(_now_local.weekday())  # Mon=0..Sun=6
        weekcross_risk = 1.0 if _wd in (3, 4) else 0.0  # Thu/Fri
        weekcross_weekday = _wd
    except Exception:
        weekcross_risk = 0.0
        weekcross_weekday = None

    # event mode classification (swing)
    try:
        next_high = ev_meta.get("next_high_hours", None)
        last_high = ev_meta.get("last_high_hours", None)
        pre_h = float(ctx_in.get("event_preblock_hours", 24.0) or 24.0)
        pre_h = float(_clamp(pre_h, 6.0, 72.0))
    except Exception:
        next_high = None
        last_high = None
        pre_h = 24.0

    event_mode = "NORMAL"
    if bool(ev_meta.get("window_high", False)) and event_block_window:
        event_mode = "EVENT_WINDOW"
    elif (last_high is not None) and (float(last_high) <= 1.0):
        event_mode = "POST_WAIT"          # 0-1h: wait
    elif (last_high is not None) and (float(last_high) <= 24.0):
        event_mode = "POST_BREAKOUT"      # 1-24h: breakout-only gate
    elif (next_high is not None) and (float(next_high) <= float(pre_h)):
        event_mode = "PRE_EVENT"          # upcoming high-impact is close
    else:
        event_mode = "NORMAL"

    # -----------------------------------------------------------------
    # 7) 動的閾値（フェーズ/構造優先 + リスク時に軽く上げる）
    # -----------------------------------------------------------------
    base_thr = _safe_float(ctx_in.get("dynamic_threshold_base", None), float("nan"))
    if not (isinstance(base_thr, (int, float)) and math.isfinite(float(base_thr))):
        base_thr = _safe_float(ctx_in.get("min_expected_R", 0.08), 0.08)
    base_thr = float(_clamp(float(base_thr), 0.03, 0.25))

    thr_mult = 1.0
    if phase_label in ("UP_TREND", "DOWN_TREND"):
        thr_mult -= 0.16 * float(strength)
    if str(phase).startswith("BREAKOUT"):
        thr_mult -= 0.22 * max(float(strength), float(breakout_strength))
    if phase_label == "RANGE":
        thr_mult += 0.10

    dynamic_threshold = float(_clamp(base_thr * thr_mult, 0.02, 0.30))

    # macro bias is weak
    dynamic_threshold = float(_clamp(dynamic_threshold + 0.03 * float(macro_risk), 0.02, 0.30))

    # upcoming event / weekend / weekcross: threshold add (but do not cause perpetual NO)
    try:
        event_thr_add = float(ctx_in.get("event_threshold_add", 0.18) or 0.18)
        event_thr_add = float(_clamp(event_thr_add, 0.10, 0.30))
        weekend_thr_add = float(ctx_in.get("weekend_threshold_add", 0.03) or 0.03)
        weekend_thr_add = float(_clamp(weekend_thr_add, 0.0, 0.20))
        weekcross_thr_add = float(ctx_in.get("weekcross_threshold_add", 0.03) or 0.03)
        weekcross_thr_add = float(_clamp(weekcross_thr_add, 0.0, 0.20))

        ef = float(ev_meta.get("factor", 0.0) or 0.0)
        # POST_BREAKOUTでは“捕獲”を優先し、閾値上乗せを弱める
        if event_mode == "POST_BREAKOUT":
            ef *= 0.40
        dynamic_threshold = float(_clamp(
            dynamic_threshold
            + event_thr_add * ef
            + weekend_thr_add * float(weekend_risk or 0.0)
            + weekcross_thr_add * float(weekcross_risk or 0.0),
            0.02, 0.30
        ))
    except Exception:
        pass


    # B-rank: condition-specific threshold optimization
    try:
        if phase_label == "RANGE":
            dynamic_threshold = float(_clamp(dynamic_threshold + 0.04, 0.02, 0.30))
        elif phase_label in ("UP_TREND", "DOWN_TREND"):
            dynamic_threshold = float(_clamp(dynamic_threshold - 0.02, 0.02, 0.30))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 8) モメンタム/通貨強弱（補助のみ、上限あり）
    # -----------------------------------------------------------------
    mom_bonus = 0.0
    if direction == "LONG" and mom > 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_up), 0.0, 1.0)
    if direction == "SHORT" and mom < 0:
        mom_bonus = 0.06 * _clamp(float(strength), 0.0, 1.0) * _clamp(float(p_dn), 0.0, 1.0)
    mom_bonus = float(_clamp(mom_bonus, 0.0, 0.06))

    ccy_strength_proxy = 0.0
    try:
        c = df["Close"].astype(float)
        r20 = (c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) >= 21 else 0.0
        r60 = (c.iloc[-1] / c.iloc[-61] - 1.0) if len(c) >= 61 else 0.0
        vol20 = float(c.pct_change().rolling(20).std().iloc[-1]) if len(c) >= 21 else 0.0
        vol60 = float(c.pct_change().rolling(60).std().iloc[-1]) if len(c) >= 61 else 0.0
        z20 = (r20 / (vol20 + 1e-9)) if vol20 > 0 else 0.0
        z60 = (r60 / (vol60 + 1e-9)) if vol60 > 0 else 0.0
        ccy_strength_proxy = float(_clamp(0.5 * z20 + 0.5 * z60, -1.0, 1.0))
    except Exception:
        ccy_strength_proxy = 0.0

    ccy_bonus = 0.0
    if direction == "LONG" and ccy_strength_proxy > 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    elif direction == "SHORT" and ccy_strength_proxy < 0:
        ccy_bonus = 0.04 * _clamp(abs(ccy_strength_proxy), 0.0, 1.0)
    ccy_bonus = float(_clamp(ccy_bonus, 0.0, 0.04))

    ev_gate = float(ev_raw + mom_bonus + ccy_bonus)
    ev_gate = ev_gate * _quality_decay(strength, breakout_ok, hhhl_ok, confidence)
    # B-rank + prop AI adjustments
    ev_gate -= float(_range_center_penalty(range_pos))
    ev_gate -= float(_event_unknown_adjust(ev_meta))
    if _liquidity_sweep(df):
        ev_gate -= 0.25
    vol_exp = float(_volatility_expansion(df))
    ev_gate += 0.08 * vol_exp
    regime = _market_regime(df)
    if regime == "RANGE":
        ev_gate -= 0.15

    # -----------------------------------------------------------------
    # 9) 構造ゲート（最優先）
    # -----------------------------------------------------------------
    breakout_pass = bool(_entry_timing_filter(df, direction) and 
        (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))
        and (float(cont_best) >= 0.57)
        and (max(float(strength), float(breakout_strength)) >= 0.35)
        and (float(macro_risk) <= 0.90)
    )

    # RANGE 端の逆張り（厳格）
    range_edge_setup = False
    try:
        # イベント直前はレンジ逆張りを避ける（事故回避）。直後捕獲はブレイク専用。
        in_pre = (event_mode == "PRE_EVENT")
        if phase_label == "RANGE" and (not in_pre) and (event_mode not in ("EVENT_WINDOW", "POST_WAIT")):
            near_edge = (range_pos <= 0.25) if direction == "LONG" else (range_pos >= 0.75)
            range_edge_setup = bool(
                near_edge
                and (float(rr) >= 1.40)
                and (float(confidence) >= 0.45)
                and (float(cont_best) >= 0.54)
                and (float(macro_risk) <= 0.85)
                and (ev_gate >= float(dynamic_threshold) - 0.02)
            )
    except Exception:
        range_edge_setup = False

    # 全体の構造妥当性
    structure_ok = True
    if phase_label == "RANGE":
        center_avoid = bool(abs(float(range_pos) - 0.5) >= 0.18)
        structure_ok = bool((breakout_pass or range_edge_setup) and center_avoid)
    else:
        if (float(strength) < 0.18) and not (breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok)):
            structure_ok = False

    # POST_BREAKOUTはブレイク根拠必須（取り逃がし防止と事故回避を両立）
    if event_mode == "POST_BREAKOUT":
        structure_ok = bool(breakout_ok or (hhhl_ok if direction == "LONG" else lllh_ok))

    # -----------------------------------------------------------------
    # 10) veto/decision（veto乱立を抑える）
    # -----------------------------------------------------------------
    veto: List[str] = []
    def _veto(msg: str) -> None:
        s = str(msg or "").strip()
        if not s:
            return
        if s not in veto:
            veto.append(s)

    why = ""
    gate_mode = "raw+mom"

    if rr_floor_fail:
        _veto(f"RR不足: {rr:.2f} < {rr_min:.2f}")
        decision = "NO_TRADE"

    # mandatory event window block
    if rr_floor_fail:
        pass
    elif event_guard_enable and event_block_window and event_mode == "EVENT_WINDOW":
        gate_mode = "event_block"
        why = f"高インパクト指標の前後（±{event_window_minutes}分）のため見送り"
        _veto(why)
        decision = "NO_TRADE"
    elif event_guard_enable and event_mode == "POST_WAIT":
        gate_mode = "post_wait"
        why = "高インパクト直後0〜1hは様子見（スプレッド/再反転の不確実性）"
        _veto(why)
        decision = "NO_TRADE"
    elif not structure_ok:
        gate_mode = "structure_veto"
        if phase_label == "RANGE":
            why = "レンジ優勢で構造根拠が不足（ブレイク or 端の逆張り条件が未達）"
        else:
            why = "価格構造の根拠が弱い（トレンド強度/HHHL/ブレイクが不足）"
        _veto(why)
        decision = "NO_TRADE"
    else:
        # EV gate (post-breakout has its own rescue)
        if ev_gate >= float(dynamic_threshold):
            decision = "TRADE"
            if bool(ctx_in.get("sbi_min_lot_guard", True)) and int(ctx_in.get("sbi_min_lot", 1) or 1) >= 1 and float(confidence) < 0.42:
                decision = "NO_TRADE"
                _veto("SBI最小1建リスク")
            if not _entry_timing_filter(df, direction):
                decision = "NO_TRADE"
                veto.append("Entry timing filter rejected")
            why = f"EV通過: {ev_gate:+.3f} ≥ 動的閾値 {float(dynamic_threshold):.3f}"
        elif event_mode == "POST_BREAKOUT" and (ev_gate >= float(dynamic_threshold) - 0.08) and float(confidence) >= 0.42:
            decision = "TRADE"
            gate_mode = "post_breakout_rescue"
            why = f"イベント後捕獲（1〜24hブレイク専用）: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        elif breakout_pass and (ev_gate >= float(dynamic_threshold) - 0.04):
            decision = "TRADE"
            gate_mode = "breakout_rescue"
            why = f"BREAKOUT通過: EV {ev_gate:+.3f} / 閾値 {float(dynamic_threshold):.3f}（救済）"
        else:
            decision = "NO_TRADE"
            _veto(f"EV不足: {ev_gate:+.3f} < 動的閾値 {float(dynamic_threshold):.3f}")

    # -----------------------------------------------------------------
    # 11) 状態確率 / EV内訳（UI用）
    # -----------------------------------------------------------------
    s_up = max(0.0, float(p_up) * (0.55 + 0.75 * float(strength)) + max(0.0, float(mom)) * 0.10)
    s_dn = max(0.0, float(p_dn) * (0.55 + 0.75 * float(strength)) + max(0.0, -float(mom)) * 0.10)
    s_range = max(0.0, (1.0 - float(strength)) * 0.95 + 0.05)
    s_risk = max(0.0, float(macro_risk) * 1.15 + (1.0 - float(cont_best)) * 0.10)

    tot = s_up + s_dn + s_range + s_risk
    if tot <= 1e-12:
        state_probs = {"trend_up": 0.25, "trend_down": 0.25, "range": 0.25, "risk_off": 0.25}
    else:
        state_probs = {
            "trend_up": float(s_up / tot),
            "trend_down": float(s_dn / tot),
            "range": float(s_range / tot),
            "risk_off": float(s_risk / tot),
        }

    if direction == "LONG":
        r_up = max(0.2, float(rr) * 0.85)
        r_dn = -1.0
    else:
        r_dn = max(0.2, float(rr) * 0.85)
        r_up = -1.0
    r_range = (0.12 * float(rr) - 0.35)
    r_riskoff = -0.75
    ev_contribs = {
        "trend_up": float(state_probs["trend_up"] * r_up),
        "trend_down": float(state_probs["trend_down"] * r_dn),
        "range": float(state_probs["range"] * r_range),
        "risk_off": float(state_probs["risk_off"] * r_riskoff),
    }

    # -----------------------------------------------------------------
    # 12) 統合スコア（単一ランキング指標）
    #   - 価格構造を最優先（structure_weight大）
    #   - EVは次点
    #   - イベント影響（通貨別）はペナルティとして統合 → 非影響通貨ペアが相対的に上位化
    # -----------------------------------------------------------------
    structure_score = (
        0.60 * float(strength)
        + (0.25 * float(breakout_strength) if bool(breakout_ok) else 0.0)
        + (0.15 if bool(hhhl_ok) else 0.0)
        + 0.10 * float(_clamp(abs(float(mom)), 0.0, 1.0))
    )
    if phase_label == "RANGE":
        structure_score *= 0.85
    structure_scaled = float(_clamp(structure_score, 0.0, 1.2) / 1.2)

    ev_scaled = float(_clamp((ev_gate + 0.15) / 1.35, 0.0, 1.0))

    ef_up = float(ev_meta.get("factor", 0.0) or 0.0)
    event_pen = 0.20 * ef_up
    if event_mode == "PRE_EVENT":
        event_pen = 0.28 * ef_up
    if event_mode == "POST_BREAKOUT":
        event_pen = 0.10 * ef_up

    event_pen += 0.08 * float(weekend_risk or 0.0) + 0.08 * float(weekcross_risk or 0.0)
    macro_pen = 0.08 * float(macro_risk)

    rank_score = float(
        2.00 * structure_scaled
        + 1.40 * ev_scaled
        + 0.40 * float(confidence)
        - float(event_pen)
        - float(macro_pen)
    )
    final_score = rank_score
    decision_score = float(ev_gate)
    ranking_score = float(rank_score)
    execution_score = float(confidence * (1.0 + max(0.0, float(strength))))

    # -----------------------------------------------------------------
    # 13) ctx（デバッグ/可視化用）
    # -----------------------------------------------------------------
    ctx_out = {
        "pair": pair,
        "phase_label": phase_label,
        "trend_strength": float(strength),
        "momentum_score": float(mom),
        "range_pos": float(range_pos),
        "range_edge_setup": bool(range_edge_setup),
        "market_regime": str(regime),
        "volatility_expansion": float(vol_exp),
        "liquidity_sweep": bool(_liquidity_sweep(df)),
        "ccy_strength_proxy": float(ccy_strength_proxy),
        "ccy_bonus": float(ccy_bonus),
        "cont_p_up": float(p_up),
        "cont_p_dn": float(p_dn),
        "hh_hl_ok": bool(hhhl_ok),
        "ll_lh_ok": bool(lllh_ok),
        "breakout_ok": bool(breakout_ok),
        "breakout_strength": float(breakout_strength),
        "breakout_pass": bool(breakout_pass),
        "rr": float(rr),
        "p_win_model": float(p_win_model),
        "p_eff": float(p_eff),
        "macro_risk_score": float(macro_risk),
        "event_mode": str(event_mode),
        "event_risk_score": float(ev_meta.get("score", 0.0) or 0.0),
        "event_risk_factor": float(ev_meta.get("factor", 0.0) or 0.0),
        "event_window_high": bool(ev_meta.get("window_high", False)),
        "event_next_high_hours": (float(ev_meta.get("next_high_hours")) if ev_meta.get("next_high_hours") is not None else None),
        "event_last_high_hours": (float(ev_meta.get("last_high_hours")) if ev_meta.get("last_high_hours") is not None else None),
        "event_feed_status": str(ev_meta.get("status", "") or ""),
        "event_feed_error": str(ev_meta.get("err", "") or ""),
        "event_upcoming": (ev_meta.get("upcoming", []) or []),
        "event_recent": (ev_meta.get("recent", []) or []),
        "event_impact_ccys": (ev_meta.get("impact_ccys", {}) or {}),
        "weekend_risk": float(weekend_risk),
        "weekcross_risk": float(weekcross_risk or 0.0),
        "weekcross_weekday": (int(weekcross_weekday) if weekcross_weekday is not None else None),
        "mom_bonus": float(mom_bonus),
        "dynamic_threshold": float(dynamic_threshold),
        "dynamic_threshold_base": float(base_thr),
        "dynamic_threshold_mult": float(thr_mult),
        "ev_gate": float(ev_gate),
        "structure_scaled": float(structure_scaled),
        "ev_scaled": float(ev_scaled),
        "rank_score": float(rank_score),
        "event_penalty": float(event_pen),
        "macro_penalty": float(macro_pen),
        "len": int(len(df)),
    }

    # -----------------------------------------------------------------
    # 14) 注文方式の提案（直前:成行禁止 / 直後:ブレイク専用）
    # -----------------------------------------------------------------
    order_type = "MARKET"
    entry_type = "MARKET_NOW"
    exec_guard_notes: List[str] = []

    # setup-based suggestion (even for NO_TRADE; UI上は参考として表示可能)
    setup_kind = "TREND"
    if phase_label == "RANGE" and bool(range_edge_setup):
        setup_kind = "RANGE_EDGE"
    elif bool(breakout_pass) or str(phase).startswith("BREAKOUT") or (event_mode == "POST_BREAKOUT"):
        setup_kind = "BREAKOUT"

    try:
        pip = _pip_size(pair)
        atr_for_entry = max(float(atr14), float(pip) * 10.0)
    except Exception:
        pip = 0.01
        atr_for_entry = float(atr14)

    # Base reco by setup
    if setup_kind == "RANGE_EDGE":
        if direction == "LONG":
            new_entry = entry - 0.25 * atr_for_entry
        else:
            new_entry = entry + 0.25 * atr_for_entry
        order_type = "LIMIT"
        entry_type = "LIMIT_PULLBACK"
        exec_guard_notes.append("レンジ端のため、押し目/戻りの指値を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    elif setup_kind == "BREAKOUT":
        if direction == "LONG":
            new_entry = entry + 0.10 * atr_for_entry
            entry_type = "STOP_BREAKOUT"
        else:
            new_entry = entry - 0.10 * atr_for_entry
            entry_type = "STOP_BREAKDOWN"
        order_type = "STOP"
        exec_guard_notes.append("ブレイク捕獲のため、逆指値（STOP）を推奨")
        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

    # High-impact is close enough → ban MARKET entry (直前:成行禁止)
    event_market_ban_active = False
    event_market_ban_hours = float(ctx_in.get("event_market_ban_hours", 12.0) or 12.0)
    if float(weekcross_risk or 0.0) > 0.0:
        event_market_ban_hours = max(event_market_ban_hours, float(ctx_in.get("weekcross_market_ban_hours", 18.0) or 18.0))
    try:
        nh = ev_meta.get("next_high_hours", None)
        if (nh is not None) and (float(nh) <= float(event_market_ban_hours)):
            event_market_ban_active = True
    except Exception:
        pass

    if decision == "TRADE" and bool(event_market_ban_active) and order_type == "MARKET":
        # If we still ended up MARKET, convert to pending
        try:
            nh = float(ev_meta.get("next_high_hours") or 0.0)
        except Exception:
            nh = None
        if phase_label == "RANGE" and (not breakout_pass):
            # pullback limit
            if direction == "LONG":
                new_entry = entry - 0.25 * atr_for_entry
            else:
                new_entry = entry + 0.25 * atr_for_entry
            order_type = "LIMIT"
            entry_type = "LIMIT_PULLBACK"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → 押し目/戻りの指値を提案"
        else:
            # breakout stop
            if direction == "LONG":
                new_entry = entry + 0.10 * atr_for_entry
                entry_type = "STOP_BREAKOUT"
            else:
                new_entry = entry - 0.10 * atr_for_entry
                entry_type = "STOP_BREAKDOWN"
            order_type = "STOP"
            msg = f"高インパクト指標まで{nh:.1f}hのため成行禁止 → ブレイク逆指値を提案"

        try:
            new_entry = _round_to_pip(float(new_entry), pair)
            delta = float(new_entry) - float(entry)
            entry = float(new_entry)
            sl = _round_to_pip(float(sl) + delta, pair)
            tp = _round_to_pip(float(tp) + delta, pair)
        except Exception:
            pass

        exec_guard_notes.append(msg)
        if why:
            why = why + " / " + msg
        else:
            why = msg

    # lot shrink factors (UI/logging)
    try:
        ef = float(ctx_out.get("event_risk_factor", 0.0) or 0.0)
        ctx_out["event_market_ban_active"] = bool(event_market_ban_active)
        ctx_out["event_market_ban_hours"] = float(event_market_ban_hours)
        ctx_out["exec_guard_notes"] = list(exec_guard_notes)
        ctx_out["order_type_reco"] = str(order_type)
        ctx_out["entry_type_reco"] = str(entry_type)
        ctx_out["lot_shrink_event_factor"] = float(_clamp(1.0 - 0.60 * ef, 0.20, 1.00))
        ctx_out["lot_shrink_weekcross_factor"] = (0.75 if float(weekcross_risk or 0.0) > 0.0 else 1.0)
        ctx_out["lot_shrink_weekend_factor"] = (0.60 if float(weekend_risk or 0.0) > 0.0 else 1.0)
    except Exception:
        pass

    # -----------------------------------------------------------------
    # 15) 保有中のイベント接近対応（縮退/一部利確/建値移動/追加禁止）
    # -----------------------------------------------------------------
    hold_manage = _hold_manage_reco(
        pair=str(pair),
        df=df,
        ctx_in=ctx_in,
        plan_like={"side": side, "entry": entry, "sl": sl, "tp": tp},
        ev_meta=(ev_meta or {}),
        weekend_risk=float(weekend_risk or 0.0),
        weekcross_risk=float(weekcross_risk or 0.0),
    )
    if isinstance(hold_manage, dict) and hold_manage:
        try:
            ctx_out["hold_manage"] = hold_manage
        except Exception:
            pass

    # Trail SL: エントリーから0.5R戻し（見せ方用）
    trail_sl = sl
    try:
        dist_sl = abs(entry - sl)
        if dist_sl > 0:
            trail_sl = entry - 0.5 * dist_sl if direction == "LONG" else (entry + 0.5 * dist_sl)
    except Exception:
        trail_sl = sl

    # 返却（main互換キー）
    plan = {
        "decision": str(decision),
        "direction": str(direction),
        "side": str(side),

        "order_type": str(order_type),
        "entry_type": str(entry_type),

        "entry": float(entry),
        "entry_price": float(entry),
        "sl": float(sl),
        "stop_loss": float(sl),
        "tp": float(tp),
        "take_profit": float(tp),

        "trail_sl": float(trail_sl),
        "extend_factor": 1.0,

        "ev_raw": float(ev_raw),
        "ev_adj": float(ev_adj),

        "expected_R_ev_raw": float(ev_raw),
        "expected_R_ev_adj": float(ev_adj),
        "expected_R_ev": float(ev_gate),

        "rank_score": float(rank_score),
        "final_score": float(final_score),
        "decision_score": float(decision_score),
        "ranking_score": float(ranking_score),
        "execution_score": float(execution_score),

        "dynamic_threshold": float(dynamic_threshold),
        "gate_mode": str(gate_mode),

        "confidence": float(confidence),
        "p_win": float(p_eff),       # UIには縮退後を提示（整合性を優先）
        "p_eff": float(p_eff),
        "p_win_ev": float(p_eff),

        "event_mode": str(event_mode),
        "event_next_high_hours": (float(next_high) if next_high is not None else None),
        "event_last_high_hours": (float(last_high) if last_high is not None else None),

        "why": str(why),
        "veto": list(veto),
        "veto_reasons": list(veto),

        "state_probs": state_probs,
        "ev_contribs": ev_contribs,

        "hold_manage": (hold_manage if isinstance(hold_manage, dict) else {}),
        "_ctx": ctx_out,
    }
    return plan

# End of file




# ===============================
# FX_AI_PRO_v7 FULL PATCH BLOCK
# (non-breaking additive upgrades)
# ===============================

# --- probability normalization ---
def _normalize_probs(p_up: float, p_dn: float):
    try:
        p_up = float(p_up)
        p_dn = float(p_dn)
        s = p_up + p_dn
        if s > 1.0 and s > 0:
            p_up = p_up / s
            p_dn = p_dn / s
        return p_up, p_dn
    except Exception:
        return 0.5, 0.5


# --- volatility regime detection ---
def _volatility_regime(df):
    try:
        atr = (df["High"] - df["Low"]).rolling(14).mean()
        atr_mean = atr.rolling(50).mean()
        if atr.iloc[-1] > 1.4 * atr_mean.iloc[-1]:
            return "VOL_EXPANSION"
        elif atr.iloc[-1] < 0.7 * atr_mean.iloc[-1]:
            return "VOL_COMPRESSION"
        return "VOL_NORMAL"
    except Exception:
        return "VOL_UNKNOWN"


# --- liquidity sweep detection ---
def _detect_liquidity_sweep(df):
    try:
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)

        recent_high = highs.tail(20).max()
        prev_high = highs.tail(40).head(20).max()

        recent_low = lows.tail(20).min()
        prev_low = lows.tail(40).head(20).min()

        sweep_up = recent_high > prev_high * 1.0005
        sweep_down = recent_low < prev_low * 0.9995

        return {"sweep_up": bool(sweep_up), "sweep_down": bool(sweep_down)}
    except Exception:
        return {"sweep_up": False, "sweep_down": False}


# --- no-lookahead helper (use closed bar) ---
def _last_closed(series):
    try:
        return series.iloc[-2]
    except Exception:
        return series.iloc[-1]


# --- improved trend strength model ---
def _trend_strength_v7(adx, slope, atr_expansion):
    try:
        return max(0.0, min(1.0, 0.5*adx + 0.3*slope + 0.2*atr_expansion))
    except Exception:
        return 0.0

# ===============================
# END PATCH
# ===============================



# ---------------------------------------------------------------------
# QUALITY FILTER PATCH (added automatically)
# Purpose: reduce low‑quality trades without reducing win rate.
# It attenuates EV when structure quality is weak.
# This patch is backward compatible and safe.
# ---------------------------------------------------------------------

def _apply_quality_filter(ev_gate, strength, confidence, breakout_ok, hhhl_ok):
    try:
        breakout_flag = 1.0 if breakout_ok else 0.0
        hhhl_flag = 1.0 if hhhl_ok else 0.0

        quality = (
            0.35 * float(strength)
            + 0.30 * float(confidence)
            + 0.20 * float(breakout_flag)
            + 0.15 * float(hhhl_flag)
        )

        quality = max(0.35, min(1.0, quality))
        return float(ev_gate) * quality
    except Exception:
        return ev_gate
