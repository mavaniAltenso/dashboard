# sections/analytics_section.py
import pandas as pd
import streamlit as st

from src.analytics import (
    compute_capacity_ac_features,
    operate_time_bounds,
    estimate_sample_rate_per_device,
    compute_capacity_dc_features,  # placeholder
)

DEVICE_ID_COL = "device-address:uid"


def analytics_section():
    st.header("Capacity Test")

    # ---- AC side ----
    # FIX: Replaced outer st.expander with st.subheader and st.container to avoid nesting error
    st.subheader("AC side using HyCon")
    with st.container():
        capacity_test_ac_section()

    # ---- DC side (coming soon) ----
    with st.expander("DC side (coming soon)", expanded=False):
        capacity_test_dc_section()  # UI skeleton only for now


# -------------------- Shared helpers --------------------

def _auto_pick_from_candidates(cols: list[str], candidates: list[str]) -> str:
    """Return the first exact match candidate present in cols; else empty string."""
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return ""


# -------------------- AC --------------------

def _auto_pick_active_power(cols: list[str]) -> str:
    """
    Keep original priority and extend with a few common aliases.
    """
    # Original priority
    if "PoiPwrAt Analog" in cols:
        return "PoiPwrAt Analog"
    if "PoiPwrAt" in cols:
        return "PoiPwrAt"

    # Additional common variants (exact matches)
    candidates = [
        "P_Active", "ActivePower", "AC_Power", "P_AC", "P_AC_kW",
        "PoiPwrAt Analog.1", "PoiPwrAt.1",
    ]
    picked = _auto_pick_from_candidates(cols, candidates)
    return picked or ""


def capacity_test_ac_section():
    df = st.session_state.get("processed_data")
    if df is None or df.empty:
        st.info("No data available. Please load & preprocess data first.")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Datetime index not found. Please create it in Preprocessing.")
        return

    cols = df.columns.astype(str).tolist()

    # ---- Status & operate value ----
    if "OpStt" in cols:
        status_col = "OpStt"
        st.caption("Status column **OpStt**")
    else:
        status_col = st.selectbox("Status column", options=cols, key="ac_status_col")

    # Build an operate value selector (default to "5: Operate" if present)
    if status_col in df.columns:
        unique_status_vals = df[status_col].astype(str).dropna().unique().tolist()
    else:
        unique_status_vals = []

    if unique_status_vals:
        sorted_vals = sorted(unique_status_vals)
        if "5: Operate" in sorted_vals:
            default_idx = sorted_vals.index("5: Operate")
        else:
            default_idx = 0
        operate_value = st.selectbox(
            "Operate value",
            options=sorted_vals,
            index=default_idx,
            key="ac_operate_value",
            help="Choose the status value that indicates the system is operating.",
        )
    else:
        operate_value = st.selectbox(
            "Operate value",
            options=["5: Operate"],
            index=0,
            key="ac_operate_value",
            help="No status values detected; using default label.",
        )

    # ---- Active Power auto-pick (no prompt when found) ----
    power_col = _auto_pick_active_power(cols)
    if power_col:
        st.caption(f"Active power **{power_col}**")
    else:
        st.warning("Could not auto-detect Active Power column. Please select it.")
        power_col = st.selectbox("Active power (kW)", options=cols, key="ac_power_col")

    # ---- Device & coarse window ----
    devs = sorted(df[DEVICE_ID_COL].astype(str).unique().tolist()) if DEVICE_ID_COL in df.columns else []
    selected_devs = (
        st.multiselect(
            "Devices",
            options=devs,
            default=devs[: min(3, len(devs))],
            key="ac_devices",
        ) if devs else []
    )

    min_t, max_t = df.index.min(), df.index.max()
    c1, c2 = st.columns(2)
    with c1:
        sd = st.date_input(
            "Start Date",
            value=min_t.date(),
            min_value=min_t.date(),
            max_value=max_t.date(),
            key="ac_start_date",
        )
        stime = st.time_input("Start Time", value=min_t.time(), key="ac_start_time")
    with c2:
        ed = st.date_input(
            "End Date",
            value=max_t.date(),
            min_value=min_t.date(),
            max_value=max_t.date(),
            key="ac_end_date",
        )
        etime = st.time_input("End Time", value=max_t.time(), key="ac_end_time")

    start_ts = (
        pd.Timestamp.combine(sd, stime).tz_localize(df.index.tz)
        if df.index.tz is not None else
        pd.Timestamp.combine(sd, stime)
    )
    end_ts = (
        pd.Timestamp.combine(ed, etime).tz_localize(df.index.tz)
        if df.index.tz is not None else
        pd.Timestamp.combine(ed, etime)
    )

    if start_ts >= end_ts:
        st.error("Start time must be before end time.")
        return

    # ---- Pre-filter to devices & coarse window ----
    pre = df.copy()
    if selected_devs and DEVICE_ID_COL in pre.columns:
        pre = pre[pre[DEVICE_ID_COL].astype(str).isin([str(x) for x in selected_devs])]
    pre = pre[(pre.index >= start_ts) & (pre.index <= end_ts)]

    # ---- Find actual Operate interval & restrict ----
    op_min, op_max = operate_time_bounds(pre, status_col=status_col, operate_value=operate_value)
    
    # Check if a valid operate interval was found
    if op_min is None or op_max is None:
        st.warning("No rows with the specified 'Operate' status in the selected device/time window.")
        return
    
    # If a valid interval is found, proceed with slicing and displaying.
    work = pre[(pre.index >= op_min) & (pre.index <= op_max)].copy()

    # FIX: Reverted to st.expander (now non-nested)
    with st.expander("Detected Operate interval", expanded=True):
        st.write(f"**{op_min} → {op_max}** (subset of your coarse selection)")

    # ---- Show Detected Sample Rate (per device) over the Operate slice ----
    sr = estimate_sample_rate_per_device(work)
    # FIX: Reverted to st.expander (now non-nested)
    with st.expander("Sample rate", expanded=True):
        if sr.empty:
            st.info("Could not estimate sampling intervals (insufficient data).")
        else:
            st.dataframe(sr, use_container_width=True)

    # ---- Calculation mode ----
    use_assume_1s = st.checkbox(
        "Sample rate 1 per sec",
        value=False,
        key="ac_dt_mode",
        help="Leave OFF to use actual Δt between timestamps; ON for legacy 1-second assumption.",
    )

    # ---- Compute ----
    if st.button("Compute Calc-PoiEgy & Calc-PoiEgyMtr", type="primary", key="ac_compute"):
        if work.empty:
            st.warning("Operate interval slice is empty.")
            return
        if power_col not in work.columns:
            st.error(f"Active power column '{power_col}' not found in the operate slice.")
            return

        out = compute_capacity_ac_features(
            work,
            status_col=status_col,
            operate_value=operate_value,
            power_col=power_col,
            use_assume_1s=use_assume_1s,
        )

        # Persist ONLY the two new columns back to processed_data for the operate slice
        df_updated = df.copy()
        for col in ["Calc-PoiEgy", "Calc-PoiEgyMtr"]:
            if col in out.columns:
                df_updated.loc[out.index, col] = out[col]
        st.session_state.processed_data = df_updated

        # Summary (per device if available)
        op_mask = work[status_col].astype(str).eq(operate_value) if status_col in work.columns else pd.Series(False, index=work.index)
        p_kw = pd.to_numeric(work[power_col], errors="coerce") if power_col in work.columns else pd.Series(index=work.index, dtype=float)

        if DEVICE_ID_COL in work.columns:
            # Use the computed cumulative meter for a meaningful "max cumulative energy" metric
            max_cum = out["Calc-PoiEgyMtr"].groupby(work[DEVICE_ID_COL]).max(min_count=1)
            summary = (
                pd.DataFrame({
                    "Energy_kWh_total": out["Calc-PoiEgy"].groupby(work[DEVICE_ID_COL]).sum(min_count=1),
                    "Operate_samples": op_mask.groupby(work[DEVICE_ID_COL]).sum(),
                    "Avg_P_kW": p_kw.where(op_mask).groupby(work[DEVICE_ID_COL]).mean(),
                    "Max_CumEnergy_kWh": max_cum,
                    "Operate_Start": op_min,
                    "Operate_End": op_max,
                    "Mode": "Δt" if not use_assume_1s else "assume 1 s",
                })
                .reset_index()
                .rename(columns={DEVICE_ID_COL: "Device"})
            )
        else:
            max_cum = out["Calc-PoiEgyMtr"].max()
            summary = pd.DataFrame({
                "Energy_kWh_total": [out["Calc-PoiEgy"].sum(min_count=1)],
                "Operate_samples": [int(op_mask.sum())],
                "Avg_P_kW": [p_kw.where(op_mask).mean()],
                "Max_CumEnergy_kWh": [max_cum],
                "Operate_Start": [op_min],
                "Operate_End": [op_max],
                "Mode": ["Δt" if not use_assume_1s else "assume 1 s"],
            })

        st.subheader("Summary (Operate interval)")
        st.dataframe(summary, use_container_width=True)

        # Preview new columns (operate slice)
        # This expander is now non-nested and will work.
        with st.expander("Preview computed features (Operate slice)", expanded=False):
            keep_cols = [c for c in ["Calc-PoiEgy", "Calc-PoiEgyMtr", power_col, status_col, DEVICE_ID_COL] if c in df_updated.columns]
            st.dataframe(df_updated.loc[work.index, keep_cols].head(50), use_container_width=True)

        # Download the operate-slice result — embed mode and time range
        mode_tag = "dt" if not use_assume_1s else "1s"
        t0 = op_min.strftime("%Y%m%d_%H%M%S")
        t1 = op_max.strftime("%Y%m%d_%H%M%S")
        csv = df_updated.loc[
            work.index,
            [c for c in ["Calc-PoiEgy", "Calc-PoiEgyMtr", DEVICE_ID_COL] if c in df_updated.columns]
        ].to_csv(index=True).encode("utf-8")

        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"ac_capacity_{mode_tag}_{t0}_{t1}.csv",
            mime="text/csv",
            key="ac_download",
        )
        st.success("'Calc-PoiEgy' & 'Calc-PoiEgyMtr' can be visualised in plotting section")


# -------------------- DC (coming soon) --------------------

def _auto_pick_dc_power(cols: list[str]) -> str:
    """
    Try common DC power column names. Prioritize 'DcTotWatt' as per your data.
    """
    # Highest priority
    if "DcTotWatt" in cols:
        return "DcTotWatt"

    candidates = [
        "P_DC", "DC_Power", "DcPower", "DcPwr", "Pdc", "Pdc_kW", "DcTotkW",
        "DcTotWatt Analog", "DcTotWatt.1",
    ]
    picked = _auto_pick_from_candidates(cols, candidates)
    return picked or ""


def _auto_pick_soc(cols: list[str]) -> str:
    """
    Try typical battery SOC column names.
    """
    candidates = ["Bat.SOCTot", "SOC", "StateOfCharge", "Battery_SOC"]
    return _auto_pick_from_candidates(cols, candidates)


def capacity_test_dc_section():
    df = st.session_state.get("processed_data")
    if df is None or df.empty:
        st.info("No data available. Please load & preprocess data first.")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("Datetime index not found. Please create it in Preprocessing.")
        return

    cols = df.columns.astype(str).tolist()

    # Device filter (unique key to avoid collisions with AC)
    devs = sorted(df[DEVICE_ID_COL].astype(str).unique().tolist()) if DEVICE_ID_COL in df.columns else []
    selected_devs_dc = (
        st.multiselect(
            "Devices (DC)",
            options=devs,
            default=devs[: min(3, len(devs))],
            key="dc_devices",
        ) if devs else []
    )

    # DC power & SOC auto-pick
    dc_power_col = _auto_pick_dc_power(cols)
    soc_col = _auto_pick_soc(cols)

    if dc_power_col:
        st.caption(f"DC power **{dc_power_col}**")
    else:
        st.warning("Could not auto-detect DC Power column. Please select it.")
        dc_power_col = st.selectbox("DC power (W or kW)", options=cols, key="dc_power_col")

    if soc_col:
        st.caption(f"SOC column **{soc_col}** (optional)")
    else:
        soc_col = st.selectbox("SOC column (optional)", options=["(none)"] + cols, index=0, key="dc_soc_col")

    # Time window (unique keys)
    min_t, max_t = df.index.min(), df.index.max()
    c1, c2 = st.columns(2)
    with c1:
        sd_dc = st.date_input(
            "Start Date (DC)",
            value=min_t.date(),
            min_value=min_t.date(),
            max_value=max_t.date(),
            key="dc_start_date",
        )
        stime_dc = st.time_input("Start Time (DC)", value=min_t.time(), key="dc_start_time")
    with c2:
        ed_dc = st.date_input(
            "End Date (DC)",
            value=max_t.date(),
            min_value=min_t.date(),
            max_value=max_t.date(),
            key="dc_end_date",
        )
        etime_dc = st.time_input("End Time (DC)", value=max_t.time(), key="dc_end_time")

    start_ts_dc = (
        pd.Timestamp.combine(sd_dc, stime_dc).tz_localize(df.index.tz)
        if df.index.tz is not None else
        pd.Timestamp.combine(sd_dc, stime_dc)
    )
    end_ts_dc = (
        pd.Timestamp.combine(ed_dc, etime_dc).tz_localize(df.index.tz)
        if df.index.tz is not None else
        pd.Timestamp.combine(ed_dc, etime_dc)
    )

    if start_ts_dc >= end_ts_dc:
        st.error("Start time must be before end time.")
        return

    # Prefilter
    pre_dc = df.copy()
    if selected_devs_dc and DEVICE_ID_COL in pre_dc.columns:
        pre_dc = pre_dc[pre_dc[DEVICE_ID_COL].astype(str).isin([str(x) for x in selected_devs_dc])]
    pre_dc = pre_dc[(pre_dc.index >= start_ts_dc) & (pre_dc.index <= end_ts_dc)]

    # Mode toggle (unique key)
    use_assume_1s_dc = st.checkbox(
        "Sample rate 1 per sec (DC)",
        value=False,
        key="dc_dt_mode",
        help="Leave OFF to use actual Δt between timestamps; ON for legacy 1-second assumption.",
    )

    st.info("🛠️ DC side computations will be implemented soon. This section is a UI placeholder.")
    st.button(
        "Compute DC energy (coming soon)",
        type="primary",
        disabled=True,
        key="dc_compute",
        help="Placeholder: DC computation not yet implemented.",
    )

    # (When ready)
    # out_dc = compute_capacity_dc_features(
    #     pre_dc,
    #     power_col=dc_power_col,
    #     soc_col=None if soc_col == "(none)" else soc_col,
    #     use_assume_1s=use_assume_1s_dc,
    # )
    # Persist/select/preview similar to AC workflow.