import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import re
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
import io
import plotly.express as px

# SECTION 1: DC DATA LOADING FUNCTION


@st.cache_data
def load_and_prep_dc_data(uploaded_file, sep=';', dayfirst=False) -> pd.DataFrame:
    """
    Loads and prepares the DC-side CSV file from an uploaded file object.
    """
    # Read with low_memory=False to prevent mixed-type warnings on large files
    df = pd.read_csv(uploaded_file, sep=sep, dtype=str, engine='python')

    # Clean up potential whitespace in column names
    df.columns = df.columns.str.strip()

    # Basic validation before proceeding
    required_time_cols = ['Date', 'Time']
    if not all(col in df.columns for col in required_time_cols):
         # Fallback: try to find a single datetime column if separate Date/Time don't exist
         raise ValueError(f"DC File must contain {required_time_cols} columns for timestamp parsing.")

    df['Date'] = df['Date'].str.strip()
    df['Time'] = df['Time'].str.strip()
    
    # Handle TZ if it exists
    if 'TZ' in df.columns:
        df['TZ'] = df['TZ'].astype(str).str.strip()
        def extract_tz_hours(tz_str):
            if pd.isna(tz_str): return 0
            m = re.search(r'([+-]?\d{1,3})', tz_str)
            if not m: return 0
            try: return int(m.group(1))
            except: return 0
        df['TZ_offset_h'] = df['TZ'].apply(extract_tz_hours)
    else:
        df['TZ_offset_h'] = 0

    # Create base datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce', dayfirst=dayfirst)
    
    # Apply TZ offset
    # We use a mask to only apply it where we have valid datetimes
    mask = df['Datetime'].notna()
    df.loc[mask, 'Datetime'] = df.loc[mask, 'Datetime'] + pd.to_timedelta(df.loc[mask, 'TZ_offset_h'], unit='h')

    # Clean up
    cols_to_drop = ['Date', 'Time', 'TZ', 'TZ_offset_h']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Remove rows with invalid timestamps and set index
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    return df


# SECTION 2: CONSOLIDATED HELPER FUNCTIONS


def _check_cadence(dt_s: pd.Series, expected_seconds: Optional[float], rtol: float = 0.02, atol: float = 0.5) -> dict:
    x = dt_s.dropna().to_numpy(dtype=float)
    x = x[x > 0]
    if x.size == 0: return dict(is_regular=False, dt_median=np.nan, dt_p95=np.nan, frac_off=1.0)
    dt_median = float(np.median(x))
    dt_p95 = float(np.quantile(x, 0.95))
    if expected_seconds is None or np.isnan(expected_seconds):
        tol = max(abs(dt_median) * rtol, atol)
        frac_off = float((np.abs(x - dt_median) > tol).mean())
        is_regular = frac_off <= 0.05
    else:
        tol = max(abs(expected_seconds) * rtol, atol)
        frac_off = float((np.abs(x - expected_seconds) > tol).mean())
        is_regular = (frac_off <= 0.05) and (abs(dt_median - expected_seconds) <= tol)
    return dict(is_regular=is_regular, dt_median=dt_median, dt_p95=dt_p95, frac_off=frac_off)

def _strip_spaces(s: str) -> str:
    if not isinstance(s, str): return s
    return s.replace('\u00A0', '').replace('\u202F', '').replace(' ', '').strip()

def _classify_value(s: str):
    if s is None or s == '': return 'other'
    has_comma = ',' in s
    has_dot = '.' in s
    if has_comma and has_dot: return 'EU' if s.rfind(',') > s.rfind('.') else 'US'
    if has_comma: return 'comma_only'
    if has_dot: return 'dot_only'
    if re.fullmatch(r'[+-]?\d+', s): return 'int'
    return 'other'

def _convert_value(s: str, preference: str):
    if s is None or (isinstance(s, float) and pd.isna(s)): return np.nan
    if not isinstance(s, str): return s
    s0 = _strip_spaces(s)
    if s0 == '' or s0.lower() in ('nan', 'none', 'null'): return np.nan
    kind = _classify_value(s0)
    if kind == 'EU':
        try: return float(s0.replace('.', '').replace(',', '.'))
        except: return np.nan
    if kind == 'US':
        try: return float(s0.replace(',', ''))
        except: return np.nan
    if kind == 'comma_only':
        if preference == 'EU':
            try: return float(s0.replace(',', '.'))
            except: return np.nan
        if preference == 'US':
            try: return float(s0.replace(',', ''))
            except: return np.nan
        last_grp = s0.split(',')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split(',')) >= 2:
             try: return float(s0.replace(',', ''))
             except: return np.nan
        try: return float(s0.replace(',', '.'))
        except: return np.nan
    if kind == 'dot_only':
        if preference == 'US':
             try: return float(s0)
             except: return np.nan
        if preference == 'EU':
             try: return float(s0.replace('.', ''))
             except: return np.nan
        last_grp = s0.split('.')[-1]
        if last_grp.isdigit() and len(last_grp) == 3 and len(s0.split('.')) >= 2:
             try: return float(s0.replace('.', ''))
             except: return np.nan
        try: return float(s0)
        except: return np.nan
    if kind == 'int':
        try: return float(s0)
        except: return np.nan
    return np.nan

@st.cache_data
def convert_mixed_numeric_columns(df_in: pd.DataFrame, exclude: set = None, verbose: bool = True) -> pd.DataFrame:
    df_out = df_in.copy()
    exclude = set() if exclude is None else set(exclude)
    diagnostics = {}
    for col in df_out.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df_out[col]): continue
        s = df_out[col].astype(str)
        if not s.str.contains(r'\d', regex=True).any(): continue
        s_clean = s.map(_strip_spaces)
        kinds = s_clean.map(_classify_value)
        eu_votes = int((kinds == 'EU').sum())
        us_votes = int((kinds == 'US').sum())
        preference = 'EU' if eu_votes > us_votes else ('US' if us_votes > eu_votes else None)
        converted = s_clean.map(lambda x: _convert_value(x, preference))
        if (np.isfinite(converted).sum() / max(len(converted), 1)) < 0.1:
             diagnostics[col] = "Skipped (low valid ratio)"
             continue
        df_out[col] = pd.Series(converted, index=df_out.index, dtype="Float64")
        diagnostics[col] = f"Converted (pref={preference})"
    
    if verbose and diagnostics:
        # Removed nested expander here to avoid Streamlit errors if called inside another expander
        for c, info in diagnostics.items(): st.text(f"- {c}: {info}")
    return df_out


# SECTION 3: DC ANALYZER CLASS


class DcCapacityTestAnalyzer:
    def __init__(self, master_config: dict, df_dc: pd.DataFrame):
        self.config = master_config
        self.df_dc = df_dc.copy()
        self.dfs_by_device = None
        self.dc_rte_summary = None
        self.dc_rte_system_totals = None
        self.dc_system_cumulative_energy = None
        self.dc_system_soc = None

    def run_analysis(self):
        with st.spinner("Preparing data..."):
            self._clean_and_partition_dc_df()
        with st.spinner("Running RTE analysis..."):
            self._run_dc_rte_analysis()
        with st.spinner("Running Energy & SOC analysis..."):
             self._run_dc_cumulative_energy_analysis()
             self._run_dc_soc_analysis()

    def _clean_and_partition_dc_df(self):
        dc_device_col = self.config['dc_device_col']
        # We only convert the columns we actually need to save time/memory
        cols_to_convert = [self.config['dc_power_col'], self.config['dc_soc_col']]
        # Ensure they exist before trying to convert
        cols_to_convert = [c for c in cols_to_convert if c in self.df_dc.columns]
        
        if cols_to_convert:
             # We use a temporary dataframe for conversion to avoid converting the whole huge file
             temp_df = self.df_dc[cols_to_convert].copy()
             temp_df = convert_mixed_numeric_columns(temp_df, verbose=False)
             # Assign back to main df
             for c in cols_to_convert:
                 self.df_dc[c] = temp_df[c]

        if dc_device_col not in self.df_dc.columns:
             raise KeyError(f"Device column '{dc_device_col}' not found.")

        self.dfs_by_device = {dev: g for dev, g in self.df_dc.groupby(dc_device_col)}

    def _run_dc_rte_analysis(self):
        rows = []
        t_ch_s, t_ch_e = self.config['charge_start'], self.config['charge_end']
        t_dis_s, t_dis_e = self.config['discharge_start'], self.config['discharge_end']
        P_COL = self.config['dc_power_col']
        
        for dev, d in self.dfs_by_device.items():
            if P_COL not in d.columns: continue
            dd = d.sort_index().copy()
            # Handle Watts vs kW
            scale = 1000.0 if self.config.get('dc_is_power_in_watts', False) else 1.0
            P = dd[P_COL].fillna(0.0).to_numpy() / scale

            if not self.config.get('dc_discharge_positive', True): P = -P
            dd["P"] = P

            # Calculate dt in seconds for integration
            dd["dt_s"] = dd.index.to_series().diff().dt.total_seconds().fillna(method='bfill')
            # Filter out any bad dt (negative or zero if dupes exist)
            dd = dd[dd["dt_s"] > 0]

            d_ch = dd[(dd.index >= t_ch_s) & (dd.index <= t_ch_e)]
            d_dis = dd[(dd.index >= t_dis_s) & (dd.index <= t_dis_e)]

            E_ch, E_dis = 0.0, 0.0
            # Trapezoidal integration for accuracy on irregular data
            if not d_ch.empty:
                 E_ch = np.trapz((-d_ch["P"]).clip(lower=0), x=d_ch.index.astype(np.int64)/1e9) / 3600.0
            if not d_dis.empty:
                 E_dis = np.trapz((d_dis["P"]).clip(lower=0), x=d_dis.index.astype(np.int64)/1e9) / 3600.0

            eta = (E_dis / E_ch) if E_ch > self.config.get('rte_min_charge_kwh', 0.01) else np.nan
            rows.append({"Device": dev, "E_in": E_ch, "E_out": E_dis, "RTE": eta})

        self.dc_rte_summary = pd.DataFrame(rows).sort_values("Device")
        self.dc_rte_system_totals = {
            "Total_E_in": self.dc_rte_summary["E_in"].sum(),
            "Total_E_out": self.dc_rte_summary["E_out"].sum(),
            "System_RTE": (self.dc_rte_summary["E_out"].sum() / self.dc_rte_summary["E_in"].sum()) if self.dc_rte_summary["E_in"].sum() > 0 else np.nan
        }

    def _run_dc_cumulative_energy_analysis(self):
        # System-level power is sum of all devices at each timestamp
        P_COL = self.config['dc_power_col']
        scale = 1000.0 if self.config.get('dc_is_power_in_watts', False) else 1.0
        
        # Pivot to get a wide dataframe: index=Time, columns=Devices, values=Power
        power_wide = self.df_dc.pivot_table(index='Datetime', columns=self.config['dc_device_col'], values=P_COL, aggfunc='first')
        power_wide = power_wide.fillna(0.0).astype(float) / scale
        
        if not self.config.get('dc_discharge_positive', True):
             power_wide = -power_wide
             
        P_system = power_wide.sum(axis=1).sort_index()
        
        # Integrate system power over time
        t_s = P_system.index.astype(np.int64) / 1e9
        # Cumulative trapezoidal integration
        e_cum_joules = np.concatenate([[0], \
             np.cumsum(0.5 * (P_system.values[:-1] + P_system.values[1:]) * np.diff(t_s))])
        
        self.dc_system_cumulative_energy = pd.Series(data=e_cum_joules/3600.0, index=P_system.index)

    def _run_dc_soc_analysis(self):
        SOC_COL = self.config['dc_soc_col']
        if SOC_COL not in self.df_dc.columns: return
        
        soc_wide = self.df_dc.pivot_table(index='Datetime', columns=self.config['dc_device_col'], values=SOC_COL, aggfunc='first')
        soc_wide = soc_wide.astype(float)
        
        # Handle % vs fraction
        if not self.config.get('dc_is_soc_percent', True):
             soc_wide = soc_wide * 100.0
             
        # Simple average across all reporting devices at each timestamp
        self.dc_system_soc = soc_wide.mean(axis=1).sort_index()


# SECTION 4: PLOTTING FUNCTIONS (UPDATED LEGENDS)


def get_dc_efficiency_bar_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    summ = analyzer.dc_rte_summary
    if summ is None or summ.empty: 
        return fig.update_layout(title="No data.")

    sys_rte = analyzer.dc_rte_system_totals.get('System_RTE')
    
    # --- Start of Enhancements ---
    
    # 1. Split data into above/below average for conditional coloring
    if pd.notna(sys_rte):
        summ_above = summ[summ['RTE'] >= sys_rte]
        summ_below = summ[summ['RTE'] < sys_rte]
    else:
        # Fallback if no system average
        summ_above = summ
        summ_below = pd.DataFrame(columns=summ.columns)

    # 2. Define a richer hover template
    hovertemp = (
        "<b>Device:</b> %{x}<br>"
        "<b>RTE:</b> %{y:.2f}%<br>"
        "<b>Energy In:</b> %{customdata[0]:.1f} kWh<br>"
        "<b>Energy Out:</b> %{customdata[1]:.1f} kWh"
        "<extra></extra>" # Hides the 'trace' info
    )

    # 3. Add separate traces for "good" and "bad"
    fig.add_trace(go.Bar(
        x=summ_above['Device'], y=summ_above['RTE']*100, name='Above/On System Avg',
        text=(summ_above['RTE']*100).apply(lambda x: f"{x:.1f}%"), 
        textposition='auto',
        marker_color='#2ca02c', # Green for good
        customdata=summ_above[['E_in', 'E_out']],
        hovertemplate=hovertemp
    ))
    
    fig.add_trace(go.Bar(
        x=summ_below['Device'], y=summ_below['RTE']*100, name='Below System Avg',
        text=(summ_below['RTE']*100).apply(lambda x: f"{x:.1f}%"), 
        textposition='auto',
        marker_color='#ff7f0e', # Orange for warning
        customdata=summ_below[['E_in', 'E_out']],
        hovertemplate=hovertemp
    ))
    
    # --- End of Enhancements ---

    if pd.notna(sys_rte):
        fig.add_hline(y=sys_rte*100, line_dash="dash", line_color="#d62728", # Red
                      annotation_text=f"System Avg: {sys_rte*100:.1f}%", 
                      annotation_position="bottom right")

    fig.update_layout(
        title="Per-Device DC Efficiency (RTE) vs. System Average", # Updated title
        xaxis_title="Device ID", yaxis_title="RTE (%)",
        template="plotly_white",
        # Updated legend, positioned at the bottom
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def get_dc_energy_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    e_data = analyzer.dc_system_cumulative_energy
    if e_data is None or e_data.empty: return fig.update_layout(title="No energy data.")
    
    fig.add_trace(go.Scatter(
        x=e_data.index, y=e_data.values, mode='lines', name='System DC Energy (Net)',
        line=dict(color='#1f77b4', width=2.5)
    ))
    fig.update_layout(
        title="System Cumulative Net DC Energy", xaxis_title="Time", yaxis_title="Energy (kWh)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig

def get_dc_soc_plot(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    fig = go.Figure()
    s_data = analyzer.dc_system_soc
    if s_data is None or s_data.empty: return fig.update_layout(title="No SOC data.")
    
    fig.add_trace(go.Scatter(
        x=s_data.index, y=s_data.values, mode='lines', name='Avg System SOC',
        line=dict(color='#2ca02c', width=2.5)
    ))
    fig.update_layout(
        title="Average System SOC", xaxis_title="Time", yaxis_title="SOC (%)",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
    )
    return fig
import plotly.express as px

def get_dc_efficiency_treemap(analyzer: DcCapacityTestAnalyzer) -> go.Figure:
    summ = analyzer.dc_rte_summary
    if summ is None or summ.empty or summ['E_in'].sum() == 0: 
        return go.Figure().update_layout(title="No data for Treemap.")

    # We need a label for the root of the treemap
    summ_copy = summ.copy()
    summ_copy['all_devices'] = 'All Devices' # This will be the parent node
    
    # Make RTE a percentage for easier formatting and coloring
    summ_copy['RTE_pct'] = summ_copy['RTE'] * 100
    
    # Use the system average as the midpoint for the color scale
    sys_rte_pct = analyzer.dc_rte_system_totals.get('System_RTE', 0.95) * 100

    fig = px.treemap(
        summ_copy,
        path=['all_devices', 'Device'], # Defines the hierarchy
        values='E_in',                 # Size of rectangles
        color='RTE_pct',               # Color of rectangles
        title="Device Efficiency (Color) vs. Energy Throughput (Size)",
        color_continuous_scale='RdYlGn', # Red -> Yellow -> Green scale
        color_continuous_midpoint=sys_rte_pct # Center color on system avg
    )
    
    # Custom text and hover info
    fig.update_traces(
        texttemplate=(
            "<b>%{label}</b><br>"
            "RTE: %{color:.1f}%<br>"
            "E_in: %{value:,.0f} kWh"
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "RTE: %{color:.1f}%<br>"
            "Energy In: %{value:,.0f} kWh<br>"
            "<extra></extra>"
        )
    )
    
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

# SECTION 5: STREAMLIT APP

st.set_page_config(layout="wide", page_title="DC Capacity Analyzer")
st.title("🔋 BESS DC-Side Capacity Test Analyzer")

# --- Session State Init ---
if 'dc_analyzer' not in st.session_state: st.session_state.dc_analyzer = None
if 'dc_df' not in st.session_state: st.session_state.dc_df = None

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload DC Data File (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load file once
    if st.session_state.dc_df is None:
        with st.spinner("Loading DC data file..."):
            try:
                st.session_state.dc_df = load_and_prep_dc_data(uploaded_file)
                st.success("File loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                st.stop()

    df = st.session_state.dc_df
    all_cols = df.columns.tolist()

    # --- Data Preview ---
    with st.expander("Show Data Preview (First 10 Rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    # --- Dynamic Column Selection ---
    st.sidebar.subheader("Column Selection")
    
    # Helper to find best default index
    def get_idx(cols, candidates):
        for cand in candidates:
            if cand in cols: return cols.index(cand)
        return 0

    dev_col = st.sidebar.selectbox("Device ID Column", all_cols, index=get_idx(all_cols, ["Device", "Cluster", "String"]))
    pwr_col = st.sidebar.selectbox("DC Power Column", all_cols, index=get_idx(all_cols, ["DcTotWatt", "Power", "DC_Power"]))
    soc_col = st.sidebar.selectbox("SOC Column", all_cols, index=get_idx(all_cols, ["Bat.SOCTot", "SOC", "StateOfCharge"]))

    st.sidebar.subheader("Test Windows")
    d_default = df.index[0].date() if not df.empty else pd.to_datetime("today").date()
    
    c1,c2 = st.sidebar.columns(2)
    with c1: ch_s_d, ch_s_t = st.date_input("Charge Start", d_default), st.time_input("Time", pd.to_datetime("10:00").time(), key="cs")
    with c2: ch_e_d, ch_e_t = st.date_input("Charge End", d_default), st.time_input("Time", pd.to_datetime("12:00").time(), key="ce")
    
    c3,c4 = st.sidebar.columns(2)
    with c3: dis_s_d, dis_s_t = st.date_input("Discharge Start", d_default), st.time_input("Time", pd.to_datetime("13:00").time(), key="ds")
    with c4: dis_e_d, dis_e_t = st.date_input("Discharge End", d_default), st.time_input("Time", pd.to_datetime("15:00").time(), key="de")

    st.sidebar.subheader("Settings")
    is_watts = st.sidebar.checkbox("Power is in Watts (will convert to kW)", True)
    dis_pos = st.sidebar.checkbox("Discharge is Positive Value", True)
    soc_is_pct = st.sidebar.checkbox("SOC is already % (0-100)", True)

    if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
        config = {
            "charge_start": pd.Timestamp(f"{ch_s_d} {ch_s_t}"),
            "charge_end": pd.Timestamp(f"{ch_e_d} {ch_e_t}"),
            "discharge_start": pd.Timestamp(f"{dis_s_d} {dis_s_t}"),
            "discharge_end": pd.Timestamp(f"{dis_e_d} {dis_e_t}"),
            "dc_device_col": dev_col, "dc_power_col": pwr_col, "dc_soc_col": soc_col,
            "dc_is_power_in_watts": is_watts, "dc_discharge_positive": dis_pos, "dc_is_soc_percent": soc_is_pct,
            "rte_min_charge_kwh": 0.1 # minimal threshold
        }
        st.session_state.dc_analyzer = DcCapacityTestAnalyzer(config, df)
        st.session_state.dc_analyzer.run_analysis()

# --- Results Display ---
if st.session_state.dc_analyzer:
    an = st.session_state.dc_analyzer
    totals = an.dc_rte_system_totals
    
    st.divider()
    st.header("Analysis Results")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total DC Energy IN", f"{totals['Total_E_in']:,.1f} kWh")
    m2.metric("Total DC Energy OUT", f"{totals['Total_E_out']:,.1f} kWh")
    rte_val = totals['System_RTE']*100 if pd.notna(totals['System_RTE']) else 0
    m3.metric("System DC RTE", f"{rte_val:.2f} %")

    # Plots
    t1, t2, t3 = st.tabs(["Efficiency by Device", "System Cumulative Energy", "Avg System SOC"])
    with t1:
        st.plotly_chart(get_dc_efficiency_bar_plot(an), use_container_width=True)
        st.dataframe(an.dc_rte_summary.style.format({"E_in": "{:,.1f}", "E_out": "{:,.1f}", "RTE": "{:.2%}"}), use_container_width=True)
    with t2:
        st.plotly_chart(get_dc_energy_plot(an), use_container_width=True)
    with t3:
        st.plotly_chart(get_dc_soc_plot(an), use_container_width=True)

elif uploaded_file is None:
    st.info("Please upload a DC data CSV file to begin.")