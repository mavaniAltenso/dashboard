import os
import re
import gc  # Garbage Collector
import tempfile
import pandas as pd
import streamlit as st

# --- 1. PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Universal Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. IMPORTS (Now safe to import custom files) ---
# Ensure these files exist in the 'sections' folder:
from sections.preprocessing_section import preprocessing_section
from sections.plot_data_section import plot_data_section

# Keep your original loader
from src.data_loader import load_data

# --- 3. PASSWORD CHECK FUNCTION ---
def check_password():
    """Returns True if the user entered the correct password."""
    # Ensure you have .streamlit/secrets.toml with [APP_PASSWORD] = "your_password"
    if "APP_PASSWORD" not in st.secrets:
        st.error("Secrets not found. Please set up .streamlit/secrets.toml")
        return False

    correct_password = st.secrets["APP_PASSWORD"]
    password_attempt = st.text_input("Enter Password", type="password", key="password_input")

    if password_attempt == correct_password:
        return True
    elif password_attempt == "":
        st.info("Please enter the password in order to access the dashboard.")
        return False
    else:
        st.error("Incorrect password. Please try again.")
        return False



# --- Data Profile Configuration ---
DATA_PROFILES = {
    "Hymon": {
        "separator": ";",
        "bad_lines_action": "skip",
        "skiprows": None,
        "description": "Hymon data (Date, Time, TZ, Device) without metadata.",
    },
    "Sc_Com / HyCon": {
        "separator": ";",
        "bad_lines_action": "skip",
        "skiprows": None,
        "description": "Sc_Com / HyCon: Detect automatically",
    },
    "Cell Data": {
        "separator": ";",
        "bad_lines_action": "skip",
        "skiprows": 5,
        "description": "Cell Data for Pulse Test",
    },
}

# --- Sc_Com / HyCon Loader ---
def load_sc_com_csv(file_path):
    """Robust Sc_Com / HyCon CSV loader. Automatically drops 'ms' column."""
    with open(file_path, "r", encoding="latin1", errors="ignore") as f:
        first_line = f.readline().strip()
        if "Hybrid Controller" in first_line:
            timestamp_row = 7
            header_rows_indices = [5, 7]  # combine 6th + 8th rows
        elif "Tool SCC" in first_line:
            timestamp_row = 11
            header_rows_indices = [9, 10]  # combine 10th + 11th rows
        else:
            st.error("Sc_Com / HyCon: Unknown type. First line must contain 'Hybrid Controller' or 'Tool SCC'.")
            return pd.DataFrame()

    # Read and combine header rows
    header_rows = pd.read_csv(
        file_path,
        encoding="latin1",
        sep=";",
        engine="python",
        skiprows=header_rows_indices[0],
        nrows=len(header_rows_indices),
        header=None,
    )
    combined_headers = header_rows.fillna("").astype(str).agg(" ".join)
    combined_headers = combined_headers.str.replace(r"\s+", " ", regex=True).str.replace("-", "").str.strip()
    
    num_expected_cols = len(combined_headers)

    # Read data below timestamp row
    df = pd.read_csv(
        file_path,
        encoding="latin1",
        sep=";",
        engine="python",
        skiprows=timestamp_row + 1,
        header=None,
        on_bad_lines="skip",
        usecols=range(num_expected_cols)
    )
    
    df.columns = combined_headers

    # Make columns unique
    def make_unique(cols):
        seen = {}
        result = []
        for col in cols:
            col = col.strip()
            if col in seen:
                seen[col] += 1
                result.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                result.append(col)
        return result

    df.columns = make_unique(df.columns)

    # Rename first column to Timestamp
    df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)

    # --- AUTOMATIC DROP LOGIC ---
    # Always check if the second column is 'ms' and drop it
    if len(df.columns) > 1 and df.columns[1].lower().strip() == "ms":
        df.drop(columns=[df.columns[1]], inplace=True)
        # st.info("Automatically dropped 'ms' column.") # Uncomment if you want to see a message

    # Parse Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


# --- Page config ---
st.set_page_config(
    page_title="Universal Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Data visualisation & Analytics")
st.markdown("Dashboard allows diff. data format handling, preprocessing, and visualization.")

# --- Session state ---
if "current_data" not in st.session_state:
    st.session_state.current_data = pd.DataFrame()
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None

# --- Main Tabs ---
# FIX: Removed 'tab_analytics' variable and label
tab_load, tab_preprocess, tab_plot = st.tabs(
    ["📂 Load Data", "🛠️ Preprocessing", "📈 Plot Data"]  
)

with tab_load:
    st.header("Upload and Load Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Parquet file", type=["csv", "parquet"], key="file_uploader"
    )

    # Sidebar options
    with st.sidebar.container():
        st.subheader("Data Loading Options")

        if uploaded_file and uploaded_file.name.lower().endswith(".csv"):
            st.info("CSV Profile Selector")
            profile_name = st.selectbox(
                "Select Data Profile", options=list(DATA_PROFILES.keys()), key="csv_profile_selector"
            )
            profile = DATA_PROFILES[profile_name]
            st.markdown(f"**Description:** _{profile['description']}_")
            st.markdown("---")
            st.caption("Applied Parameters:")
            st.text(f"Delimiter: '{profile['separator'] if profile['separator'] else 'auto'}'")
            st.text(f"Bad Lines Action: '{profile['bad_lines_action']}'")
            st.text(f"Skip Rows: {profile['skiprows'] if profile['skiprows'] else 'None'}")

    # FIX: Indentation moved BACK (Left) so this is NOT in the sidebar
    if st.button("Load Data"):
        # --- Aggressive Memory Cleanup ---
        st.session_state.current_data = pd.DataFrame()
        st.session_state.processed_data = None
        st.cache_data.clear()
        gc.collect()
        # ---------------------------------

        if uploaded_file is not None:
            st.session_state.file_type = uploaded_file.name.split(".")[-1].lower()
            st.session_state.uploaded_file = uploaded_file
            
            # FIX: .write() is now INSIDE the 'with' block
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.file_type}") as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                if st.session_state.file_type == "csv":
                    selected_profile = st.session_state.get("csv_profile_selector", "Hymon")
                    if selected_profile == "Sc_Com / HyCon":
                        df = load_sc_com_csv(tmp_file_path)
                    else:
                        profile = DATA_PROFILES[selected_profile]
                        df = load_data(
                            uploaded_file,
                            tmp_file_path,
                            profile["separator"],
                            profile["bad_lines_action"],
                            st.session_state.file_type,
                            profile["skiprows"],
                        )
                else:
                    df = load_data(uploaded_file, tmp_file_path, None, None, st.session_state.file_type, None)

                if df is not None and not df.empty:
                    # Normalize device column
                    if "device-address:uid" in df.columns:
                        pass
                    elif "Device" in df.columns:
                        df.rename(columns={"Device": "device-address:uid"}, inplace=True)
                    else:
                        source_label = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                        df["device-address:uid"] = source_label

                    st.session_state.current_data = df
                    st.session_state.processed_data = df.copy()
                    st.success("Data loaded successfully!")
                    st.write("First 5 rows of the loaded data:")
                    st.dataframe(st.session_state.current_data.head())
                else:
                    st.error("Failed to load data.")
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        else:
            st.warning("Please upload a file first.")

with tab_preprocess:
    data_to_use = (
        st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.current_data
    )
    preprocessing_section(data_to_use)

with tab_plot:
    # Plot section includes its own export (under the chart) and keeps the chart visible after downloads
    plot_data_section()