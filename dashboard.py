import os
import re
import tempfile
import pandas as pd
import streamlit as st

# Ensure these files exist in the 'sections' folder:
from sections.preprocessing_section import preprocessing_section
from sections.plot_data_section import plot_data_section
# from sections.analytics_section import analytics_section

# Keep your original loader
from src.data_loader import load_data, convert_to_parquet

# --- Page config (MOVED TO THE TOP) ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Universal Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PASSWORD CHECK FUNCTION ---
def check_password():
    """Returns True if the user entered the correct password."""
    
    # Get the password from Streamlit Secrets
    # "APP_PASSWORD" must match the name of the secret you created
    correct_password = st.secrets["APP_PASSWORD"]

    # Ask the user for the password
    # The key "password_input" makes this input unique
    password_attempt = st.text_input("Enter Password", type="password", key="password_input")

    # Check if the password is correct
    if password_attempt == correct_password:
        return True  # Password is correct
    elif password_attempt == "":
        # If no password, just show a prompt
        st.info("Please enter the password in order to access the dashboard.")
        return False
    else:
        # If password is wrong, show an error
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
        "description": "Sc_Com / HyCon: Detect  automatically",
    },
    "Cell Data": {
        "separator": ";",
        "bad_lines_action": "skip",
        "skiprows": 5,
        "description": "Cell Data for Pulse Test",
    },
}

# --- Sc_Com / HyCon Loader ---
def load_sc_com_csv(file_path, drop_ms_option=False):
    """Robust Sc_Com / HyCon CSV loader for dashboard (Hybrid Controller or Tool SCC)."""
    with open(file_path, "r", encoding="latin1", errors="ignore") as f:
        first_line = f.readline().strip()
        if "Hybrid Controller" in first_line:
            timestamp_row = 7
            header_rows_indices = [5, 7]  # combine 6th + 8th rows
            has_ms_column = False
        elif "Tool SCC" in first_line:
            timestamp_row = 11
            header_rows_indices = [9, 10]  # combine 10th + 11th rows
            has_ms_column = True
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
    
    # --- NEW FIX ---
    # Count the number of headers we *actually* found
    num_expected_cols = len(combined_headers)
    # --- END NEW FIX ---


    # Read data below timestamp row
    df = pd.read_csv(
        file_path,
        encoding="latin1",
        sep=";",
        engine="python",
        skiprows=timestamp_row + 1,
        header=None,
        on_bad_lines="skip",
        usecols=range(num_expected_cols)  # <--- THIS IS THE FIX
    )
    
    # This line will now work
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

    # Optional drop ms column
    if has_ms_column and drop_ms_option and len(df.columns) > 1 and df.columns[1].lower().strip() == "ms":
        df.drop(columns=[df.columns[1]], inplace=True)
        st.info("Dropped 'ms' column for Tool SCC.")

    # Parse Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


# --- MAIN APP LOGIC (WRAPPED IN PASSWORD CHECK) ---
if check_password():

    # All the code below this is INDENTED
    
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
    tab_load, tab_preprocess, tab_plot, tab_analytics = st.tabs(
        ["📂 Load Data", "🛠️ Preprocessing", "📈 Plot Data"]  # "📊 Analytics" 
    )

    with tab_load:
        st.header("Upload and Load Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV or Parquet file", type=["csv", "parquet"], key="file_uploader"
        )

        # Sidebar options
        with st.sidebar.container():
            st.subheader("Data Loading Options")
            drop_ms_option = st.checkbox("Drop 'ms' column (for Tool SCC)", value=False)

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

        if st.button("Load Data"):
            if uploaded_file is not None:
                st.session_state.file_type = uploaded_file.name.split(".")[-1].lower()
                st.session_state.uploaded_file = uploaded_file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.file_type}") as tmp_file:
                    uploaded_file.seek(0)
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                try:
                    if st.session_state.file_type == "csv":
                        selected_profile = st.session_state.get("csv_profile_selector", "Hymon")
                        if selected_profile == "Sc_Com / HyCon":
                            df = load_sc_com_csv(tmp_file_path, drop_ms_option=drop_ms_option)
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

    # with tab_analytics:
    #     analytics_section()
