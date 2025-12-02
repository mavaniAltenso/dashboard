import pandas as pd
import streamlit as st
import io
import os
import pyarrow 

# --- NEW: Robust Optimization Function ---
def optimize_dtypes(df):

    for col in df.columns:
        if df[col].dtype == 'object':
            
            # --- STEP 1: robust numeric conversion ---
            try:
                # 1. Clean whitespace from the ends
                df[col] = df[col].astype(str).str.strip()
                
                # 2. Identify rows that actually have content (not empty, not 'nan')
                mask_has_content = (df[col] != '') & (df[col].str.lower() != 'nan') & (df[col].notna())
                content_values = df.loc[mask_has_content, col]
                
                # If the column is totally empty, skip it
                if len(content_values) == 0:
                    continue

                # 3. Create a clean version: Remove commas (US format thousands)
                cleaned = content_values.str.replace(',', '', regex=False)
                
                # 4. Try converting to numeric
                converted = pd.to_numeric(cleaned, errors='coerce')
                
                # 5. Ratio: Valid Numbers / Total Non-Empty Entries
                success_count = converted.notna().sum()
                total_count = len(content_values)
                
                # If > 80% of the *non-empty* data is numeric, convert the whole column
                if total_count > 0 and (success_count / total_count) > 0.8:
                    # Apply transformation to the FULL column
                    df[col] = pd.to_numeric(
                        df[col].str.replace(',', '', regex=False), 
                        errors='coerce'
                    )
                    continue 

            except Exception:
                pass

            # --- STEP 2: Category Optimization ---
            # Only runs if the numeric check above FAILED
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            
            if num_total_values > 0 and (num_unique_values / num_total_values) < 0.5:
                df[col] = df[col].astype('category')
                
    return df


def load_data(file, file_path, separator, bad_lines_action, file_type, skiprows=None):
    try:
        if file_type == 'csv':
            st.info(f"Attempting to load CSV with delimiter: '{separator}', skipping rows: {skiprows if skiprows is not None else 'None'}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                df = pd.read_csv(
                    f,
                    sep=separator,
                    on_bad_lines=bad_lines_action, 
                    engine='python',
                    skiprows=skiprows 
                )
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            st.error("Unsupported file type selected.")
            return None
        
        if df.empty or len(df.columns) <= 1:
            if len(df.columns) == 1 and file_type == 'csv':
                st.error(f"Data parsed into a single column. The separator '{separator}' is likely incorrect.")
            else:
                st.error("The loaded data is either empty or could not be parsed correctly.")
            return None

        # --- Apply Optimization ---
        df = optimize_dtypes(df)
        # ------------------------------------

        return df

    except Exception as e:
        st.error(f"An error occurred while loading the data. Error: {e}")
        return None


def convert_to_parquet(uploaded_file, separator, bad_lines_action, skiprows=None):
    try:
        # Read CSV directly from the uploaded file buffer
        uploaded_file.seek(0)
        csv_df = pd.read_csv(
            uploaded_file, 
            sep=separator, 
            on_bad_lines=bad_lines_action, 
            engine='python',
            skiprows=skiprows 
        )
        
        if csv_df.empty:
            st.warning("The CSV file is empty. Cannot convert to Parquet.")
            return None, None
            
        buffer = io.BytesIO()
        csv_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Determine the new filename
        file_name, _ = os.path.splitext(uploaded_file.name)
        new_file_name = f"{file_name}.parquet"
        
        return buffer.getvalue(), new_file_name

    except Exception as e:
        st.error(f"Failed to convert file to Parquet. Check the CSV profile settings. Error: {e}")
        return None, None