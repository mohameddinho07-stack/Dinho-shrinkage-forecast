import streamlit as st
import pandas as pd
from forecast_engine import forecast_lob

st.set_page_config(page_title="AI Shrinkage Forecast", layout="wide")
st.title("AI Shrinkage Forecast Tool")

uploaded_file = st.file_uploader("Upload File", type=["xlsx", "xlsm"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = ["LOB","Date","Scheduled Hours","Abs Hours","Offline"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        lobs = sorted(df["LOB"].dropna().unique())
        selected_lob = st.selectbox("Select LOB", lobs)

        if st.button("Generate Forecast"):
            result = forecast_lob(df, selected_lob)
            st.dataframe(result, use_container_width=True)
