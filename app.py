import streamlit as st
import pandas as pd
from forecast_engine import forecast_lob

st.set_page_config(
    page_title="AI Shrinkage Forecast",
    layout="wide"
)

st.title("AI Shrinkage Forecast Tool")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xlsm", "xls"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    required_cols = [
        "LOB",
        "Date",
        "Scheduled Hours",
        "Abs Hours",
        "Offline"
    ]

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:

        st.error(
            f"Missing columns: {', '.join(missing)}"
        )

    else:

        lobs = sorted(
            df["LOB"].dropna().unique()
        )

        selected_lob = st.selectbox(
            "Select LOB",
            lobs
        )

        if st.button("Generate Forecast"):

            result = forecast_lob(
                df,
                selected_lob
            )

            st.subheader("Forecast Result")

            # Display percentages correctly
          st.dataframe(
    result.style.format({
        "ABS Forecast": "{:.2%}",
        "Offline Forecast": "{:.2%}",
        "Total Shrinkage": "{:.2%}"
    }),
    use_container_width=True
)

            # Trend chart
            if "Total Shrinkage" in result.columns:

                st.subheader("Shrinkage Trend")

                chart_df = result.copy()

                chart_df["Date"] = pd.to_datetime(
                    chart_df["Date"]
                )

                st.line_chart(
                    chart_df.set_index("Date")[
                        ["Total Shrinkage %"]
                    ]
                )

            # Download CSV
            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Forecast",
                data=csv,
                file_name="Shrinkage_Forecast.csv",
                mime="text/csv"
            )
