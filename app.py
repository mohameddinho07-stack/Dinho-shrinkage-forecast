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

        st.success(
            f"{len(lobs)} LOBs detected"
        )

        if st.button("Generate Forecast For All LOBs"):

            all_results = []

            for lob in lobs:

                try:

                    forecast = forecast_lob(
                        df,
                        lob
                    )

                    if len(forecast) > 0:

                        forecast.insert(
                            0,
                            "LOB",
                            lob
                        )

                        all_results.append(
                            forecast
                        )

                except Exception as e:

                    st.warning(
                        f"Error with {lob}: {e}"
                    )

            if len(all_results) > 0:

                result = pd.concat(
                    all_results,
                    ignore_index=True
                )

                st.subheader(
                    "Forecast Results"
                )

                st.dataframe(
                    result.style.format({
                        "ABS Forecast": "{:.2%}",
                        "Offline Forecast": "{:.2%}",
                        "Total Shrinkage": "{:.2%}"
                    }),
                    use_container_width=True
                )

                if "Total Shrinkage" in result.columns:

                    st.subheader(
                        "Shrinkage Trend"
                    )

                    chart_df = result.copy()

                    chart_df["Date"] = pd.to_datetime(
                        chart_df["Date"]
                    )

                    st.line_chart(
                        chart_df.pivot_table(
                            index="Date",
                            columns="LOB",
                            values="Total Shrinkage"
                        )
                    )

                export_df = result.copy()

                csv = export_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Forecast",
                    data=csv,
                    file_name="All_LOBs_Forecast.csv",
                    mime="text/csv"
                )

            else:

                st.error(
                    "No forecast results generated"
                )
