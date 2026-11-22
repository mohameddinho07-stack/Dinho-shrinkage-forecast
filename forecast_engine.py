import pandas as pd
import numpy as np


def remove_outliers(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return series.clip(lower, upper)


def forecast_lob(df, lob):

    df = df[df["LOB"] == lob].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Scheduled Hours"] > 0]

    # Percentages as decimal
    df["ABS"] = (
        df["Abs Hours"]
        / df["Scheduled Hours"]
    )

    df["OFFLINE"] = (
        df["Offline"]
        / df["Scheduled Hours"]
    )

    df["TOTAL"] = (
        df["ABS"]
        + df["OFFLINE"]
    )

    df["Weekday"] = df["Date"].dt.day_name()

    # Remove Extreme Outliers
    df["ABS"] = remove_outliers(df["ABS"])
    df["OFFLINE"] = remove_outliers(df["OFFLINE"])
    df["TOTAL"] = remove_outliers(df["TOTAL"])

    # Detect working days automatically
    weekday_counts = (
        df["Weekday"]
        .value_counts()
        .to_dict()
    )

    working_days = []

    for day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]:

        if weekday_counts.get(day, 0) >= 3:
            working_days.append(day)

    if not working_days:
        working_days = list(
            df["Weekday"].unique()
        )

    last_date = df["Date"].max()

    forecast_rows = []

    current_date = last_date

    target_days = len(working_days)

    generated = 0

    while generated < target_days:

        current_date += pd.Timedelta(days=1)

        day_name = current_date.day_name()

        if day_name not in working_days:
            continue

        historical = df[
            df["Weekday"] == day_name
        ].copy()

        historical = historical.sort_values(
            "Date"
        )

        # Last 8 occurrences
        historical = historical.tail(8)

        abs_forecast = historical["ABS"].median()

        offline_forecast = (
            historical["OFFLINE"]
            .median()
        )

        total_forecast = (
            abs_forecast
            + offline_forecast
        )

        forecast_rows.append({

            "Date":
            current_date.date(),

            "Day":
            day_name,

            "ABS Forecast":
            round(abs_forecast, 4),

            "Offline Forecast":
            round(offline_forecast, 4),

            "Total Shrinkage":
            round(total_forecast, 4)

        })

        generated += 1

    result = pd.DataFrame(
        forecast_rows
    )

    return result
