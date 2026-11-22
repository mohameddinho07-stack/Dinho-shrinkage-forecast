import pandas as pd
import numpy as np


def remove_outliers_iqr(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return series.clip(lower, upper)


def weighted_forecast(values):

    if len(values) == 0:
        return 0

    values = list(values)

    weights = np.arange(1, len(values) + 1)

    return np.average(values, weights=weights)


def forecast_lob(df, lob):

    df = df[df["LOB"] == lob].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Scheduled Hours"] > 0]

    # Recalculate from source
    df["ABS_RATE"] = (
        df["Abs Hours"] /
        df["Scheduled Hours"]
    )

    df["OFFLINE_RATE"] = (
        df["Offline"] /
        df["Scheduled Hours"]
    )

    df["TOTAL_RATE"] = (
        df["ABS_RATE"] +
        df["OFFLINE_RATE"]
    )

    # Remove bad data
    df["ABS_RATE"] = remove_outliers_iqr(
        df["ABS_RATE"]
    )

    df["OFFLINE_RATE"] = remove_outliers_iqr(
        df["OFFLINE_RATE"]
    )

    df["TOTAL_RATE"] = remove_outliers_iqr(
        df["TOTAL_RATE"]
    )

    df["Weekday"] = df["Date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    # Detect real operating days
    working_days = []

    weekday_counts = (
        df["Weekday"]
        .value_counts()
        .to_dict()
    )

    max_count = max(
        weekday_counts.values()
    )

    for day in weekday_order:

        cnt = weekday_counts.get(day, 0)

        if cnt >= (max_count * 0.5):
            working_days.append(day)

    if len(working_days) == 0:

        working_days = list(
            df["Weekday"].unique()
        )

    last_date = df["Date"].max()

    future_rows = []

    current_date = last_date

    generated = 0

    target_days = len(working_days)

    while generated < target_days:

        current_date += pd.Timedelta(days=1)

        day_name = current_date.day_name()

        if day_name not in working_days:
            continue

        day_history = (
            df[
                df["Weekday"] == day_name
            ]
            .sort_values("Date")
            .copy()
        )

        # Latest history only
        day_history = day_history.tail(8)

        abs_fc = weighted_forecast(
            day_history["ABS_RATE"]
        )

        offline_fc = weighted_forecast(
            day_history["OFFLINE_RATE"]
        )

        total_fc = (
            abs_fc +
            offline_fc
        )

        future_rows.append({

            "Date":
                current_date.date(),

            "Day":
                day_name,

            "ABS Forecast":
                round(abs_fc, 4),

            "Offline Forecast":
                round(offline_fc, 4),

            "Total Shrinkage":
                round(total_fc, 4)

        })

        generated += 1

    result = pd.DataFrame(
        future_rows
    )

    return result
