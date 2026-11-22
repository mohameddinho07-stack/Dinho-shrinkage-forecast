import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


def normalize_series(series):

    median = series.median()

    mad = np.median(
        np.abs(series - median)
    )

    if mad == 0:
        return series

    lower = median - (3 * 1.4826 * mad)
    upper = median + (3 * 1.4826 * mad)

    return series.clip(
        lower=lower,
        upper=upper
    )


def detect_operation(df):

    weekday_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    active_days = []

    for day_number in range(7):

        day_df = df[
            df["Date"].dt.weekday
            == day_number
        ]

        if len(day_df) == 0:
            continue

        avg_hours = (
            day_df["Scheduled Hours"]
            .mean()
        )

        if avg_hours > 0:

            active_days.append(
                weekday_map[day_number]
            )

    return active_days


def confidence_score(record_count):

    score = min(
        95,
        max(
            40,
            round(
                (record_count / 90) * 100
            )
        )
    )

    return score


def forecast_lob(df, lob):

    df = df.copy()

    df = df[
        df["LOB"] == lob
    ]

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["ABS%"] = (
        df["Abs Hours"] /
        df["Scheduled Hours"]
    )

    df["Offline%"] = (
        df["Offline"] /
        df["Scheduled Hours"]
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    record_count = len(df)

    operation_days = detect_operation(df)

    operation_type = (
        f"{len(operation_days)} Day Operation"
    )

    df["Weekday"] = (
        df["Date"]
        .dt.day_name()
    )

    outlier_count = 0

    if len(df) >= 10:

        iso = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        preds = iso.fit_predict(
            df[
                [
                    "ABS%",
                    "Offline%"
                ]
            ]
        )

        outlier_count = int(
            (preds == -1).sum()
        )

        df = df.assign(
            Outlier=preds
        )

    df["ABS%"] = normalize_series(
        df["ABS%"]
    )

    df["Offline%"] = normalize_series(
        df["Offline%"]
    )

    last_date = (
        df["Date"].max()
    )

    forecast_rows = []

    for i in range(1, 8):

        forecast_date = (
            last_date
            + pd.Timedelta(days=i)
        )

        weekday = (
            forecast_date.day_name()
        )

        day_df = df[
            df["Weekday"]
            == weekday
        ]

        if len(day_df) == 0:

            abs_fc = (
                df["ABS%"].mean()
            )

            off_fc = (
                df["Offline%"].mean()
            )

        else:

            weights = np.arange(
                1,
                len(day_df) + 1
            )

            abs_fc = np.average(
                day_df["ABS%"],
                weights=weights
            )

            off_fc = np.average(
                day_df["Offline%"],
                weights=weights
            )

        forecast_rows.append(
            {
                "Date":
                    forecast_date.date(),
                "Day":
                    weekday,
                "Operating Day":
                    "Yes"
                    if weekday in operation_days
                    else "No",
                "ABS Forecast %":
                    round(
                        abs_fc * 100,
                        2
                    ),
                "Offline Forecast %":
                    round(
                        off_fc * 100,
                        2
                    ),
                "Total Shrinkage %":
                    round(
                        (
                            abs_fc +
                            off_fc
                        ) * 100,
                        2
                    )
            }
        )

    forecast = pd.DataFrame(
        forecast_rows
    )

    forecast.attrs[
        "operation_type"
    ] = operation_type

    forecast.attrs[
        "working_days"
    ] = operation_days

    forecast.attrs[
        "confidence"
    ] = confidence_score(
        record_count
    )

    forecast.attrs[
        "outliers"
    ] = outlier_count

    forecast.attrs[
        "records"
    ] = record_count

    return forecast
