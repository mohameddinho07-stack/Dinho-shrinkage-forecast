import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


# ------------------------
# Utilities
# ------------------------

def remove_outliers_iqr(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)

    return series.clip(lower, upper)


def create_features(data):

    data = data.copy()

    data["WeekdayNum"] = data["Date"].dt.weekday
    data["Month"] = data["Date"].dt.month
    data["WeekOfYear"] = (
        data["Date"]
        .dt.isocalendar()
        .week.astype(int)
    )

    data["DayOfMonth"] = (
        data["Date"]
        .dt.day
    )

    data["Lag1"] = (
        data["TARGET"]
        .shift(1)
    )

    data["Rolling7"] = (
        data["TARGET"]
        .rolling(7, min_periods=1)
        .mean()
    )

    data["Rolling14"] = (
        data["TARGET"]
        .rolling(14, min_periods=1)
        .mean()
    )

    return data


def weighted_forecast(series):

    values = list(series)

    if len(values) == 0:
        return 0

    weights = np.arange(
        1,
        len(values) + 1
    )

    return np.average(
        values,
        weights=weights
    )


# ------------------------
# Main
# ------------------------

def forecast_lob(df, lob):

    df = df[
        df["LOB"] == lob
    ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df[
        df["Scheduled Hours"] > 0
    ]

    # Recalculate shrinkage
    df["ABS_RATE"] = (
        df["Abs Hours"]
        /
        df["Scheduled Hours"]
    )

    df["OFFLINE_RATE"] = (
        df["Offline"]
        /
        df["Scheduled Hours"]
    )

    df["TOTAL_RATE"] = (
        df["ABS_RATE"]
        +
        df["OFFLINE_RATE"]
    )

    # Data cleaning
    df["ABS_RATE"] = remove_outliers_iqr(
        df["ABS_RATE"]
    )

    df["OFFLINE_RATE"] = remove_outliers_iqr(
        df["OFFLINE_RATE"]
    )

    df["TOTAL_RATE"] = remove_outliers_iqr(
        df["TOTAL_RATE"]
    )

    df["Weekday"] = (
        df["Date"]
        .dt.day_name()
    )

    # Detect actual operating days
    weekday_counts = (
        df["Weekday"]
        .value_counts()
    )

    working_days = list(
        weekday_counts.index
    )

    num_operating_days = len(
        working_days
    )

    last_date = (
        df["Date"].max()
    )

    # Model selector

    total_days = len(
        df["Date"].unique()
    )

    if total_days < 21:
        selected_model = "historical"

    elif total_days < 60:
        selected_model = "rf"

    else:
        selected_model = "gbr"

    results = []

    current_date = last_date

    generated = 0

    while generated < num_operating_days:

        current_date += pd.Timedelta(days=1)

        day_name = (
            current_date.day_name()
        )

        if day_name not in working_days:
            continue

        day_history = df[
            df["Weekday"] == day_name
        ].copy()

        abs_fc = forecast_metric(
            day_history,
            "ABS_RATE",
            selected_model
        )

        off_fc = forecast_metric(
            day_history,
            "OFFLINE_RATE",
            selected_model
        )

        total_fc = (
            abs_fc + off_fc
        )

        results.append({

            "Date":
            current_date.date(),

            "Day":
            day_name,

            "ABS Forecast":
            round(abs_fc, 4),

            "Offline Forecast":
            round(off_fc, 4),

            "Total Shrinkage":
            round(total_fc, 4),

            "Model":
            selected_model.upper()

        })

        generated += 1

    return pd.DataFrame(
        results
    )


# ------------------------
# Forecast Method
# ------------------------

def forecast_metric(
    data,
    target_col,
    model_type
):

    data = data.copy()

    data["TARGET"] = (
        data[target_col]
    )

    if len(data) < 5:

        return data[
            "TARGET"
        ].median()

    if model_type == "historical":

        return weighted_forecast(
            data["TARGET"].tail(8)
        )

    data = create_features(
        data
    )

    data = data.dropna()

    features = [

        "WeekdayNum",
        "Month",
        "WeekOfYear",
        "DayOfMonth",
        "Lag1",
        "Rolling7",
        "Rolling14"

    ]

    X = data[features]

    y = data["TARGET"]

    if len(X) < 10:

        return weighted_forecast(
            y.tail(8)
        )

    if model_type == "rf":

        model = RandomForestRegressor(

            n_estimators=300,
            max_depth=8,
            random_state=42

        )

    else:

        model = GradientBoostingRegressor(

            n_estimators=300,
            random_state=42

        )

    model.fit(X, y)

    latest = data.iloc[-1]

    future = pd.DataFrame([{

        "WeekdayNum":
        latest["WeekdayNum"],

        "Month":
        latest["Month"],

        "WeekOfYear":
        latest["WeekOfYear"],

        "DayOfMonth":
        latest["DayOfMonth"],

        "Lag1":
        latest["TARGET"],

        "Rolling7":
        data["TARGET"]
        .tail(7)
        .mean(),

        "Rolling14":
        data["TARGET"]
        .tail(14)
        .mean()

    }])

    pred = model.predict(
        future
    )[0]

    return max(
        pred,
        0
    )
