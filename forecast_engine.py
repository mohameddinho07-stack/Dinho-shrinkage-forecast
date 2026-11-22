import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def forecast_lob(df, lob):
    df = df[df["LOB"] == lob].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ABS%"] = df["Abs Hours"] / df["Scheduled Hours"]
    df["Offline%"] = df["Offline"] / df["Scheduled Hours"]
    df["Weekday"] = df["Date"].dt.day_name()

    if len(df) >= 10:
        model = IsolationForest(contamination=0.05, random_state=42)
        mask = model.fit_predict(df[["ABS%","Offline%"]])
        df = df[mask == 1]

    last_date = df["Date"].max()
    rows=[]
    for i in range(1,8):
        d = last_date + pd.Timedelta(days=i)
        wd = d.day_name()
        day_data = df[df["Weekday"]==wd]
        abs_fc = day_data["ABS%"].mean() if len(day_data) else df["ABS%"].mean()
        off_fc = day_data["Offline%"].mean() if len(day_data) else df["Offline%"].mean()
        rows.append({"Date":d.date(),"Day":wd,"ABS Forecast %":round(abs_fc*100,2),"Offline Forecast %":round(off_fc*100,2)})
    return pd.DataFrame(rows)
