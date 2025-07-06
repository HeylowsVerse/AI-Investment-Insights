import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime


def quarter_to_date(q: str) -> pd.Timestamp:
    """Convert a YYYYQ# string to the first day of that quarter."""
    year = int(q[:4])
    quarter = int(q[-1])
    month = 3 * (quarter - 1) + 1
    return pd.Timestamp(datetime(year, month, 1))


def generate_sample_msi() -> pd.DataFrame:
    """Return sample quarterly MSI data for 2024-2025."""
    quarterly_data = {
        "Quarter": ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"],
        "MSI": [2834, 3035, 3214, 3183, 2896],
    }
    df_q = pd.DataFrame(quarterly_data)
    df_q["date"] = df_q["Quarter"].apply(quarter_to_date)
    df_q = df_q.rename(columns={"MSI": "msi"}).drop(columns="Quarter")
    return df_q


def interpolate_monthly(msi_quarterly: pd.DataFrame) -> pd.Series:
    monthly = (
        msi_quarterly.set_index("date")
        .resample("MS")
        .interpolate("linear")
    )
    return monthly["msi"]


def sarimax_forecast(msi_monthly: pd.Series, steps: int = 12):
    model = SARIMAX(
        msi_monthly,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean, pred.conf_int(), res


def simulate_pmi(dates: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(42)
    t = np.linspace(0, 3 * np.pi, len(dates))
    pmi = 50 + 2 * np.sin(t) + np.random.normal(0, 1.5, len(dates))
    return pd.Series(pmi, index=dates, name="pmi")


def _series_from_data(data: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Return a Series with the given name from Series or single-column DataFrame."""
    if isinstance(data, pd.DataFrame):
        if len(data.columns) != 1:
            raise ValueError("PMI input must have a single column")
        data = data.iloc[:, 0]
    return pd.Series(data.values, index=data.index, name=name)


def regression_predict(msi: pd.Series | pd.DataFrame, pmi: pd.Series | pd.DataFrame, future_pmi: pd.Series | pd.DataFrame):
    """Predict future MSI values based on PMI."""
    msi = _series_from_data(msi, "msi")
    pmi = _series_from_data(pmi, "pmi")
    future_pmi = _series_from_data(future_pmi, "pmi")

    df = pd.DataFrame({"msi": msi, "pmi": pmi}).dropna()
    X = df[["pmi"]]
    y = df["msi"]

    model = LinearRegression().fit(X, y)

    future_df = future_pmi.to_frame()
    future_df.columns = ["pmi"]
    # Pass raw values to avoid feature name mismatches
    preds = model.predict(future_df.values)

    rmse = mean_squared_error(y, model.predict(X)) ** 0.5
    return preds, model.coef_[0], model.intercept_, rmse


def plot_results(msi, forecast, ci, pmi, reg_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    msi.plot(ax=ax, label="Actual MSI")
    forecast.plot(ax=ax, label="SARIMAX Forecast")
    ax.fill_between(
        ci.index, ci["lower msi"], ci["upper msi"], color="lightblue", alpha=0.3
    )
    reg_pred.plot(ax=ax, label="PMI Regression", linestyle="--")
    ax.set_ylabel("MSI")
    ax.legend()
    ax2 = ax.twinx()
    pmi.plot(ax=ax2, color="gray", alpha=0.4, label="PMI")
    ax2.set_ylabel("PMI")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def main():
    msi_q = generate_sample_msi()
    msi_m = interpolate_monthly(msi_q)
    forecast, ci, _ = sarimax_forecast(msi_m)
    all_dates = msi_m.index.append(ci.index)
    pmi = simulate_pmi(all_dates)
    reg_pred, slope, intercept, rmse = regression_predict(
        msi_m, pmi.loc[msi_m.index], pmi.loc[ci.index]
    )
    reg_pred = pd.Series(reg_pred, index=ci.index, name="PMI Regression")

    print(f"Regression slope: {slope:.3f} intercept: {intercept:.3f} RMSE: {rmse:.3f}")
    plot_results(pd.concat([msi_m, forecast]), forecast, ci, pmi, reg_pred)


if __name__ == "__main__":
    main()
