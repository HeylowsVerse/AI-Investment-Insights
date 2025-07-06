import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_sample_msi(start="2020-01-01", periods=8):
    """Generate simulated quarterly MSI data."""
    rng = pd.date_range(start=start, periods=periods, freq="QE")
    trend = np.linspace(100, 120, periods)
    noise = np.random.randn(periods) * 2
    return pd.DataFrame({"date": rng, "msi": trend + noise})


def interpolate_monthly(msi_quarterly: pd.DataFrame) -> pd.Series:
    monthly = (
        msi_quarterly.set_index("date")
        .resample("ME")
        .interpolate("linear")
    )
    return monthly["msi"]


def sarimax_forecast(msi_monthly: pd.Series, steps: int = 12):
    model = SARIMAX(msi_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean, pred.conf_int(), res


def simulate_pmi(dates: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(0)
    pmi = 50 + np.sin(np.arange(len(dates)) / 6) * 5 + np.random.randn(len(dates))
    return pd.Series(pmi, index=dates, name="pmi")


def regression_predict(msi: pd.Series, pmi: pd.Series, future_pmi: pd.Series):
    df = pd.DataFrame({"msi": msi, "pmi": pmi}).dropna()
    X = df[["pmi"]]
    y = df["msi"]
    model = LinearRegression().fit(X, y)
    future_df = future_pmi.to_frame()
    future_df.columns = ["pmi"]
    preds = model.predict(future_df)
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
