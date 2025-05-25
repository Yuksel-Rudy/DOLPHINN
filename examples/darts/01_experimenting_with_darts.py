from darts import TimeSeries
from darts.utils.utils import generate_index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.datasets import AirPassengersDataset

x = np.linspace(0, 2*np.pi, 100)
s = np.sin(x)
c = np.cos(x)

dates = generate_index("2020-01-01", length=len(x), freq="D")

df = pd.DataFrame({"sine": s, "cosine": c, "time": dates})

series = TimeSeries.from_dataframe(df, time_col="time")
series.plot()
plt.show()


series = AirPassengersDataset().load()  
train, val = series.split_before(pd.Timestamp("19580101"))
train.plot()
val.plot()
plt.show()

from darts.models import AutoARIMA, ExponentialSmoothing, Theta
from darts.metrics import mape
def eval_model(model):
    model.fit(train)
    forecast = model.predict(len(val))
    print(f"model {model} obtains MAPE: {mape(val, forecast)}")
    forecast.plot()

train.plot()
val.plot()

eval_model(AutoARIMA())
eval_model(ExponentialSmoothing())
eval_model(Theta())
plt.show()

thetas = 2 - np.linspace(-10, 10, 50)
best_mape = float("inf")
best_theta = 0

for theta in thetas:
    model = Theta(theta)
    model.fit(train)
    pred_theta = model.predict(len(val))
    res = mape(val, pred_theta)

    if res < best_mape:
        best_mape = res
        best_theta = theta

best_theta_model = Theta(best_theta)
best_theta_model.fit(train)
pred_best_theta = best_theta_model.predict(len(val))

print(f"Lowest MAPE is: {mape(val, pred_best_theta):.2f}, with theta = {best_theta}.")

train.plot(label="train")
val.plot(label="true")
pred_best_theta.plot(label="prediction")
plt.legend()
plt.show()

# Historical Forecast
hfc_params = {
    "series": series,
    "start": pd.Timestamp(
        "1956-01-01"
    ),  # can also be a float for the fraction of the series to start at
    "forecast_horizon": 3,
    "verbose": True,
}
historical_fcast_theta = best_theta_model.historical_forecasts(
    last_points_only=True, **hfc_params
)

series.plot(label="data")
historical_fcast_theta.plot(label="backtest 3-months ahead forecast (Theta)")
print(f"MAPE = {mape(series, historical_fcast_theta):.2f}%")