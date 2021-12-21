from pickle import FALSE
from zipfile import ZipFile
import urllib.request as urllib
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pandas.core.frame import DataFrame
import pmdarima as pm
from pmdarima.arima.utils import ndiffs, nsdiffs
import statsmodels.tsa.arima.model as m
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def read_df():
    print('[Reading file]')
    csv_path = "src/mly532.csv"
    df = pd.read_csv(csv_path, names=['year', 'month', 'temp'], header=16, usecols=[0, 1, 2], index_col=0, parse_dates=[[0, 1]], skiprows=2, skipfooter=10, engine='python')

    print('[File read]')
    return df

def build_model(data_frame: DataFrame, auto: bool):
    print('[Building model]')

    if auto:
        print('[Finding parameters]')
        # ARIMA(2,0,2)(1,0,1)[12] intercept   : AIC=2907.243, Time=13.19 sec
        # ARIMA(0,0,0)(0,0,0)[12] intercept   : AIC=4233.554, Time=0.06 sec
        # ARIMA(1,0,0)(1,0,0)[12] intercept   : AIC=3240.299, Time=4.07 sec
        # ARIMA(0,0,1)(0,0,1)[12] intercept   : AIC=3550.434, Time=1.99 sec
        # ARIMA(0,0,0)(0,0,0)[12]             : AIC=6323.297, Time=0.09 sec
        # ARIMA(2,0,2)(0,0,1)[12] intercept   : AIC=2905.260, Time=10.23 sec
        # ARIMA(2,0,2)(0,0,0)[12] intercept   : AIC=2918.394, Time=1.25 sec
        # ARIMA(2,0,2)(0,0,2)[12] intercept   : AIC=2933.561, Time=24.18 sec
        # ARIMA(2,0,2)(1,0,0)[12] intercept   : AIC=inf, Time=9.48 sec
        # ARIMA(2,0,2)(1,0,2)[12] intercept   : AIC=inf, Time=35.66 sec
        # ARIMA(1,0,2)(0,0,1)[12] intercept   : AIC=3404.582, Time=2.99 sec
        # ARIMA(2,0,1)(0,0,1)[12] intercept   : AIC=3219.072, Time=8.15 sec
        # ARIMA(3,0,2)(0,0,1)[12] intercept   : AIC=inf, Time=10.32 sec
        # ARIMA(2,0,3)(0,0,1)[12] intercept   : AIC=2872.378, Time=9.52 sec
        # ARIMA(2,0,3)(0,0,0)[12] intercept   : AIC=inf, Time=1.82 sec
        # ARIMA(2,0,3)(1,0,1)[12] intercept   : AIC=inf, Time=12.12 sec
        # ARIMA(2,0,3)(0,0,2)[12] intercept   : AIC=2874.057, Time=24.74 sec
        # ARIMA(2,0,3)(1,0,0)[12] intercept   : AIC=inf, Time=10.39 sec
        # ARIMA(2,0,3)(1,0,2)[12] intercept   : AIC=inf, Time=28.29 sec
        # ARIMA(1,0,3)(0,0,1)[12] intercept   : AIC=3379.934, Time=4.15 sec
        # ARIMA(3,0,3)(0,0,1)[12] intercept   : AIC=2905.880, Time=10.44 sec
        # ARIMA(2,0,4)(0,0,1)[12] intercept   : AIC=inf, Time=12.89 sec
        # ARIMA(1,0,4)(0,0,1)[12] intercept   : AIC=3356.046, Time=10.71 sec
        # ARIMA(3,0,4)(0,0,1)[12] intercept   : AIC=2892.519, Time=13.98 sec
        # ARIMA(2,0,3)(0,0,1)[12]             : AIC=3538.128, Time=4.17 sec
        model = pm.auto_arima(data_frame,
                            p=1,
                            seasonal=True,
                            m=12,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
    else:
        print('[Using manual parameters]')
        model = m.ARIMA(data_frame,order=(2, 0, 3), seasonal_order=(0, 0, 2, 12)).fit()

    print('[Model builded]')
    return model

def forecast(smodel: m.ARIMAResults, test: DataFrame, alpha: float):
    print('[Forecasting ' + str(test.size) + ' points]')
    result = smodel.get_prediction(start=smodel.nobs, end=smodel.nobs + test.size - 1)
    conf = result.conf_int(alpha=alpha)

    fc_series = pd.Series(result.predicted_mean, index=test.index)
    lower_series = pd.Series(conf["lower temp"], index=test.index)
    upper_series = pd.Series(conf["upper temp"], index=test.index)

    print('[RMSE: ' + str(rmse(result.predicted_mean, test['temp'])) + ']')

    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actual')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

def rmse(forecast: DataFrame, actual: DataFrame):
    return np.mean((forecast - actual)**2)**.5

def analyze(data: DataFrame):
    result = seasonal_decompose(data, model="additive")
    result.plot()

    print(ndiffs(data))
    print(nsdiffs(data, 24))

    plot_acf(data)
    plot_pacf(data)
    plt.show()

def main():
    df = read_df()

    analyze(df)
    exit()

    split = 756 # 63 year of train data, 16 years of test data (around 80/20 split)
    train = df[:split]
    test = df[split:]

    auto = False
    smodel = build_model(train, auto)
    smodel.summary()

    alpha = 0.05
    forecast(smodel, test, alpha)

if __name__ == '__main__':
    main()