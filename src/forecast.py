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
        # ARIMA(2,0,2)(1,0,1)[12] intercept   : AIC=inf, Time=17.35 sec
        # ARIMA(0,0,0)(0,0,0)[12] intercept   : AIC=4171.137, Time=0.07 sec
        # ARIMA(1,0,0)(1,0,0)[12] intercept   : AIC=2771.028, Time=4.13 sec
        # ARIMA(0,0,1)(0,0,1)[12] intercept   : AIC=3275.026, Time=1.75 sec
        # ARIMA(0,0,0)(0,0,0)[12]             : AIC=5671.382, Time=0.03 sec
        # ARIMA(1,0,0)(0,0,0)[12] intercept   : AIC=3381.856, Time=0.22 sec
        # ARIMA(1,0,0)(2,0,0)[12] intercept   : AIC=inf, Time=13.20 sec
        # ARIMA(1,0,0)(1,0,1)[12] intercept   : AIC=inf, Time=8.24 sec
        # ARIMA(1,0,0)(0,0,1)[12] intercept   : AIC=3133.417, Time=1.50 sec
        # ARIMA(1,0,0)(2,0,1)[12] intercept   : AIC=inf, Time=22.26 sec
        # ARIMA(0,0,0)(1,0,0)[12] intercept   : AIC=inf, Time=3.42 sec
        # ARIMA(2,0,0)(1,0,0)[12] intercept   : AIC=2769.029, Time=6.94 sec
        # ARIMA(2,0,0)(0,0,0)[12] intercept   : AIC=3110.499, Time=0.37 sec
        # ARIMA(2,0,0)(2,0,0)[12] intercept   : AIC=inf, Time=19.03 sec
        # ARIMA(2,0,0)(1,0,1)[12] intercept   : AIC=inf, Time=10.40 sec
        # ARIMA(2,0,0)(0,0,1)[12] intercept   : AIC=3062.754, Time=2.24 sec
        # ARIMA(2,0,0)(2,0,1)[12] intercept   : AIC=inf, Time=25.03 sec
        # ARIMA(3,0,0)(1,0,0)[12] intercept   : AIC=2768.910, Time=9.63 sec
        # ARIMA(3,0,0)(0,0,0)[12] intercept   : AIC=2862.989, Time=0.37 sec
        # ARIMA(3,0,0)(2,0,0)[12] intercept   : AIC=inf, Time=30.56 sec
        # ARIMA(3,0,0)(1,0,1)[12] intercept   : AIC=2391.583, Time=13.28 sec
        # ARIMA(3,0,0)(0,0,1)[12] intercept   : AIC=2857.519, Time=2.92 sec
        # ARIMA(3,0,0)(2,0,1)[12] intercept   : AIC=inf, Time=34.97 sec
        # ARIMA(3,0,0)(1,0,2)[12] intercept   : AIC=inf, Time=27.37 sec
        # ARIMA(3,0,0)(0,0,2)[12] intercept   : AIC=2833.304, Time=15.06 sec
        # ARIMA(3,0,0)(2,0,2)[12] intercept   : AIC=inf, Time=35.84 sec
        # ARIMA(4,0,0)(1,0,1)[12] intercept   : AIC=2469.696, Time=14.71 sec
        # ARIMA(3,0,1)(1,0,1)[12] intercept   : AIC=inf, Time=18.41 sec
        # ARIMA(2,0,1)(1,0,1)[12] intercept   : AIC=inf, Time=13.63 sec
        # ARIMA(4,0,1)(1,0,1)[12] intercept   : AIC=inf, Time=nan sec
        # ARIMA(3,0,0)(1,0,1)[12]             : AIC=inf, Time=12.70 sec
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
        model = m.ARIMA(data_frame,order=(3, 0, 0), seasonal_order=(1, 0, 1, 12)).fit()

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

    # analyze(df)

    split = 756 # 63 year of train data, 16 years of test data (around 80/20 split)
    train = df[:split]
    test = df[split:]

    auto = True
    smodel = build_model(train, auto)
    smodel.summary()

    alpha = 0.05
    forecast(smodel, test, alpha)

if __name__ == '__main__':
    main()