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

def download(rows: int):
    print('[Downloading file]')

    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    filehandle, _ = urllib.urlretrieve(url)
    zip_file = ZipFile(filehandle)
    zip_file.extractall()

    print('[File downloaded]')

    print('[Reading file]')
    csv_path = "jena_climate_2009_2016.csv"
    df = pd.read_csv(csv_path, names=['date', 'temp'], header=0, usecols=[0, 2], index_col=0, parse_dates=[0], nrows=rows)

    print('[File read]')
    return df

def prepare_df(rows: int):
    print('[Preparing data]')

    df = download(rows)

    print('[Grouping temperature values into hours]')
    mean_df =  df.resample('H').median()

    print('[Temperature values grouped]')
    print('[Data prepared]')
    return mean_df

def build_model(data_frame: DataFrame, auto: bool):
    print('[Building model]')

    if auto:
        print('[Finding parameters]')
        # ARIMA(2,1,2)(1,0,1)[24] intercept   : AIC=inf, Time=97.06 sec
        # ARIMA(0,1,0)(0,0,0)[24] intercept   : AIC=63586.314, Time=0.21 sec
        # ARIMA(1,1,0)(1,0,0)[24] intercept   : AIC=62304.969, Time=6.72 sec
        # ARIMA(0,1,1)(0,0,1)[24] intercept   : AIC=62555.760, Time=6.56 sec
        # ARIMA(0,1,0)(0,0,0)[24]             : AIC=63584.315, Time=0.16 sec
        # ARIMA(1,1,0)(0,0,0)[24] intercept   : AIC=62315.649, Time=0.55 sec
        # ARIMA(1,1,0)(2,0,0)[24] intercept   : AIC=61937.582, Time=47.45 sec
        # ARIMA(1,1,0)(2,0,1)[24] intercept   : AIC=inf, Time=239.19 sec
        # ARIMA(1,1,0)(1,0,1)[24] intercept   : AIC=inf, Time=54.43 sec
        # ARIMA(0,1,0)(2,0,0)[24] intercept   : AIC=62693.366, Time=40.65 sec
        # ARIMA(2,1,0)(2,0,0)[24] intercept   : AIC=61740.714, Time=65.45 sec
        # ARIMA(2,1,0)(1,0,0)[24] intercept   : AIC=62038.649, Time=10.12 sec
        # ARIMA(2,1,0)(2,0,1)[24] intercept   : AIC=inf, Time=311.95 sec
        # ARIMA(2,1,0)(1,0,1)[24] intercept   : AIC=inf, Time=68.32 sec
        # ARIMA(3,1,0)(2,0,0)[24] intercept   : AIC=61686.854, Time=75.22 sec
        # ARIMA(3,1,0)(1,0,0)[24] intercept   : AIC=61973.828, Time=11.76 sec
        # ARIMA(3,1,0)(2,0,1)[24] intercept   : AIC=inf, Time=326.68 sec
        # ARIMA(3,1,0)(1,0,1)[24] intercept   : AIC=inf, Time=71.53 sec
        # ARIMA(4,1,0)(2,0,0)[24] intercept   : AIC=61688.854, Time=85.33 sec
        # ARIMA(3,1,1)(2,0,0)[24] intercept   : AIC=61688.854, Time=126.36 sec
        # ARIMA(2,1,1)(2,0,0)[24] intercept   : AIC=61701.802, Time=103.40 sec
        # ARIMA(4,1,1)(2,0,0)[24] intercept   : AIC=61690.838, Time=104.26 sec
        # ARIMA(3,1,0)(2,0,0)[24]             : AIC=61684.855, Time=11.77 sec
        # ARIMA(3,1,0)(1,0,0)[24]             : AIC=61971.828, Time=3.52 sec
        # ARIMA(3,1,0)(2,0,1)[24]             : AIC=inf, Time=156.97 sec
        # ARIMA(3,1,0)(1,0,1)[24]             : AIC=inf, Time=46.95 sec
        # ARIMA(2,1,0)(2,0,0)[24]             : AIC=61738.715, Time=15.55 sec
        # ARIMA(4,1,0)(2,0,0)[24]             : AIC=61686.855, Time=18.50 sec
        model = pm.auto_arima(data_frame,
                            seasonal=True,
                            m=24,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
    else:
        print('[Using manual parameters]')
        model = m.ARIMA(data_frame,order=(4, 1, 0), seasonal_order=(2, 0, 0, 24)).fit()

    print('[Model builded]')
    return model

def forecast(smodel: m.ARIMAResults, test: DataFrame, alpha: float):
    print('[Forecasting ' + str(test.size) + ' points]')
    result = smodel.get_prediction(start=0, end=157821, dynamic=smodel.nobs - 8760)
    conf = result.conf_int(alpha=alpha)

    fc_series = pd.Series(result.predicted_mean, index=test.index)
    lower_series = pd.Series(conf["lower temp"], index=test.index)
    upper_series = pd.Series(conf["upper temp"], index=test.index)

    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.15)
    plt.title('Forecast vs Actual')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

def analyze(data: DataFrame):
    result = seasonal_decompose(data, model="additive")
    fig = result.plot()

    print(ndiffs(data))
    print(nsdiffs(data, 24))

    # plot_acf(dd)
    # plot_pacf(dd)
    plt.show()

def main():
    rows = 157821 # 3 years worth of data
    df = prepare_df(rows)

    # analyze(df)

    split = 17520 # 2 years train set
    end = 26280 # 1 year test set
    train = df[:split]
    test = df[split:]

    auto = False
    smodel = build_model(train, auto)
    smodel.summary()

    alpha = 0.05
    forecast(smodel, train, alpha)

if __name__ == '__main__':
    main()