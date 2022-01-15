from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pandas.core.frame import DataFrame
import pmdarima as pm
import statsmodels.tsa.arima.model as m
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from collections import namedtuple
import getopt
import sys

# seasonal frequency
S = 12

# forecast results
Results = namedtuple('Results', ['forecast', 'lower', 'upper'])

### COMMON
def read_df(path = None):
    print('[Reading file]')

    if isinstance(path, type(None)):
        path = "mly532.csv"
    df = pd.read_csv(path,
                     names=['year', 'month', 'temp'],
                     header=16,
                     usecols=[0, 1, 2],
                     index_col=0,
                     parse_dates=[[0, 1]],
                     skiprows=2,
                     skipfooter=21,
                     engine='python')
    df = df.resample('MS').asfreq()
    print('[File read]')
    return df

# root mean squared error
def rmse(forecast: DataFrame, actual: DataFrame):
    return np.mean((forecast - actual)**2)**.5

# mean absolute percentage error
def mape(forecast: DataFrame, actual: DataFrame):
    return np.mean(np.abs(forecast - actual)/np.abs(actual))

def show_performance(forecast: DataFrame, actual: DataFrame):
    print('[RMSE: ' + str(rmse(forecast, actual)) + ']')
    print('[MAPE: ' + str(mape(forecast, actual)) + ']')

def plot_data(results: Results, prefix: str):
    plt.plot(results.forecast, label=prefix + ' forecast')

    if results.lower is not None and results.upper is not None:
        plt.fill_between(results.lower.index,
                         results.lower,
                         results.upper,
                         color='k', alpha=.15)

def plot_single(results: Results, test:DataFrame, label: str):
    plt.figure(figsize=(12,5), dpi=100, tight_layout=True)
    plt.plot(test, label='Actual')
    plot_data(results=results, prefix=label)
    plt.title(label)
    plt.legend(loc='upper left', fontsize=8)

def process_results(test: DataFrame, sarima_results: Results, arima_results: Results):

    plt.figure(figsize=(12,5), dpi=100, tight_layout=True)
    plt.plot(test, label='Actual')

    # process SARIMA
    print('[SARIMA]')
    show_performance(sarima_results.forecast, test['temp'])
    plot_data(sarima_results, 'SARIMA')

    # process ARIMA
    print('[ARIMA]')
    show_performance(arima_results.forecast, test['temp'])
    plot_data(arima_results, 'ARIMA')

    plt.title('SARIMA vs Group ARIMA')
    plt.legend(loc='upper left', fontsize=8)

    plot_single(sarima_results, test, 'SARIMA')
    plot_single(arima_results, test, 'ARIMA')
    plt.show()

# helper method for data analysis
def analyze(data: DataFrame):
    result = seasonal_decompose(data, model="additive")
    result.plot()

    plot_acf(data)
    plot_pacf(data, method="ywm")
    plt.show()


### SARIMA
def build_sarima_model(data: DataFrame, auto: bool):
    print('[Building SARIMA model]')

    if auto:
        print('[Finding parameters]')
        model = pm.auto_arima(data,
                            p=1,
                            d=0,
                            q=2,
                            seasonal=True,
                            start_P=1,
                            D=1,
                            Q=0,
                            m=S,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
    else:
        print('[Using manual parameters]')
        model = m.ARIMA(data, order=(1, 0, 2), seasonal_order=(2, 1, 0, S)).fit()

    print('[Model builded]')
    return model

def forecast_sarima(train: DataFrame, test: DataFrame, alpha: float):
    print('[Forecasting with SARIMA model]')

    auto = False
    model = build_sarima_model(train, auto)

    print('[Forecasting ' + str(test.size) + ' points with SARIMA]')
    result = model.get_prediction(start=model.nobs, end=model.nobs + test.size - 1)
    conf = result.conf_int(alpha=alpha)

    fc_series = pd.Series(result.predicted_mean, index=test.index)
    lower_series = pd.Series(conf["lower temp"], index=test.index)
    upper_series = pd.Series(conf["upper temp"], index=test.index)

    return Results(fc_series, lower_series, upper_series)

### ARIMA
ModelData = namedtuple('ModelData', ['data', 'frequency'])

# split data frame to correct interval with offset
def split_dataframe(data: DataFrame, interval: int, offset: int):\
    return data[offset:].resample(str(interval) + 'MS').asfreq()

# split data into groups
def split_data(data: DataFrame, splits: np.ndarray):
    return [[ModelData(split_dataframe(data, i, j), i) for j in range(0, i)] for i in splits]

def build_arima_machine(data: DataFrame, interval: int):
    # calculate terms
    it = S / interval
    return m.ARIMA(data, order=(it, 0, it)).fit()

def build_arima_machines(train: DataFrame, splits: np.ndarray):
    print('[Building ARIMA model]')

    train_splitted = split_data(train, splits)
    machines = [[build_arima_machine(data.data, data.frequency) for data in group] for group in train_splitted]

    print('[Model builded]')
    return machines

def forecast_arima_machines(machines, end: datetime):
    results_separate = [[machine.get_prediction(start=machine.nobs, end=end).predicted_mean for machine in group] for group in machines]
    groups_combined = [pd.concat(group).sort_index()[:end] for group in results_separate]
    return pd.concat(groups_combined, axis=1).mean(axis=1)

def forecast_arima(train: DataFrame, test: DataFrame, splits: np.ndarray):
    print('[Forecasting with ARIMA model]')

    machines = build_arima_machines(train, splits)

    print('[Forecasting ' + str(test.size) + ' points with ARIMA]')
    results = forecast_arima_machines(machines, test.index[-1])

    return Results(results, None, None)

### MAIN
def help():
    print('forecast.py -i <pathToData> [-a]')

def main(argv):
    path = None
    onlyAnalyze = False

    try:
        opts, _ = getopt.getopt(argv, "hai:", ["input=", "analyze"])
    except getopt.GetoptError:
        help()
        exit(2)
    for opt, arg in opts:
        if opt == '-h':
            help()
            exit()
        elif opt in ("-i", "--input"):
            path = arg
        elif opt in ("-a", "--analyze"):
            onlyAnalyze = True

    df = read_df(path)

    if (onlyAnalyze):
        analyze(df)
        exit()

    split = 744 # 62 year of train data, 15 years of test data (around 80/20 split)
    train = df[:split]
    test = df[split:]

    alpha = 0.05
    sresults = forecast_sarima(train, test, alpha)
    aresults = forecast_arima(train, test, np.array([2, 3, 4, 6]))

    process_results(test, sresults, aresults)

if __name__ == '__main__':
    main(sys.argv[1:])