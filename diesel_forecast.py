import matplotlib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn.metrics as metrics
import math
from pandas.tseries.offsets import DateOffset


def forecastDiesel(months: int):
    data = pd.read_csv("assets/dataset.csv")
    ## Cleaning up the data
    data.drop(columns=['Petrol'], inplace=True)

    # Convert Month into Datetime
    data['Date'] = pd.to_datetime(data['Month'])
    data.set_index('Date', inplace=True)

    data.drop(columns=['Month'], inplace=True)

    # plt.plot(data)
    # plt.show()

    def test_stationarity(timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(window=12).mean()
        rolstd = timeseries.rolling(window=12).std()

        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

    # test_stationarity(data)

    def difference(dataset, interval=1):
        index = list(dataset.index)
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset["Diesel"][i] - dataset["Diesel"][i - interval]
            diff.append(value)
        return (diff)

    diff = difference(data)
    # plt.plot(diff)
    # plt.show()

    ts_log = np.log(data)
    # plt.title('Log of the data')
    # plt.plot(ts_log)
    # plt.show()

    moving_avg = ts_log.rolling(12).mean()
    # plt.plot(ts_log)
    # plt.title('12 years of Moving average')
    # plt.plot(moving_avg, color='blue')
    # plt.show()

    ts_log_moving_avg_diff = ts_log - moving_avg

    ts_log_moving_avg_diff.dropna(inplace=True)
    # test_stationarity(ts_log_moving_avg_diff)

    expwighted_avg = ts_log.ewm(halflife=12).mean()
    # parameter halflife is used to define the amount of exponential decay
    # plt.plot(ts_log)
    # plt.plot(expwighted_avg, color='red')
    # plt.show()

    ts_log_ewma_diff = ts_log - expwighted_avg
    # test_stationarity(ts_log_ewma_diff)

    ts_log_diff = ts_log - ts_log.shift()
    # plt.plot(ts_log_diff)
    # plt.show()

    ts_log_diff.dropna(inplace=True)
    # test_stationarity(ts_log_diff)

    decomposition = seasonal_decompose(ts_log)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # plt.subplot(411)
    # plt.plot(ts_log, label='Original')
    # plt.legend(loc='best')
    # plt.subplot(412)
    # plt.plot(trend, label='Trend')
    # plt.legend(loc='best')
    # plt.subplot(413)
    # plt.plot(seasonal,label='Seasonality')
    # plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    # plt.show()
    # plt.tight_layout()

    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    # test_stationarity(ts_log_decompose)

    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    # plt.show()
    # plt.title('Autocorrelation Function')
    #
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.axhline(y=0, linestyle='--', color='gray')
    # plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    # plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    # plt.title('Partial Autocorrelation Function')
    # plt.show()
    # plt.tight_layout()

    # Arima Model
    model = ARIMA(ts_log, order=(2, 2, 2), freq=ts_log.index.inferred_freq)
    results_ARIMA = model.fit()
    # plt.plot(ts_log_diff)
    # plt.plot(results_ARIMA.fittedvalues, color='red')
    # plt.show()

    train = ts_log[ts_log.index < pd.to_datetime("2018-11-01", format='%Y-%m-%d')]
    test = ts_log[ts_log.index >= pd.to_datetime("2018-11-01", format='%Y-%m-%d')]
    # print(test)
    # plt.plot(train, color="black", label='Training')
    # plt.plot(test, color="red", label='Testing')
    # plt.ylabel('Diesel')
    # plt.xlabel('Date')
    # plt.xticks(rotation=45)
    # plt.title("Train/Test split for df Data")

    y = train['Diesel']

    SARIMAXmodel = SARIMAX(y, order=(2, 1, 2), seasonal_order=(2, 2, 2, 12), freq=ts_log.index.inferred_freq)
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred = SARIMAXmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Forecast"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Forecast"]

    # plt.plot(y_pred_out, color='green', label='SARIMA Forecast')
    # plt.plot(train, color="black", label='Training')
    # plt.plot(test, color="red", label='Testing')

    # plt.legend()

    pred_out = np.exp(y_pred_out)
    print(pred_out)

    x_true = []
    for i in test['Diesel']:
        rounded = math.ceil(i)
        x_true.append(rounded)

    x_pred = []
    for i in y_pred_out:
        rounded = math.ceil(i)
        x_pred.append(rounded)

    # print(x_true)
    # print(x_pred)

    columns = ['Model', 'accuracy score', ' Precision', 'Recall', 'f1_score']
    evaluation_df = pd.DataFrame(columns=columns)

    # evaluation_df
    # creating a printing function for our models
    def print_results(model_name, y_test, y_pred, pred_prob=None):
        print(model_name)
        print('--------------------------------------------------------------------------')

        precision_score = metrics.precision_score(y_test, y_pred, zero_division=1, pos_label=10)
        recall_score = metrics.recall_score(y_test, y_pred, zero_division=1, pos_label=10)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)
        print(f'accuracy score :{accuracy_score}')

        f1_score = metrics.f1_score(y_test, y_pred, zero_division=1, pos_label=10)

        classification_report = metrics.classification_report(y_test, y_pred)
        print(classification_report)

        #   save scores into dataframe for comparison
        evaluation_df.loc[len(evaluation_df.index)] = [model_name, accuracy_score, precision_score, recall_score,
                                                       f1_score]

    #     if pred_prob is not None:
    #         Plot_roc_curve(y_test, pred_prob, model_name, accuracy_score)

    # Created a common function to plot confusion matrix
    def Plot_confusion_matrix(y, pred, model_name):
        cm = metrics.confusion_matrix(y, pred)
        plt.clf()
        plt.imshow(cm, cmap=plt.cm.Accent)
        categoryNames = ['Test', 'Predicted']
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        ticks = np.arange(len(categoryNames))
        plt.xticks(ticks, categoryNames, rotation=45)
        plt.yticks(ticks, categoryNames)
        s = [['TN', 'FP'], ['FN', 'TP']]

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]), fontsize=12)
        plt.savefig('assets/graphs/diesel_confusion_matrix.png')

    # Define Forecasted Dates

    future_dates = [ts_log.index[-1] + DateOffset(months=x) for x in range(0, 60)]

    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=data.columns)

    # future_datest_df.tail()
    future_df = pd.concat([ts_log, future_datest_df])
    # future_df.tail()

    future_df['Forecast'] = SARIMAXmodel.predict(len(ts_log), len(ts_log) + months)
    plt.plot(future_df[['Diesel', 'Forecast']])
    plt.savefig('assets/graphs/forecasted_diesel_graph.png')

    # Precision,Accuracy,F1,recall
    print_results('Combined Arima & Sarima Model', x_true, x_pred)

    # confusion matrix
    Plot_confusion_matrix(x_true, x_pred, "Combined Arima & Sarima Model")

    pred_log = future_df['Forecast']
    pred_out = np.exp(pred_log.dropna())
    print(pred_out)

    # Get Fraud Transactions
    forecasted_data = pd.DataFrame(pred_out)
    final_forecast = forecasted_data.round(0)

    # saving the dataframe
    final_forecast.to_csv('assets/forecasted/diesel_forecasted_data.csv')
