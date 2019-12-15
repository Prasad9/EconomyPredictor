from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics

from GenerateData import GenerateData


class RunArima:
    def __init__(self, params):
        self._p = params['p']
        self._d = params['d']
        self._q = params['q']

        self._generate_data = GenerateData(params)

    def analyse_data(self):
        train_data = self._generate_data.get_train_sentiment_data()
        arima = ARIMA(train_data, order=(self._p, self._d, self._q))
        arima_fit = arima.fit(disp=0)
        print(arima_fit.summary())

        residuals = pd.DataFrame(arima_fit.resid)
        residuals.plot()
        plt.show()

        residuals.plot(kind='kde')
        plt.show()

        print(residuals.describe())

    def test_model_data(self):
        train_values = self._generate_data.get_train_sentiment_data()
        predictions = []

        test_data = self._generate_data.get_test_sentiment_data()
        for test_val in test_data:
            arima = ARIMA(train_values, order=(self._p, self._d, self._q))
            arima_fit = arima.fit(disp=0)
            output = arima_fit.forecast()

            prediction = max(output[0], -1.0)
            prediction = min(prediction, 1.0)
            predictions.append(prediction)
            train_values.append(test_val)

            # print('predicted=%f, expected=%f' % (output[0], test_val))

        error = sklearn.metrics.mean_squared_error(test_data, predictions)

        print('Test MSE: {:.3f}'.format(error))
        plt.clf()
        plt.plot(test_data, color='blue', label='Ground Truth')
        plt.plot(predictions, color='red', label='Prediction')
        plt.xlabel('Time Steps', fontsize=13)
        plt.ylabel('Sentiment', fontsize=13)
        plt.title('ARIMA Results')
        plt.legend()
        plt.show()

    def forecast_data(self):
        data = self._generate_data.get_train_sentiment_data() + self._generate_data.get_test_sentiment_data()

        arima = ARIMA(data, order=(self._p, self._d, self._q))
        arima_fit = arima.fit(disp=0)
        output = arima_fit.forecast()

        print('Forecasted output: {:.3f}'.format(output[0][0]))

if __name__ == '__main__':
    params = {
        'p': 5,
        'd': 1,
        'q': 0,

        'SPEAKER': 'Warren Buffet',
        'META_FILE': './Data/meta.csv',
        'DATA_FOLDER': './Data',

        'TRAIN_TEST_RATIO': 0.95,
    }
    arima = RunArima(params)
    # arima.analyse_data()
    arima.test_model_data()
    arima.forecast_data()
