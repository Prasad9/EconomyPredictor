import tensorflow as tf
import sklearn.metrics
import matplotlib.pyplot as plt

from GenerateData import GenerateData


class RunLSTM:
    def __init__(self, params):
        self._epochs = params['EPOCHS']
        self._batch_size = params['BATCH_SIZE']
        self._time_lag = params['TIME_LAG']

        self._generate_data = GenerateData(params)
        self._generate_data.convert_to_time_series(self._time_lag)

        units = params['UNITS']
        input_shape = (1, self._time_lag)
        self._model = self._generate_network(units, input_shape)

    def _generate_network(self, units, input_shape):
        inputs = tf.keras.layers.Input(input_shape)
        hidden = inputs
        for layer_no, unit in enumerate(units):
            return_sequences = not(layer_no == len(units) - 1)
            hidden = tf.keras.layers.LSTM(unit, return_sequences=return_sequences)(hidden)

        hidden = tf.keras.layers.Dense(8, activation='relu')(hidden)
        output = tf.keras.layers.Dense(1, activation='tanh')(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        model.compile('adam', loss='mse')
        return model

    def train_model_data(self):
        data, labels = self._generate_data.get_train_sentiment_data()
        self._model.fit(data, labels, epochs=self._epochs)

    def test_model_data(self):
        data, labels = self._generate_data.get_test_sentiment_data()
        predictions = self._model.predict(data)

        error = sklearn.metrics.mean_squared_error(labels, predictions)
        print('Test MSE: {:.3f}'.format(error))

        plt.clf()
        plt.plot(labels, color='blue', label='Ground Truth')
        plt.plot(predictions, color='red', label='Prediction')
        plt.xlabel('Time Steps', fontsize=13)
        plt.ylabel('Sentiment', fontsize=13)
        plt.title('LSTM Results')
        plt.legend()
        plt.show()

    def forecast_data(self):
        data = self._generate_data.get_train_sentiment_data() + self._generate_data.get_test_sentiment_data()

        predictions = self._model.predict(data)[-1][0]
        print('Forecasted output: ', predictions)


if __name__ == '__main__':
    params = {
        'UNITS': [128],
        'TIME_LAG': 30,

        'SPEAKER': 'Warren Buffet',
        'DATA_FOLDER': './Data',
        'META_FILE': './Data/meta.csv',
        'TRAIN_TEST_RATIO': 0.95,

        'EPOCHS': 25,
        'BATCH_SIZE': 64
    }
    run_lstm = RunLSTM(params)
    run_lstm.train_model_data()
    run_lstm.test_model_data()
    run_lstm.forecast_data()
