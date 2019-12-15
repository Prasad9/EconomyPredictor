import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp

from GenerateData import GenerateData


class RunSTS:
    def __init__(self, params):
        generate_data = GenerateData(params)
        self._train_data = np.array(generate_data.get_train_sentiment_data())
        self._test_data = generate_data.get_test_sentiment_data()

        seasons = params['SEASONS']
        self._model = self._build_model(seasons)
        # Build the variational surrogate posteriors `qs`.
        self._variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self._model)

    def _build_model(self, seasons):
        trend = tfp.sts.LocalLinearTrend(observed_time_series=self._train_data)
        seasonal = tfp.sts.Seasonal(num_seasons=seasons, observed_time_series=self._train_data)
        model = tfp.sts.Sum([trend, seasonal], observed_time_series=self._train_data)
        return model

    def train_model_data(self):
        print('Training....')
        # Minimize the variational loss.

        # Allow external control of optimization to reduce test runtimes.
        num_variational_steps = 200

        optimizer = tf.optimizers.Adam(learning_rate=.1)

        # Using fit_surrogate_posterior to build and optimize the variational loss function.
        tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self._model.joint_log_prob(observed_time_series=self._train_data),
            surrogate_posterior=self._variational_posteriors,
            optimizer=optimizer,
            num_steps=num_variational_steps)

    def test_model_data(self):
        print('Testing....')
        # Draw samples from the variational posterior.
        data_samples = self._variational_posteriors.sample(50)
        data_series = self._train_data
        predictions = []

        for test_val in tqdm(self._test_data):
            forecast_dist = tfp.sts.forecast(self._model,
                                    observed_time_series=data_series,
                                    parameter_samples=data_samples,
                                    num_steps_forecast=1)
            data_series = np.append(data_series, [test_val], axis=0)

            prediction = forecast_dist.mean().numpy()[..., 0][0]
            prediction = max(-1.0, prediction)
            prediction = min(1.0, prediction)
            predictions.append(prediction)

        error = sklearn.metrics.mean_squared_error(self._test_data, predictions)

        print('Test MSE: {:.3f}'.format(error))
        plt.clf()
        plt.plot(self._test_data, color='blue', label='Ground Truth')
        plt.plot(predictions, color='red', label='Prediction')
        plt.xlabel('Time Steps', fontsize=13)
        plt.ylabel('Sentiment', fontsize=13)
        plt.title('TFP Results')
        plt.legend()
        plt.show()

    def forecast_data(self):
        print('Forecasting.....')
        data_series = np.append(self._train_data, self._test_data, axis=0)
        # Draw samples from the variational posterior.
        data_samples = self._variational_posteriors.sample(50)
        forecast_dist = tfp.sts.forecast(self._model,
                                 observed_time_series=data_series,
                                 parameter_samples=data_samples,
                                 num_steps_forecast=1)
        output = forecast_dist.mean().numpy()[..., 0][0]
        print('Forecasted output: {:.3f}'.format(output))


if __name__ == '__main__':
    params = {
        'SPEAKER': 'Warren Buffet',
        'META_FILE': './Data/meta.csv',
        'DATA_FOLDER': './Data',
        'TRAIN_TEST_RATIO': 0.95,

        'SEASONS': 10
    }
    run_sts = RunSTS(params)
    run_sts.train_model_data()
    run_sts.test_model_data()
    run_sts.forecast_data()