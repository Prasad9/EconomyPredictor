import pandas as pd
import os
import numpy as np

from Constants import *


class GenerateData:
    def __init__(self, params, is_zero_center=True):
        speaker = params['SPEAKER']
        meta_file = os.path.abspath(params['META_FILE'])
        data_folder = os.path.abspath(params['DATA_FOLDER'])
        train_test_ratio = params['TRAIN_TEST_RATIO']

        sentiment_data, self._transcripts = self._generate_data(speaker, meta_file, data_folder)
        print('Total sentiments of {} is {}'.format(speaker, len(sentiment_data)))

        if not is_zero_center:
            sentiment_data = [(s + 1) / 2.0 for s in sentiment_data]

        self._train_data, self._test_data = self._split_data(sentiment_data, train_test_ratio)
        self._train_labels, self._test_labels = None, None

    def _generate_data(self, speaker, meta_file, data_folder):
        meta_df = pd.read_csv(meta_file)
        speaker_data = meta_df[meta_df['Speaker'] == speaker]
        video_ids = speaker_data['VideoId']

        score_series = []
        transcripts = []
        for video_id in video_ids:
            transcript_path = os.path.join(data_folder, video_id + '.csv')
            transcript_df = pd.read_csv(transcript_path)
            sentiment_data = transcript_df[kDocSentimentScore].tolist()
            score_series.extend(sentiment_data)
            transcript = transcript_df['text']
            transcripts.extend(transcript)

        return score_series, transcripts

    def _split_data(self, data, train_test_ratio):
        train_samples = int(train_test_ratio * len(data))
        train_data = data[:train_samples]
        test_data = data[train_samples:]
        return train_data, test_data

    # Credits: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
    def _generate_time_series(self, data, n_in, n_out=1):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)

        labels = agg.pop('var1(t)')
        labels = labels.values
        data_series = agg.values
        data_series = np.expand_dims(data_series, axis=1)

        return data_series, labels

    def convert_to_time_series(self, time_lag):
        self._train_data, self._train_labels = self._generate_time_series(self._train_data, time_lag)
        self._test_data, self._test_labels = self._generate_time_series(self._test_data, time_lag)

    def get_train_sentiment_data(self):
        if self._train_labels is None:
            return self._train_data
        return self._train_data, self._train_labels

    def get_test_sentiment_data(self):
        if self._test_labels is None:
            return self._test_data
        return self._test_data, self._test_labels

    def get_transcripts(self):
        return self._transcripts
