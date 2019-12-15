import os
import io
import tensorflow as tf
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from GenerateSentiment import GenerateSentiment
from Constants import *


class GenerateWordEmbeddings:
    def __init__(self, params):
        self._epochs = params['EPOCHS']
        self._vocab_size = params['VOCAB_SIZE']

        self._train_ds, self._val_ds = self._prepare_dataset(params)

        embedding_dim = params['EMBEDDING_DIM']
        self._model = self._create_model(embedding_dim)

    def _prepare_dataset(self, params):
        sentiment_filepath = os.path.abspath(params['GENERAL_SENTIMENTS'])
        train_val_ratio = params['TRAIN_VAL_RATIO']
        batch_size = params['BATCH_SIZE']

        # Do not generate sentiments again as it is costly
        if not os.path.isfile(sentiment_filepath):
            pdf_folder = os.path.abspath(params['PDF_FOLDER'])
            min_sentence_length = params['MIN_SENTENCE_LENGTH']

            raw_text = self._read_pdf_files(pdf_folder)
            sentences = raw_text.split('.')
            sentences = [s for s in sentences if len(s) > min_sentence_length]
            print('Sentences = ', len(sentences))

            self._generate_sentiments(sentences, sentiment_filepath)

        data_sequence, labels = self._preprocess_dataset(sentiment_filepath)
        X_train, X_val, y_train, y_val = train_test_split(data_sequence, labels,
                                                          train_size=train_val_ratio,
                                                          shuffle=True)
        print('Train samples: {}, Val samples: {}'.format(len(y_train), len(y_val)))
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        return train_ds, val_ds

    # Credits: https://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text
    def _read_pdf_files(self, pdf_folder):
        print('Reading PDF files.....')
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        pdf_files = os.listdir(pdf_folder)
        raw_data = ''
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_folder, pdf_file)
            print('Processing ', file_path)

            file_data = ''
            with open(file_path, 'rb') as fp:
                # Process each page contained in the document.
                for page in PDFPage.get_pages(fp):
                    interpreter.process_page(page)
                    file_data = retstr.getvalue()
            raw_data += file_data + '\n'

        return raw_data

    def _generate_sentiments(self, sentences, sentiment_filepath):
        print('Generating sentiments.....')
        gs = GenerateSentiment()
        sentiments = []
        for sentence in tqdm(sentences):
            sentiment_dict = gs.generate_sentiment(sentence)
            sentiments.append(sentiment_dict[kDocSentimentScore])
        df = pd.DataFrame(data={'Sentences': sentences, 'Sentiments': sentiments})
        df.to_csv(sentiment_filepath, index=False)

    def _preprocess_dataset(self, sentiment_filepath):
        print('Preprocessing dataset.....')
        df = pd.read_csv(sentiment_filepath)
        sentences = df['Sentences']
        sentiments = df['Sentiments']

        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>', num_words=self._vocab_size)
        tokenizer.fit_on_texts(sentences)

        word_index = tokenizer.word_index
        print('Word index = ', len(word_index))
        with open('word_index.json', 'w') as fid:
            json.dump(word_index, fid)
        keys = list(word_index.keys())
        print({i: word_index[i] for i in keys[:20]})

        sequences = tokenizer.texts_to_sequences(sentences)
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

        print('Sentence[1] = ', sentences[1])
        print('Padded seq[1] = ', padded_seq[1])
        print('Padded shape = ', padded_seq.shape)
        print('Labels shape = ', len(sentiments))
        return padded_seq, sentiments

    def _create_model(self, embedding_dim):
        print('Creating model....')
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self._vocab_size, embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation='tanh')
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self):
        print('Training model....')
        self._model.fit(self._train_ds,
                        epochs=self._epochs,
                        validation_data=self._val_ds)

    def save_embedding_layer(self):
        embedding_layer = self._model.layers[0]
        weights = embedding_layer.get_weights()[0]
        np.save('vocabulary_weights.npy', weights)


if __name__ == '__main__':
    params = {
        'EPOCHS': 5,
        'BATCH_SIZE': 64,
        'TRAIN_VAL_RATIO': 0.9,

        'EMBEDDING_DIM': 128,                              # Size of your embedding vector
        'MIN_SENTENCE_LENGTH': 10,                         # Minimum length of the sentence
        'VOCAB_SIZE': 10000,                               # Vocabulary size of your language model

        'PDF_FOLDER': './PDFs',                            # The location where input PDF files are kept
        'GENERAL_SENTIMENTS': 'general_sentiments.csv'     # Sentiment generation is a costly process.
                                                           # Do not repeat the entire procedure if you wish to train
                                                           # again. Store the intermediate generated values of
                                                           # sentiments in this file. If this file exists,
                                                           # sentiments will not be generated again.
    }
    word_embeddings = GenerateWordEmbeddings(params)
    word_embeddings.train()
    word_embeddings.save_embedding_layer()