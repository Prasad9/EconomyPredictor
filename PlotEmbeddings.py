import io
import tensorflow_hub as tfhub
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json

from sklearn.decomposition import PCA

from GenerateData import GenerateData

class PlotEmbeddings:
    def __init__(self, params):
        self._batch_size = params['BATCH_SIZE']
        save_custom_sentences = params['SAVE_CUSTOM_SENTENCES']

        generate_data = GenerateData(params)
        sentiment_transcripts = generate_data.get_transcripts()
        # If you wish to study only custom transcripts, uncomment below line
        # sentiment_transcripts = []

        custom_transcript = ['How is the economy doing in this country?',
                             'The current state of affairs is not doing good.',
                             'Life will get difficult when inflation kicks in.',
                             'We are in a bull market.',
                            ]

        self._meta_data = ['Statement_{:04d}'.format(i) for i in range(len(sentiment_transcripts))]
        self._transcripts = sentiment_transcripts
        if save_custom_sentences:
            self._meta_data += ['Custom_{}'.format(i + 1) for i in range(len(custom_transcript))]
            self._transcripts += custom_transcript

        print('Len of transcripts = ', len(self._transcripts), len(self._meta_data))

    # Use this method if you are using TF1 Hub module
    def plot_tf1_hub_embeddings(self, hub_url):
        assert tf.__version__.startswith('1'), 'Please use TF 1.x version'
        module = tfhub.Module(hub_url)
        out_v = io.open('vecs_tf1.tsv', 'a', encoding='utf-8')
        out_m = io.open('meta_tf1.tsv', 'a', encoding='utf-8')

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            for offset in tqdm(range(0, len(self._transcripts), self._batch_size)):
                end = offset + self._batch_size
                sentiment_data = self._transcripts[offset: end]
                meta_data = self._meta_data[offset: end]

                # Sentence embeddings will be converted to word embeddings
                batch_embeddings = sess.run(module(sentiment_data))

                out_m.write('\n'.join([name for name in meta_data]) + '\n')
                for embedding in batch_embeddings:
                    out_v.write('\t'.join([str(e) for e in embedding]) + '\n')
        out_v.close()
        out_m.close()

    # Use this method if you are using TF2 Hub module
    def plot_tf2_hub_embeddings(self, hub_url):
        assert tf.__version__.startswith('2'), 'Please use TF 2.x version'
        module = tfhub.load(hub_url)
        out_v = io.open('vecs_tf2.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta_tf2.tsv', 'w', encoding='utf-8')

        for offset in tqdm(range(0, len(self._transcripts), self._batch_size)):
            end = offset + self._batch_size
            sentiment_data = self._transcripts[offset: end]
            meta_data = self._meta_data[offset: end]

            # Sentence embeddings will be converted to word embeddings
            batch_embeddings = module(sentiment_data)

            out_m.write('\n'.join([name for name in meta_data]) + '\n')
            for embedding in batch_embeddings:
                out_v.write('\t'.join([str(e) for e in embedding]) + '\n')
        out_v.close()
        out_m.close()

    def plot_learned_embeddings(self, embeddings_file, word_index_file):
        vocab_embeddings = np.load(embeddings_file)
        with open(word_index_file, 'r') as fid:
            word_index = json.load(fid)
        vocab_size = vocab_embeddings.shape[0]

        sentence_embeddings = []

        out_v = io.open('vecs_le.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta_le.tsv', 'w', encoding='utf-8')

        for transcript, name in tqdm(zip(self._transcripts, self._meta_data)):
            word_embeddings = []
            words = transcript.split()
            for word in words:
                word = word.lower()
                if word in word_index:
                    index_at = word_index[word]
                    if index_at >= vocab_size:
                        word_embeddings.append(0)
                    else:
                        word_embeddings.append(index_at)
                else:
                    word_embeddings.append(0)

            indices = tf.convert_to_tensor(np.arange(len(words)))
            indices = tf.stack((tf.zeros_like(indices), indices), axis=1)
            word_embeddings = tf.convert_to_tensor(word_embeddings)
            word_embeddings_sparse = tf.sparse.SparseTensor(indices, word_embeddings, [1, len(words)])
            sentence_embedding = tf.nn.embedding_lookup_sparse(vocab_embeddings, word_embeddings_sparse, None,
                                                               combiner='sqrtn')[0]

            sentence_embeddings.append(sentence_embedding)

            out_m.write(name + "\n")
            out_v.write('\t'.join([str(x) for x in sentence_embedding]) + '\n')

        out_v.close()
        out_m.close()

        # Uncomment below lines if you wish to generate PCA components as well
        # pca = PCA(n_components=3)
        # pca_se = pca.fit_transform(sentence_embeddings)
        # print('SE shape = ', pca_se.shape)
        #
        # out_v = io.open('vecs_pca_le.tsv', 'w', encoding='utf-8')
        # for se in pca_se:
        #     out_v.write('\t'.join([str(x) for x in se]) + '\n')
        # out_v.close()


if __name__ == '__main__':
    params = {
        'DATA_FOLDER': './Data',
        'META_FILE': './Data/meta.csv',
        'SPEAKER': 'Warren Buffet',                                        # Whose data you wish to analyse
        'TRAIN_TEST_RATIO': 1.0,                                           # There is no testing data involved

        'BATCH_SIZE': 64,
        'SAVE_CUSTOM_SENTENCES': True                                      # Should the custom sample sentences be saved too
    }
    p = PlotEmbeddings(params)
    p.plot_tf1_hub_embeddings('https://tfhub.dev/google/universal-sentence-encoder/2')

    # p.plot_learned_embeddings('vocabulary_weights.npy', 'word_index.json')