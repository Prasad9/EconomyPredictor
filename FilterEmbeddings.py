import os
import nmslib
import pandas as pd


class FilterEmbeddings:
    def __init__(self, params):
        self._vecs_file = os.path.abspath(params['VECS_FILE'])
        self._meta_file = os.path.abspath(params['NAME_FILE'])

        self._vector_space = self._create_vector_space(self._vecs_file)
        print(self._vector_space)

        embedding_file = os.path.abspath(params['EMBEDDING_FILE'])
        self._filter_embeddings = self._read_tsv_file(embedding_file)

    def _create_vector_space(self, file_path):
        vector_data = self._read_tsv_file(file_path)
        vector_space = nmslib.init(method='hnsw', space='cosinesimil')
        vector_space.addDataPointBatch(vector_data)
        vector_space.createIndex({'post': 2}, print_progress=True)
        return vector_space

    def _read_tsv_file(self, tsv_file):
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        vector_data = df.values
        return vector_data

    def filter_data(self, nearest_point):
        # get all nearest neighbours for all the datapoint
        # using a pool of 4 threads to compute
        neighbours = self._vector_space.knnQueryBatch(self._filter_embeddings, k=nearest_point, num_threads=4)

        filtered_id_set = set()
        for vector_ids, _ in neighbours:
            filtered_id_set.update(vector_ids)

        filtered_ids = list(filtered_id_set)
        filtered_ids.sort()

        self._filter_rows_and_store(self._vecs_file, filtered_ids)
        self._filter_rows_and_store(self._meta_file, filtered_ids)

    def _filter_rows_and_store(self, file_path, filter_rows):
        df = pd.read_csv(file_path, sep='\t', header=None)
        df = df.iloc[filter_rows, :]
        save_file_path = os.path.abspath('filtered_' + os.path.basename(file_path))
        df.to_csv(save_file_path, index=False)
        print('Filtered data has been saved at ', save_file_path)


if __name__ == '__main__':
    params = {
        'VECS_FILE': 'vecs_tf1.tsv',
        'NAME_FILE': 'meta_tf1.tsv',
        'EMBEDDING_FILE': 'custom_vecs_tf1.tsv'
    }

    fe = FilterEmbeddings(params)
    fe.filter_data(nearest_point=500)
