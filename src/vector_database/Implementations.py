import faiss
import pandas as pd
import numpy as np
from miscellaneous.Math import Activations
from sentence_transformers import SentenceTransformer

class Faiss:

    encoder = None
    index = None

    @staticmethod
    def prepare_data(all_qa):
        
        return pd.DataFrame([[item['question'],item['answer']] for item in all_qa],
        columns = ['question','answer'])

    @staticmethod
    def vectorize(dataframe):

        questions = dataframe['question']
        encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        vectors = encoder.encode(questions)
        return vectors, encoder

    @staticmethod
    def create_index(dataframe):

        vectors, encoder = Faiss.vectorize(dataframe)
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        Faiss.index = index
        Faiss.encoder = encoder

    @staticmethod
    def create_search_vector(query):

        search_vector = Faiss.encoder.encode(query)
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)
        return _vector

    @staticmethod
    def search(query, dataframe):

        query_vector = Faiss.create_search_vector(query)
        k = Faiss.index.ntotal
        distances, ann = Faiss.index.search(query_vector, k = k)
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        merge = pd.merge(results, dataframe, left_on='ann',right_index=True)
        labels = dataframe['answer']
        return Activations.sigmoid(merge['distances'][0]), labels[ann[0][0]]