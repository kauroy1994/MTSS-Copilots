import requests
import json
import torch
import time
import nltk
import torch.nn as nn
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from bs4 import BeautifulSoup
from tqdm import tqdm
from decouple import config
from groq import Groq
from random import choice
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('wordnet')

class Symbolic_Model:

    def __init__(self):

        def vector_similarity(keywords1,keywords2):
            """
            implements jaccard similarity based set similarity
            """

            A, B = set(keywords1), set(keywords2)
            # Intersaction and Union of two sets can also be done using & and | operators.
            C = A.intersection(B)
            D = A.union(B)
            return float(len(C))/float(len(D))

        def vectorize(sentence):
            """
            implements a symbolic vectorizer instance
            """

            return word_tokenize(sentence)

        self.vectorize = vectorize
        self.vector_similarity = vector_similarity

class Neural_Net:

    def __init__(self):

        def vector_similarity(vector1,vector2):
            """
            implements a vector similarity instance
            """

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            return cos(vector1,vector2)

        def vectorize(sentence):
            """
            implements a vectorizer instance
            """

            embedding_model = TransformerDocumentEmbeddings('bert-base-uncased')
            sentence = Sentence(sentence)
            embedding_model.embed(sentence)
            return sentence.embedding

        self.vectorize = vectorize
        self.vector_similarity = vector_similarity


class Retr:

    @staticmethod
    def retrieve_context_neural(text_splits,random_question,neural_net,top_k = 3):
        """
        retrieves top k context based on vector similarity search
        """

        query_vector = neural_net.vectorize(random_question); query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [neural_net.vectorize(split) for split in text_splits]
        similarities = [neural_net.vector_similarity(x[0],x[1]).item() for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return [text_splits[idx] for idx in top_3_idxs]

    def retrieve_context_symbolic(text_splits,random_question,symb_model,top_k = 3):
        """
        retrieves top k context based on symbolic search
        """

        query_vector = symb_model.vectorize(random_question); query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [symb_model.vectorize(split) for split in text_splits]
        similarities = [symb_model.vector_similarity(x[0],x[1]) for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return [text_splits[idx] for idx in top_3_idxs]

    def retrieve_context(text_splits,query,neural_net=None,symb_model=None,top_k = 3):
        """
        retrieves top k context based on hybrid search
        """
        neural_context, symbolic_context = [], []
        if not neural_net is None:
            neural_context += Retr.retrieve_context_neural(text_splits,query,neural_net)
        if not symb_model is None:
            symbolic_context += Retr.retrieve_context_symbolic(text_splits,query,symb_model)
        return neural_context + symbolic_context

class Memory:

    @staticmethod
    def organzie(text):
        pass

    @staticmethod
    def retrieve():
        pass