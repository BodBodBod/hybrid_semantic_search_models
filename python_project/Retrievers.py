import numpy as np
import torch
from tqdm.auto import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk # stop-words
from nltk.corpus import stopwords
import pymorphy3
import re
import os

# Transformer Retriever
class TransformerRetriever:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.query_embeddings = {}
        self.corpus_embeddings = None

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts, batch_size=16):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            pooled = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(pooled)
        return torch.cat(embeddings, dim=0).numpy()

    def cache_query_embeddings(self, queries):
        embeddings = self.encode(queries)
        self.query_embeddings = {
            query: emb for query, emb in zip(queries, embeddings)
        }

    def get_query_embedding(self, query):
        return self.query_embeddings.get(query, None)

    def cache_corpus_embeddings(self, corpus):
        embeddings = self.encode(corpus)
        self.corpus_embeddings = embeddings

    def get_corpus_embeddings(self):
        return self.corpus_embeddings

    def save_embeddings(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'query_embeddings': self.query_embeddings,
                'corpus_embeddings': self.corpus_embeddings
            }, f)

    def load_embeddings(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.query_embeddings = data['query_embeddings']
            self.corpus_embeddings = data['corpus_embeddings']


# russian stop-words
nltk.download('stopwords')

morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

def preprocess_text(text):

    # to lower
    text = text.lower()
    # removing punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # tokenizing
    tokens = text.split()
    # lemmatization, stop words removal
    lemmas = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token not in russian_stopwords
    ]
    return lemmas

# Lexical Retriever
class LexicalRetriever:
    def __init__(self, use_bm25=True, use_tfidf=True):
        self.use_bm25 = use_bm25
        self.use_tfidf = use_tfidf
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.corpus = None
        self.preprocessed_corpus = None
        self.query_scores = {}


    def fit(self, corpus):

        self.corpus = corpus

        if self._load_embeddings():
            print("Loaded embeddings from saved files.")
        else:
            print("Fitting and encoding corpus...")
            self.preprocessed_corpus = [" ".join(preprocess_text(doc)) for doc in tqdm(corpus, desc="Preprocessing corpus")]

            if self.use_bm25:
                tokenized_corpus = [doc.split() for doc in self.preprocessed_corpus]
                self.bm25 = BM25Okapi(tokenized_corpus)

            if self.use_tfidf:
                self.tfidf_vectorizer = TfidfVectorizer(analyzer="word")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_corpus)

            self._save_embeddings()

    def _save_embeddings(self, file_path):
        files = file_path + '/lexical_embeddings'

        if not os.path.exists(files):
            os.makedirs()

        with open(os.path.join(files, "lexical_embeddings.pkl"), 'wb') as f:
            pickle.dump({
              'bm25': self.bm25,
              'tfidf_vectorizer': self.tfidf_vectorizer,
              'tfidf_matrix': self.tfidf_matrix,
              'preprocessed_corpus': self.preprocessed_corpus,
              'query_scores': self.query_scores
            }, f)


    def _load_embeddings(self, file_path):
        files = file_path + '/lexical_embeddings'
        try:
            with open(os.path.join(files, "lexical_embeddings.pkl"), 'rb') as f:
              data = pickle.load(f)
              self.bm25 = data.get('bm25')
              self.tfidf_vectorizer = data.get('tfidf_vectorizer')
              self.tfidf_matrix = data.get('tfidf_matrix')
              self.preprocessed_corpus = data.get('preprocessed_corpus')
              self.query_scores = data.get('query_scores', {})
            return True
        except FileNotFoundError:
            return False


    def cache_query_scores(self, queries):
        for query in tqdm(queries, desc="Lexical scoring", leave=False):
            self.query_scores[query] = self.get_scores(query)

    def get_scores(self, query):

        preprocessed_query = preprocess_text(query)
        scores = {}

        if self.use_bm25:
            scores['bm25'] = np.array(self.bm25.get_scores(preprocessed_query))

        if self.use_tfidf:
            # tokens concatenation
            query_string = " ".join(preprocessed_query)
            query_vec = self.tfidf_vectorizer.transform([query_string])
            scores['tfidf'] = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        return scores

    def get_query_scores(self, query):
        return self.query_scores.get(query, None)

# Hybrid Scorer
class HybridScorer:
    def __init__(self, alpha=0.33, beta=0.33, gamma=0.34):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(self, bm25_scores, tfidf_scores, transformer_scores):
        scaler = MinMaxScaler()
        s_bm25 = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten() if isinstance(bm25_scores, np.ndarray) else 0
        s_tfidf = scaler.fit_transform(tfidf_scores.reshape(-1, 1)).flatten() if isinstance(tfidf_scores, np.ndarray) else 0
        s_transformer = scaler.fit_transform(transformer_scores.reshape(-1, 1)).flatten() if isinstance(transformer_scores, np.ndarray) else 0

        return self.alpha * s_bm25 + self.beta * s_tfidf + self.gamma * s_transformer