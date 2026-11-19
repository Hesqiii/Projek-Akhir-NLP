import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors

def get_tokens(text_series):
    """Tokenisasi sederhana berbasis split spasi."""
    return [str(text).split() for text in text_series]

class FeatureExtractor:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 3, seed: int = 42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.glove_model = None
        
    def load_glove(self, glove_path: str = "models/glove/glove-twitter-100.kv"):
        if self.glove_model is None:
            if not os.path.exists(glove_path):
                from src.setup_glove import get_or_download_glove
                self.glove_model = get_or_download_glove()
            else:
                self.glove_model = KeyedVectors.load(glove_path, mmap='r')
        return self.glove_model

    def build_bow(self, X_train, X_test):
        """1. Bag-of-Words (CountVectorizer ngram 1-2)."""
        vec = CountVectorizer(ngram_range=(1, 2), min_df=self.min_count)
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)
        return X_tr, X_te, 0.0

    def build_tfidf(self, X_train, X_test):
        """2. TF-IDF (TfidfVectorizer ngram 1-2)."""
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_count)
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)
        return X_tr, X_te, 0.0

    def train_w2v(self, X_train, sg: int = 0):
        """3 & 4. Word2Vec (sg=0 CBOW, sg=1 Skip-gram) hanya dilatih pada X_train."""
        tokens_train = get_tokens(X_train)
        model = Word2Vec(
            sentences=tokens_train,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=sg,
            seed=self.seed,
            workers=1
        )
        return model

    def train_fasttext(self, X_train):
        """6. fastText (min_n=3, max_n=6) hanya dilatih pada X_train."""
        tokens_train = get_tokens(X_train)
        model = FastText(
            sentences=tokens_train,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            min_n=3,
            max_n=6,
            seed=self.seed,
            workers=1
        )
        return model

    def calculate_oov_rate(self, model, X_test, is_fasttext: bool = False):
        """
        Menghitung persentase token OOV pada X_test.
        Untuk Word2Vec dan GloVe: token dihitung OOV jika tidak ada di key_to_index.
        Untuk fastText: subword n-gram menangani kata baru sehingga OOV = 0%.
        """
        if is_fasttext:
            return 0.0
            
        total_tokens = 0
        oov_tokens = 0
        kv = model.wv if hasattr(model, 'wv') else model
        
        for text in X_test:
            tokens = str(text).split()
            for token in tokens:
                total_tokens += 1
                if token not in kv.key_to_index:
                    oov_tokens += 1
                    
        return (oov_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    def pool_dense(self, model, texts, pooling: str = 'mean', tfidf_vec=None, is_fasttext: bool = False):
        """
        Menghitung vektor representasi dokumen dense (Mean Pooling atau TF-IDF Weighted Pooling).
        """
        kv = model.wv if hasattr(model, 'wv') else model
        tokens_list = get_tokens(texts)
        num_docs = len(texts)
        doc_vectors = np.zeros((num_docs, self.vector_size), dtype=np.float32)
        
        if pooling == 'tfidf' and tfidf_vec is not None:
            tfidf_mat = tfidf_vec.transform(texts)
            feature_names = tfidf_vec.get_feature_names_out()
            feature_to_idx = {name: i for i, name in enumerate(feature_names)}
            
            for i, tokens in enumerate(tokens_list):
                if not tokens:
                    continue
                weights = []
                vectors = []
                row = tfidf_mat.getrow(i)
                row_indices = row.indices
                row_data = row.data
                idx_to_weight = dict(zip(row_indices, row_data))
                
                for token in tokens:
                    weight = 0.0
                    if token in feature_to_idx:
                        col_idx = feature_to_idx[token]
                        weight = idx_to_weight.get(col_idx, 0.0)
                    if weight == 0.0:
                        weight = 0.1 # default minimum weight
                        
                    if is_fasttext:
                        vec = model.wv[token]
                        vectors.append(vec)
                        weights.append(weight)
                    elif token in kv.key_to_index:
                        vec = kv[token]
                        vectors.append(vec)
                        weights.append(weight)
                        
                if vectors:
                    weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
                    weighted_sum = np.sum(np.array(vectors) * weights, axis=0)
                    sum_weights = np.sum(weights)
                    if sum_weights > 0:
                        doc_vectors[i] = weighted_sum / sum_weights
                    else:
                        doc_vectors[i] = np.mean(vectors, axis=0)
        else: # mean pooling
            for i, tokens in enumerate(tokens_list):
                if not tokens:
                    continue
                vectors = []
                for token in tokens:
                    if is_fasttext:
                        vectors.append(model.wv[token])
                    elif token in kv.key_to_index:
                        vectors.append(kv[token])
                        
                if vectors:
                    doc_vectors[i] = np.mean(vectors, axis=0)
                    
        return doc_vectors
