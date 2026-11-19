import os
import gensim.downloader as api
from gensim.models import KeyedVectors

def get_or_download_glove():
    glove_dir = os.path.join("models", "glove")
    os.makedirs(glove_dir, exist_ok=True)
    glove_kv_path = os.path.join(glove_dir, "glove-twitter-100.kv")
    
    if os.path.exists(glove_kv_path):
        print(f"Loading cached GloVe model from {glove_kv_path}...")
        return KeyedVectors.load(glove_kv_path, mmap='r')
    
    print("Downloading 'glove-twitter-100' via gensim.downloader...")
    model = api.load("glove-twitter-100")
    print(f"Caching GloVe model to {glove_kv_path}...")
    model.save(glove_kv_path)
    print("GloVe model cached successfully!")
    return model

if __name__ == "__main__":
    model = get_or_download_glove()
    print(f"GloVe loaded. Vocab size: {len(model.key_to_index):,}, Vector dimension: {model.vector_size}")
