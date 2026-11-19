import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

# Tambahkan path lokal nltk_data dalam workspace
local_nltk_dir = os.path.abspath("nltk_data")
if os.path.exists(local_nltk_dir) and local_nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk_dir)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inisialisasi NLTK lemmatizer dan stopwords
lemmatizer = WordNetLemmatizer()

NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing', 'cannot', 'none', 'nowhere'
}

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english')) - NEGATION_WORDS
except Exception:
    nltk.download('stopwords')
    ENGLISH_STOPWORDS = set(stopwords.words('english')) - NEGATION_WORDS

# Kamus ekspansi kontraksi bahasa Inggris
CONTRACTION_MAP = {
    r"can\'t": "can not",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
    r"ain\'t": "is not",
    r"gonna": "going to",
    r"wanna": "want to",
    r"gotta": "got to",
    r"lemme": "let me",
    r"kinda": "kind of"
}

def minimal_clean(text: str) -> str:
    """
    Skenario 1: Minimal Cleaning
    - Lowercase
    - Ganti URL -> <URL>
    - Ganti @mention maskapai -> <USER> (mencegah bias entitas maskapai)
    - Mempertahankan kontraksi asli (can't, won't, didn't)
    - Mempertahankan tanda baca emosi (! dan ?)
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. Ganti URL
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    
    # 2. Ganti @mention maskapai / user
    text = re.sub(r'@\w+', '<USER>', text)
    
    # 3. Pisahkan tanda baca emosi (! dan ?) dengan spasi agar menjadi token mandiri
    text = re.sub(r'([!?])', r' \1 ', text)
    
    # 4. Hapus karakter non-alfanumerik KECUALI apostrof untuk kontraksi, tag <URL>, <USER>, dan ! ?
    text = re.sub(r"[^a-zA-Z0-9'<>\s!?]", ' ', text)
    
    # 5. Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def full_norm(text: str) -> str:
    """
    Skenario 2: Full Normalization
    - Lowercase, URL -> <URL>, @mention -> <USER>
    - Ekspansi kontraksi (can't -> can not, won't -> will not, dll.)
    - Penghapusan stopwords bahasa Inggris KECUALI kata negasi
    - Lemmatization (WordNetLemmatizer)
    - Pembersihan tanda baca
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. Ganti URL & Mention
    text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    
    # 2. Ekspansi kontraksi
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(contraction, expansion, text)
    
    # 3. Hapus tanda baca (kecuali tag placeholder <URL> dan <USER>)
    tokens = []
    # Pisahkan kata dengan regex kata dan tag
    raw_tokens = re.findall(r'<[A-Z]+>|[a-z0-9]+', text)
    
    for token in raw_tokens:
        # 4. Filter stopwords (kecuali kata negasi yang wajib dipertahankan)
        if token in ENGLISH_STOPWORDS:
            continue
        # 5. Lemmatization
        if not token.startswith('<'):
            lemma = lemmatizer.lemmatize(token)
        else:
            lemma = token
        tokens.append(lemma)
        
    return ' '.join(tokens)

def load_and_split_data(data_path: str = "data/raw/Tweets.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Memuat dataset Twitter Airline, menghapus duplikat teks eksak,
    dan melakukan train_test_split (80% train, 20% test, stratify=y)
    SEBELUM dilakukan fitting vectorizer / embedding (bebas data leakage).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Pilih kolom target dan fitur
    df = df.dropna(subset=['text', 'airline_sentiment']).copy()
    
    # Hapus duplikat teks eksak
    total_before = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    total_after = len(df)
    dropped_dups = total_before - total_after
    
    X = df['text']
    y = df['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "total_raw": total_before,
        "total_unique": total_after,
        "dropped_dups": dropped_dups
    }

if __name__ == "__main__":
    print("=" * 80)
    print("VERIFIKASI PIPELINE PREPROCESSING & DATA SPLIT")
    print("=" * 80)
    
    split_data = load_and_split_data()
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    
    print(f"\n1. Integritas Data & Split:")
    print(f"   - Total Baris Mentah        : {split_data['total_raw']:,}")
    print(f"   - Duplikat Dieliminasi      : {split_data['dropped_dups']:,}")
    print(f"   - Total Baris Bersih Unik   : {split_data['total_unique']:,}")
    print(f"   - Ukuran Train (80%)        : {len(X_train):,} baris")
    print(f"   - Ukuran Test  (20%)        : {len(X_test):,} baris")
    
    print(f"\n2. Verifikasi Stratifikasi Kelas:")
    train_dist = y_train.value_counts(normalize=True) * 100
    test_dist = y_test.value_counts(normalize=True) * 100
    for cls in train_dist.index:
        print(f"   - Kelas '{cls:<8}': Train = {train_dist[cls]:.2f}% | Test = {test_dist[cls]:.2f}%")
        
    print(f"\n3. Tampilan 5 Sampel Komparasi Tokenisasi ('minimal_clean' vs 'full_norm'):")
    sample_indices = [0, 15, 42, 108, 250]
    for idx, sample_idx in enumerate(sample_indices, 1):
        raw_text = X_train.iloc[sample_idx]
        label = y_train.iloc[sample_idx]
        min_text = minimal_clean(raw_text)
        fn_text = full_norm(raw_text)
        
        print(f"\n--- Sampel #{idx} [Label: {label.upper()}] ---")
        print(f"  [Original]     : {raw_text}")
        print(f"  [minimal_clean]: {min_text}")
        print(f"  [full_norm]    : {fn_text}")
