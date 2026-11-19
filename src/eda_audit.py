import re
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate

def audit_dataset(df, name, text_col, label_col, lang="id", random_seed=42):
    print("=" * 80)
    print(f"AUDIT LAPORAN: {name.upper()}")
    print("=" * 80)
    
    # 1. Dimensi, Duplikasi & Null
    raw_count = len(df)
    null_text = df[text_col].isnull().sum()
    null_label = df[label_col].isnull().sum()
    dup_count = df.duplicated(subset=[text_col]).sum()
    
    # Bersihkan null pada kolom teks dan label untuk analisis
    df_clean = df.dropna(subset=[text_col, label_col]).copy()
    clean_count = len(df_clean)
    
    # Hitung panjang kata
    df_clean['word_count'] = df_clean[text_col].astype(str).apply(lambda x: len(x.strip().split()))
    
    mean_words = df_clean['word_count'].mean()
    median_words = df_clean['word_count'].median()
    min_words = df_clean['word_count'].min()
    max_words = df_clean['word_count'].max()
    std_words = df_clean['word_count'].std()
    
    table_integrity = [
        ["Total Baris Mentah", f"{raw_count:,}"],
        ["Null pada Teks", f"{null_text:,} ({(null_text/raw_count)*100:.2f}%)"],
        ["Null pada Label", f"{null_label:,} ({(null_label/raw_count)*100:.2f}%)"],
        ["Duplikasi Teks", f"{dup_count:,} ({(dup_count/raw_count)*100:.2f}%)"],
        ["Baris Bersih (Siap Olah)", f"{clean_count:,} ({(clean_count/raw_count)*100:.2f}%)"],
        ["Rata-rata Panjang Kata", f"{mean_words:.2f} kata"],
        ["Median Panjang Kata", f"{median_words:.1f} kata"],
        ["Rentang Panjang Kata (Min - Max)", f"{min_words} - {max_words:,} kata (Std: {std_words:.2f})"]
    ]
    print("\n[1] DATA INTEGRITY & WORD LENGTH SUMMARY")
    print(tabulate(table_integrity, headers=["Metrik", "Nilai"], tablefmt="fancy_grid"))
    
    # 2. Distribusi Kelas Sentimen Eksak
    class_dist = df_clean[label_col].value_counts()
    class_pct = df_clean[label_col].value_counts(normalize=True) * 100
    
    table_dist = []
    for cls in class_dist.index:
        table_dist.append([str(cls), f"{class_dist[cls]:,}", f"{class_pct[cls]:.2f}%"])
        
    majority_count = class_dist.max()
    minority_count = class_dist.min()
    imbalance_ratio = majority_count / minority_count
    
    print("\n[2] EXACT SENTIMENT CLASS DISTRIBUTION")
    print(tabulate(table_dist, headers=["Kelas Sentimen", "Jumlah Baris", "Persentase"], tablefmt="fancy_grid"))
    print(f"-> Rasio Ketimpangan (Majority : Minority) = {imbalance_ratio:.2f} : 1")
    if imbalance_ratio > 10:
        print("   [PERINGATAN] Kelas sangat timpang! Perlu stratifikasi ketat & pembobotan class weight.")
    elif imbalance_ratio > 3:
        print("   [CATATAN] Kelas moderat timpang, ideal diukur dengan Macro F1-Score.")
    else:
        print("   [OPTIMAL] Kelas relatif seimbang.")

    # 3. Bukti Keaslian Teks (5 Sampel Acak per Kelas)
    print("\n[3] BUKTI KEASLIAN TEKS (5 Sampel Acak per Kelas - Seed 42)")
    np.random.seed(random_seed)
    for cls in class_dist.index:
        print(f"\n--- Sampel Kelas: {str(cls).upper()} ---")
        sub_df = df_clean[df_clean[label_col] == cls]
        sample_size = min(5, len(sub_df))
        samples = sub_df.sample(sample_size, random_state=random_seed)[text_col].tolist()
        for idx, text in enumerate(samples, 1):
            cleaned_preview = text.replace("\n", " ").strip()
            w_len = len(cleaned_preview.split())
            print(f"  {idx}. [{w_len} kata] \"{cleaned_preview}\"")

    # 4. Statistik Linguistik & OOV Potential
    # Tokenisasi sederhana
    all_words = []
    negation_count = 0
    question_count = 0
    
    if lang == "id":
        neg_patterns = r"\b(tidak|bukan|ga|gak|ngga|nggak|gk|blm|belum|tak|bkn|tdk)\b"
    else:
        neg_patterns = r"\b(not|no|never|none|neither|cannot|cant|can't|wont|won't|dont|don't|doesnt|doesn't|isnt|isn't|arent|aren't)\b"
    
    for text in df_clean[text_col]:
        text_str = str(text).lower()
        if "?" in text_str:
            question_count += 1
        if re.search(neg_patterns, text_str):
            negation_count += 1
            
        words = re.findall(r"\b\w+\b", text_str)
        all_words.extend(words)
        
    vocab_counter = Counter(all_words)
    total_tokens = len(all_words)
    total_unique_vocab = len(vocab_counter)
    
    # Kata langka (frekuensi < 3) -> indikator utama kata yang terancam OOV pada Word2Vec
    rare_words = [w for w, c in vocab_counter.items() if c < 3]
    rare_vocab_count = len(rare_words)
    rare_vocab_ratio = (rare_vocab_count / total_unique_vocab) * 100
    
    neg_pct = (negation_count / clean_count) * 100
    q_pct = (question_count / clean_count) * 100
    
    table_ling = [
        ["Total Token Kata", f"{total_tokens:,}"],
        ["Total Kosakata Unik (Vocabulary)", f"{total_unique_vocab:,}"],
        ["Kata Langka / Calon OOV (Frekuensi < 3)", f"{rare_vocab_count:,} ({rare_vocab_ratio:.2f}% dari vocab)"],
        ["Kalimat Mengandung Negasi", f"{negation_count:,} ({neg_pct:.2f}% dari ulasan)"],
        ["Kalimat Mengandung Pertanyaan ('?')", f"{question_count:,} ({q_pct:.2f}% dari ulasan)"]
    ]
    print("\n[4] STATISTIK LINGUISTIK & FENOMENA BAHASA (Amunisi Fitur Representasi)")
    print(tabulate(table_ling, headers=["Parameter Linguistik", "Temuan"], tablefmt="fancy_grid"))
    
    return {
        "name": name,
        "clean_count": clean_count,
        "mean_words": mean_words,
        "median_words": median_words,
        "imbalance_ratio": imbalance_ratio,
        "rare_ratio": rare_vocab_ratio,
        "neg_pct": neg_pct,
        "q_pct": q_pct
    }

def print_glove_technical_plan():
    print("\n" + "=" * 80)
    print("[5] RENCANA TEKNIS MEMUAT GloVe SESUAI RULES .agents/rules/")
    print("=" * 80)
    plan_info = [
        ["Bahasa Inggris (Twitter Airline)", 
         "Stanford GloVe Twitter 27B / 6B", 
         "1. Download `glove.twitter.27B.100d.txt` (terlatih pada 2 miliar tweet twitter asli).\n"
         "2. Konversi ke format Word2Vec menggunakan `gensim.scripts.glove2word2vec`:\n"
         "   `from gensim.scripts.glove2word2vec import glove2word2vec`\n"
         "   `glove2word2vec('glove.twitter.27B.100d.txt', 'glove_twitter_w2v.txt')`\n"
         "3. Muat via KeyedVectors: `KeyedVectors.load_word2vec_format('glove_twitter_w2v.txt')`\n"
         "4. Ekstrak vektor embedding dokumen via average pooling atau TF-IDF weighted sum."],
        
        ["Bahasa Indonesia (Kredivo 50k)", 
         "Indonesian Pretrained GloVe / Local Training", 
         "1. Opsi A (Pretrained External): Menggunakan Stanford GloVe yang di-train pada Wikipedia ID / Indo4B 50d/100d.\n"
         "2. Opsi B (Stanford GloVe C Source - Reproducible): Menjalankan binary official Stanford GloVe C-code\n"
         "   (vocab_count, cooccur, shuffle, glove) langsung pada korpus Kredivo untuk menghasilkan vektor 100d lokal murni.\n"
         "3. Format teks diubah ke Word2Vec header line (`<vocab_size> <vector_dim>`) agar bisa diload via `gensim.models.KeyedVectors`.\n"
         "4. Bandingkan secara adil dengan Word2Vec gensim & fastText gensim lokal pada korpus yang sama persis."]
    ]
    print(tabulate(plan_info, headers=["Bahasa & Dataset", "Sumber Bobot GloVe", "Langkah Integrasi Teknis"], tablefmt="fancy_grid"))

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    kredivo_path = data_dir / "kredivo_reviews_50k.csv"
    airline_path = data_dir / "Tweets.csv"
    
    print("Memuat dataset...")
    df_kredivo = pd.read_csv(kredivo_path)
    df_airline = pd.read_csv(airline_path)
    
    res_kredivo = audit_dataset(df_kredivo, "Kredivo Reviews 50k (Indonesia)", text_col="content", label_col="sentiment", lang="id")
    res_airline = audit_dataset(df_airline, "Twitter US Airline Sentiment (English)", text_col="text", label_col="airline_sentiment", lang="en")
    
    print_glove_technical_plan()
    
    # Perbandingan Head-to-Head
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD MATRIX PERBANDINGAN STRATEGIS")
    print("=" * 80)
    comparison = [
        ["Jumlah Baris Bersih", f"{res_kredivo['clean_count']:,}", f"{res_airline['clean_count']:,}"],
        ["Panjang Kata (Mean / Median)", f"{res_kredivo['mean_words']:.1f} / {res_kredivo['median_words']:.0f} kata", f"{res_airline['mean_words']:.1f} / {res_airline['median_words']:.0f} kata"],
        ["Rasio Ketimpangan Kelas", f"{res_kredivo['imbalance_ratio']:.2f} : 1 (Mayoritas Positif)", f"{res_airline['imbalance_ratio']:.2f} : 1 (Mayoritas Negatif)"],
        ["Rasio Kosakata Langka / Potensi OOV", f"{res_kredivo['rare_ratio']:.2f}%", f"{res_airline['rare_ratio']:.2f}%"],
        ["Kepadatan Kalimat Negasi", f"{res_kredivo['neg_pct']:.2f}%", f"{res_airline['neg_pct']:.2f}%"],
        ["Kepadatan Kalimat Tanya ('?')", f"{res_kredivo['q_pct']:.2f}%", f"{res_airline['q_pct']:.2f}%"]
    ]
    print(tabulate(comparison, headers=["Dimensi Audit", "Kredivo Reviews 50k", "Twitter Airline Sentiment"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
