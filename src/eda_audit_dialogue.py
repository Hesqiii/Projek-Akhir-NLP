import os
import re
from collections import Counter
import pandas as pd
import numpy as np
from tabulate import tabulate

def audit_meld():
    print("=" * 80)
    print("AUDIT DATASET 1: MELD (Multimodal EmotionLines Dataset)")
    print("=" * 80)
    
    base_dir = os.path.join("data", "raw", "meld", "JSON files", "JSON files", "CSV Processed")
    train_path = os.path.join(base_dir, "train_sent_emo_cleaned_processed.csv")
    dev_path = os.path.join(base_dir, "dev_sent_emo_cleaned_processed.csv")
    test_path = os.path.join(base_dir, "test_sent_emo_cleaned_processed.csv")
    
    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)
    
    table_files = [
        ["train_sent_emo_cleaned_processed.csv", f"{len(df_train):,} baris", "Training Set"],
        ["dev_sent_emo_cleaned_processed.csv", f"{len(df_dev):,} baris", "Validation / Dev Set"],
        ["test_sent_emo_cleaned_processed.csv", f"{len(df_test):,} baris", "Evaluation Test Set"],
        ["TOTAL GABUNGAN", f"{len(df_train) + len(df_dev) + len(df_test):,} baris", "Full Dialogue Corpus"]
    ]
    print("\n[1] STRUKTUR FILE & DATA SPLITS")
    print(tabulate(table_files, headers=["Nama File", "Jumlah Baris", "Fungsi"], tablefmt="fancy_grid"))
    
    df = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    
    # Kolom utama
    print(f"\nKolom Tersedia: {df.columns.tolist()}")
    print("-> Kolom Teks Ujaran   : 'Utterance'")
    print("-> ID Percakapan & Turn : 'Dialogue_ID', 'Utterance_ID', 'Speaker'")
    print("-> Kolom Target Sentimen: 'Sentiment' (Positive, Negative, Neutral)")
    print("-> Kolom Target Emosi   : 'Emotion' (7 kelas)")
    
    # Integritas & Panjang
    null_text = df['Utterance'].isnull().sum()
    null_sent = df['Sentiment'].isnull().sum()
    null_emo = df['Emotion'].isnull().sum()
    dup_count = df.duplicated(subset=['Utterance']).sum()
    
    df_clean = df.dropna(subset=['Utterance', 'Sentiment', 'Emotion']).copy()
    df_clean['word_count'] = df_clean['Utterance'].astype(str).apply(lambda x: len(x.strip().split()))
    
    mean_len = df_clean['word_count'].mean()
    median_len = df_clean['word_count'].median()
    ge8_count = (df_clean['word_count'] >= 8).sum()
    ge8_pct = (ge8_count / len(df_clean)) * 100
    
    table_integ = [
        ["Total Ujaran Bersih", f"{len(df_clean):,}"],
        ["Null pada Teks / Sentimen / Emosi", f"{null_text} / {null_sent} / {null_emo}"],
        ["Duplikasi Ujaran Eksak", f"{dup_count:,} ({(dup_count/len(df))*100:.2f}%)"],
        ["Rata-rata Panjang Kata", f"{mean_len:.2f} kata"],
        ["Median Panjang Kata", f"{median_len:.1f} kata"],
        ["Ujaran Panjang (>= 8 kata)", f"{ge8_count:,} ({ge8_pct:.2f}%)"]
    ]
    print("\n[2] AUDIT INTEGRITAS & PANJANG UJARAN")
    print(tabulate(table_integ, headers=["Metrik", "Nilai"], tablefmt="fancy_grid"))
    
    # Distribusi Kelas Sentimen & Emosi
    sent_dist = df_clean['Sentiment'].str.lower().value_counts()
    sent_pct = df_clean['Sentiment'].str.lower().value_counts(normalize=True) * 100
    
    table_sent = []
    for k in sent_dist.index:
        table_sent.append([k.capitalize(), f"{sent_dist[k]:,}", f"{sent_pct[k]:.2f}%"])
    print("\n[3A] DISTRIBUSI KELAS SENTIMEN EKSAT")
    print(tabulate(table_sent, headers=["Kelas Sentimen", "Jumlah", "Persentase"], tablefmt="fancy_grid"))
    
    emo_dist = df_clean['Emotion'].str.lower().value_counts()
    emo_pct = df_clean['Emotion'].str.lower().value_counts(normalize=True) * 100
    table_emo = []
    for k in emo_dist.index:
        table_emo.append([k.capitalize(), f"{emo_dist[k]:,}", f"{emo_pct[k]:.2f}%"])
    print("\n[3B] DISTRIBUSI 7 KELAS EMOSI")
    print(tabulate(table_emo, headers=["Kelas Emosi", "Jumlah", "Persentase"], tablefmt="fancy_grid"))
    
    # Statistik Linguistik
    contraction_pattern = r"\b(can't|cannot|won't|don't|doesn't|didn't|i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|i'll|you'll|he'll|she'll|we'll|they'll|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|wouldn't|shouldn't|couldn't|gonna|wanna|gotta)\b"
    negation_pattern = r"\b(not|no|never|none|neither|nothing|nobody|nowhere|nor)\b"
    
    all_words = []
    contract_cnt = 0
    neg_cnt = 0
    
    for text in df_clean['Utterance']:
        t = str(text).lower()
        if re.search(contraction_pattern, t):
            contract_cnt += 1
        if re.search(negation_pattern, t):
            neg_cnt += 1
        words = re.findall(r"\b\w+\b", t)
        all_words.extend(words)
        
    vocab_counter = Counter(all_words)
    total_tokens = len(all_words)
    unique_vocab = len(vocab_counter)
    rare_words = [w for w, c in vocab_counter.items() if c < 3]
    rare_ratio = (len(rare_words) / unique_vocab) * 100
    
    table_ling = [
        ["Total Token Kata", f"{total_tokens:,}"],
        ["Total Kosakata Unik (Vocab)", f"{unique_vocab:,}"],
        ["Kata Langka / Calon OOV (Freq < 3)", f"{len(rare_words):,} ({rare_ratio:.2f}% dari vocab)"],
        ["Ujaran Mengandung Kontraksi Slang", f"{contract_cnt:,} ({(contract_cnt/len(df_clean))*100:.2f}%)"],
        ["Ujaran Mengandung Negasi", f"{neg_cnt:,} ({(neg_cnt/len(df_clean))*100:.2f}%)"]
    ]
    print("\n[4] STATISTIK LINGUISTIK (AMUNISI REPRESENTASI TEKS)")
    print(tabulate(table_ling, headers=["Parameter Linguistik", "Nilai"], tablefmt="fancy_grid"))
    
    # Bukti Keaslian Dialog (3 Pasangan Turn Berurutan)
    print("\n[5] BUKTI PASANGAN DIALOG BERURUTAN (Turn[t] -> Turn[t+1])")
    dialogues_sampled = df_clean['Dialogue_ID'].unique()[:3]
    for d_id in dialogues_sampled:
        sub = df_clean[df_clean['Dialogue_ID'] == d_id].sort_values('Utterance_ID')
        if len(sub) >= 2:
            turn1 = sub.iloc[0]
            turn2 = sub.iloc[1]
            print(f"\n--- Dialogue ID #{d_id} ---")
            print(f"  [Input]  {turn1['Speaker']}: \"{turn1['Utterance']}\" [Sentimen: {turn1['Sentiment']}, Emosi: {turn1['Emotion']}]")
            print(f"  [Output] {turn2['Speaker']}: \"{turn2['Utterance']}\" [Sentimen: {turn2['Sentiment']}, Emosi: {turn2['Emotion']}]")
            
    return {
        "name": "MELD",
        "clean_rows": len(df_clean),
        "mean_len": mean_len,
        "median_len": median_len,
        "ge8_pct": ge8_pct,
        "rare_ratio": rare_ratio,
        "contract_pct": (contract_cnt/len(df_clean))*100,
        "neg_pct": (neg_cnt/len(df_clean))*100,
        "classes": len(sent_dist)
    }

def audit_empathetic():
    print("\n" + "=" * 80)
    print("AUDIT DATASET 2: EMPATHETIC DIALOGUES (Facebook AI)")
    print("=" * 80)
    
    csv_path = os.path.join("data", "raw", "empathetic", "emotion-emotion_69k.csv")
    df_raw = pd.read_csv(csv_path)
    
    raw_count = len(df_raw)
    print(f"\n[1] STRUKTUR FILE: emotion-emotion_69k.csv ({raw_count:,} baris)")
    print(f"Kolom Mentah: {df_raw.columns.tolist()[:5]}")
    
    valid_32_emotions = {
        'surprised', 'excited', 'angry', 'proud', 'sad', 'annoyed', 'lonely', 'afraid',
        'grateful', 'terrified', 'guilty', 'furious', 'disgusted', 'confident', 'anxious',
        'anticipating', 'hopeful', 'impressed', 'nostalgic', 'disappointed', 'jealous',
        'joyful', 'prepared', 'content', 'devastated', 'embarrassed', 'sentimental',
        'caring', 'trusting', 'ashamed', 'apprehensive', 'faithful'
    }
    
    # Ekstraksi Customer Utterance dan Agent Label
    df = df_raw.dropna(subset=['empathetic_dialogues', 'labels', 'emotion']).copy()
    # Filter hanya 32 emosi valid (menghilangkan baris CSV leaking delimiter)
    df = df[df['emotion'].isin(valid_32_emotions)].copy()
    
    df['cust_utt'] = df['empathetic_dialogues'].astype(str).str.replace(r'^Customer\s*:', '', regex=True).str.replace(r'\nAgent\s*:', '', regex=True).str.strip()
    df['agent_reply'] = df['labels'].astype(str).str.strip()
    
    print("\n-> Kolom Teks Input (Customer) : 'cust_utt' (diekstrak dari 'empathetic_dialogues')")
    print("-> Kolom Teks Output (Agent)    : 'agent_reply' ('labels')")
    print("-> Kolom Konteks Situasi        : 'Situation'")
    print("-> Kolom Target Emosi           : 'emotion' (32 kelas emosi)")
    
    clean_count = len(df)
    null_count = df['cust_utt'].isnull().sum() + df['agent_reply'].isnull().sum()
    dup_cust = df.duplicated(subset=['cust_utt']).sum()
    
    df['word_count'] = df['cust_utt'].apply(lambda x: len(x.split()))
    df['agent_word_count'] = df['agent_reply'].apply(lambda x: len(x.split()))
    
    mean_len = df['word_count'].mean()
    median_len = df['word_count'].median()
    ge8_count = (df['word_count'] >= 8).sum()
    ge8_pct = (ge8_count / clean_count) * 100
    
    table_integ = [
        ["Total Pasangan Dialog Bersih", f"{clean_count:,}"],
        ["Baris Tereliminasi (Delimiter Leaks)", f"{raw_count - clean_count:,} ({((raw_count - clean_count)/raw_count)*100:.2f}%)"],
        ["Null pada Teks / Respon", f"{null_count}"],
        ["Duplikasi Input Eksak", f"{dup_cust:,} ({(dup_cust/clean_count)*100:.2f}%)"],
        ["Rata-rata Panjang Kata (Customer)", f"{mean_len:.2f} kata (Agent: {df['agent_word_count'].mean():.2f} kata)"],
        ["Median Panjang Kata (Customer)", f"{median_len:.1f} kata (Agent: {df['agent_word_count'].median():.1f} kata)"],
        ["Ujaran Panjang (>= 8 kata)", f"{ge8_count:,} ({ge8_pct:.2f}%)"]
    ]
    print("\n[2] AUDIT INTEGRITAS & PANJANG UJARAN")
    print(tabulate(table_integ, headers=["Metrik", "Nilai"], tablefmt="fancy_grid"))
    
    # Distribusi 32 Emosi
    emo_counts = df['emotion'].value_counts()
    print("\n[3A] DISTRIBUSI 32 KELAS EMOSI (TOP 10 DITAMPILKAN)")
    table_top10_emo = [[k.capitalize(), f"{emo_counts[k]:,}", f"{(emo_counts[k]/clean_count)*100:.2f}%"] for k in emo_counts.index[:10]]
    print(tabulate(table_top10_emo, headers=["Emosi", "Jumlah", "Persentase"], tablefmt="fancy_grid"))
    print(f"-> Total Emosi: 32 kelas unik. Rata-rata per kelas: ~{clean_count/32:.0f} ulasan (seimbang sempurna ~3.1% per kelas).")
    
    # Pemetaan Deterministik ke 3 Kelas Sentimen
    positive_emotions = {
        'joyful', 'excited', 'proud', 'grateful', 'impressed', 'hopeful',
        'confident', 'content', 'caring', 'trusting', 'faithful'
    }
    negative_emotions = {
        'angry', 'furious', 'annoyed', 'sad', 'lonely', 'devastated',
        'disappointed', 'disgusted', 'guilty', 'ashamed', 'jealous',
        'anxious', 'afraid', 'terrified', 'embarrassed'
    }
    neutral_emotions = {
        'surprised', 'anticipating', 'nostalgic', 'sentimental', 'prepared', 'apprehensive'
    }
    
    def map_sentiment(e):
        if e in positive_emotions:
            return 'positive'
        elif e in negative_emotions:
            return 'negative'
        else:
            return 'neutral'
            
    df['sentiment_mapped'] = df['emotion'].apply(map_sentiment)
    sent_dist = df['sentiment_mapped'].value_counts()
    sent_pct = df['sentiment_mapped'].value_counts(normalize=True) * 100
    
    table_mapped = [
        ["Negative", f"{sent_dist['negative']:,}", f"{sent_pct['negative']:.2f}%", "15 emosi (angry, sad, afraid, guilty, dll.)"],
        ["Positive", f"{sent_dist['positive']:,}", f"{sent_pct['positive']:.2f}%", "11 emosi (joyful, proud, grateful, dll.)"],
        ["Neutral", f"{sent_dist['neutral']:,}", f"{sent_pct['neutral']:.2f}%", "6 emosi (surprised, anticipating, nostalgic, dll.)"]
    ]
    print("\n[3B] PEMETAAN DETERMINISTIK KE 3 KELAS SENTIMEN (SIMULASI)")
    print(tabulate(table_mapped, headers=["Kelas Sentimen", "Jumlah Baris", "Persentase", "Cakupan Emosi"], tablefmt="fancy_grid"))
    
    # Statistik Linguistik
    contraction_pattern = r"\b(can't|cannot|won't|don't|doesn't|didn't|i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|i'll|you'll|he'll|she'll|we'll|they'll|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|wouldn't|shouldn't|couldn't|gonna|wanna|gotta)\b"
    negation_pattern = r"\b(not|no|never|none|neither|nothing|nobody|nowhere|nor)\b"
    
    all_words = []
    contract_cnt = 0
    neg_cnt = 0
    
    for text in df['cust_utt']:
        t = str(text).lower()
        if re.search(contraction_pattern, t):
            contract_cnt += 1
        if re.search(negation_pattern, t):
            neg_cnt += 1
        words = re.findall(r"\b\w+\b", t)
        all_words.extend(words)
        
    vocab_counter = Counter(all_words)
    total_tokens = len(all_words)
    unique_vocab = len(vocab_counter)
    rare_words = [w for w, c in vocab_counter.items() if c < 3]
    rare_ratio = (len(rare_words) / unique_vocab) * 100
    
    table_ling = [
        ["Total Token Kata", f"{total_tokens:,}"],
        ["Total Kosakata Unik (Vocab)", f"{unique_vocab:,}"],
        ["Kata Langka / Calon OOV (Freq < 3)", f"{len(rare_words):,} ({rare_ratio:.2f}% dari vocab)"],
        ["Ujaran Mengandung Kontraksi Slang", f"{contract_cnt:,} ({(contract_cnt/clean_count)*100:.2f}%)"],
        ["Ujaran Mengandung Negasi", f"{neg_cnt:,} ({(neg_cnt/clean_count)*100:.2f}%)"]
    ]
    print("\n[4] STATISTIK LINGUISTIK (AMUNISI REPRESENTASI TEKS)")
    print(tabulate(table_ling, headers=["Parameter Linguistik", "Nilai"], tablefmt="fancy_grid"))
    
    # Bukti Keaslian Dialog (3 Contoh Nyata Pasangan Dialog)
    print("\n[5] BUKTI PASANGAN DIALOG BERURUTAN (Customer -> Empathetic Agent)")
    samples = df.sample(3, random_state=42)
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n--- Contoh #{idx} (Emosi: {row['emotion'].upper()} -> Sentimen: {row['sentiment_mapped'].upper()}) ---")
        print(f"  Situasi Konteks: \"{row['Situation']}\"")
        print(f"  [Customer Input]: \"{row['cust_utt']}\"")
        print(f"  [Agent Response]: \"{row['agent_reply']}\"")
        
    return {
        "name": "EmpatheticDialogues",
        "clean_rows": clean_count,
        "mean_len": mean_len,
        "median_len": median_len,
        "ge8_pct": ge8_pct,
        "rare_ratio": rare_ratio,
        "contract_pct": (contract_cnt/clean_count)*100,
        "neg_pct": (neg_cnt/clean_count)*100,
        "classes": 3
    }

def main():
    res_meld = audit_meld()
    res_emp = audit_empathetic()
    
    print("\n" + "=" * 80)
    print("MATRIKS PERBANDINGAN STRATEGIS HEAD-TO-HEAD")
    print("=" * 80)
    matrix = [
        ["Jumlah Pasangan Dialog Bersih", f"{res_meld['clean_rows']:,}", f"{res_emp['clean_rows']:,}"],
        ["Panjang Rata-rata Kata", f"{res_meld['mean_len']:.2f} kata", f"{res_emp['mean_len']:.2f} kata"],
        ["Panjang Median Kata", f"{res_meld['median_len']:.1f} kata", f"{res_emp['median_len']:.1f} kata"],
        ["Persentase Ujaran Panjang (>= 8 kata)", f"{res_meld['ge8_pct']:.2f}%", f"{res_emp['ge8_pct']:.2f}%"],
        ["Rasio Kosakata Langka / Calon OOV", f"{res_meld['rare_ratio']:.2f}%", f"{res_emp['rare_ratio']:.2f}%"],
        ["Kepadatan Kontraksi Slang Bahasa Inggris", f"{res_meld['contract_pct']:.2f}%", f"{res_emp['contract_pct']:.2f}%"],
        ["Kepadatan Kalimat Negasi", f"{res_meld['neg_pct']:.2f}%", f"{res_emp['neg_pct']:.2f}%"],
        ["Kesiapan Kolom Chatbot (Input & Output)", "Tersedia via urutan Utterance_ID", "Tersedia langsung (cust_utt & agent_reply)"],
        ["Kesiapan Kolom Sentimen", "Eksplisit ('Sentiment' 3-kelas)", "Tersedia via mapping 32 emosi"]
    ]
    print(tabulate(matrix, headers=["Dimensi Evaluasi", "MELD (Friends TV)", "EmpatheticDialogues (Facebook AI)"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
