# Analisis Sentimen Tweet Maskapai

Proyek ini membandingkan enam representasi teks untuk mengklasifikasikan tweet menjadi `negative`, `neutral`, atau `positive`. Eksperimen memakai split train/test stratified 80:20 dengan `random_state=42`, lalu mengukur Accuracy dan Macro F1.

## Persiapan

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Dataset lokal `data/raw/Tweets.csv` dengan kolom `text`, `airline_sentiment`, `negativereason`, `airline`, dan `tweet_id`

Dataset Twitter US Airline Sentiment berasal dari CrowdFlower. Salah satu salinan publik 14.640 baris tersedia di [Hugging Face](https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment). Dataset dan model GloVe tidak disimpan di repository; `.gitignore` menjaga salinan lokal tetap di komputer pengguna.

Siapkan environment dari root proyek:

```powershell
uv sync
```

Letakkan CSV di `data/raw/Tweets.csv`. Model `glove-twitter-100` akan diunduh dan disimpan lokal saat pertama kali diperlukan oleh Gensim.

## Menjalankan

Jalankan seluruh benchmark (80 konfigurasi):

```powershell
uv run python src/train_eval.py
```

Hasil benchmark disimpan ke `results/benchmark_summary.csv`.

Buat grafik dan dokumen bukti yang dapat ditelusuri dari benchmark dan dataset:

```powershell
uv run python src/debate_prep.py
```

Skrip membangun ulang konfigurasi TF-IDF terbaik pada split tetap dan memastikan Accuracy serta Macro F1 cocok dengan CSV sebelum menghasilkan confusion matrix. Hasilnya:

- `results/plots/representations_comparison.png`
- `results/plots/preprocessing_ablation.png`
- `results/plots/confusion_matrix_tfidf.png`
- `results/debate_ammunition.md`

Contoh tweet yang berpotensi sarkastik atau memuat beberapa aspek keluhan perlu diperiksa manusia sebelum dipresentasikan. Pencocokan kata untuk kategori keluhan hanya petunjuk, bukan label manual.

## Struktur kode

- `src/preprocessing.py`: pembersihan teks, penghapusan duplikat, dan split data.
- `src/embeddings.py`: BoW, TF-IDF, Word2Vec, GloVe, fastText, dan pooling.
- `src/train_eval.py`: benchmark classifier dan penyimpanan metrik.
- `src/debate_prep.py`: grafik, confusion matrix, sampel OOV, dan audit kesalahan.
