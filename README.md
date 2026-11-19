# Analisis Sentimen Tweet Maskapai

Proyek ini membandingkan enam representasi teks untuk mengklasifikasikan tweet menjadi `negative`, `neutral`, atau `positive`. Eksperimen memakai split train/test stratified 80:20 dengan `random_state=42`, lalu mengukur Accuracy dan Macro F1.

## Pembagian tugas debat

Gunakan tabel ini sebagai panduan belajar dan persiapan presentasi. Angka rinci dapat dilihat di `results/benchmark_summary.csv`; setiap anggota sebaiknya memahami alasan di balik hasil bagiannya, bukan hanya menghafal angka.

| Orang | Tugas utama saat presentasi | Yang harus dikuasai (angka & konsep) | File/bukti yang dibuka saat presentasi |
|---|---|---|---|
| **Kamu — koordinator dan pembuka** | Menjelaskan tujuan, dataset, pembagian data, lalu menyambungkan bagian teman-teman | Target kita adalah `airline_sentiment` dengan tiga kelas: negative, neutral, positive. Setelah 213 teks duplikat dibuang, data dibagi menjadi 11.541 train dan 2.886 test. | Satu diagram alur eksperimen dan ringkasan hasil |
| **Anggota 2 — TF-IDF dan preprocessing** | Menjelaskan metode terbaik dan alasan preprocessing dipertahankan | TF-IDF + LinearSVC + `minimal_clean` mendapat Macro F1 **0,7495**. Hasil rata-rata `minimal_clean` juga lebih baik daripada `full_norm`: **0,6502 vs 0,6195**. | Tabel ringkas BoW vs TF-IDF, contoh teks sebelum/sesudah preprocessing |
| **Anggota 3 — embedding** | Menjelaskan Word2Vec, GloVe, fastText, pooling, dan OOV | Skip-gram mendapat Macro F1 **0,6888**, lebih tinggi dari CBOW **0,6588**. fastText mencatat OOV **0,00%** pada test menurut definisi metrik di kode proyek. | Diagram sederhana cara kerja embedding dan tabel perbandingan |
| **Anggota 4 — pembanding dan penanya** | Memetakan perbedaan tugas lawan dan menyiapkan pertanyaan yang adil | Kita memprediksi **sentimen**; lawan memprediksi **kategori/alasan keluhan**. Dataset sama, tetapi targetnya berbeda. | Tabel “tugas kita vs tugas lawan” dan 3 pertanyaan untuk sesi sanggahan |
| **Anggota 5 — audit kesalahan dan penutup** | Menunjukkan contoh prediksi yang keliru, menjelaskan keterbatasan, lalu menutup presentasi | Model belum sempurna; satu skor tidak menjelaskan semua jenis kesalahan. Siapkan contoh kesalahan dari data test, jangan mengarang contoh. | Confusion matrix, 5–10 contoh salah prediksi yang sudah diperiksa, kesimpulan 30–45 detik |

## Persiapan

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Dataset lokal `data/raw/Tweets.csv` dengan kolom `text`, `airline_sentiment`, `negativereason`, `airline`, dan `tweet_id`

Dataset Twitter US Airline Sentiment berasal dari CrowdFlower. Salah satu salinan publik 14.640 baris tersedia di [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). Dataset dan model GloVe tidak disimpan di repository; `.gitignore` menjaga salinan lokal tetap di komputer pengguna.

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
