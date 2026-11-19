# Analisis Sentimen Tweet Maskapai

Proyek ini membandingkan enam representasi teks untuk mengklasifikasikan tweet menjadi `negative`, `neutral`, atau `positive`. Eksperimen memakai split train/test stratified 80:20 dengan `random_state=42`, lalu mengukur Accuracy dan Macro F1.

## Pembagian tugas debat

Gunakan tabel ini sebagai panduan belajar dan persiapan presentasi. Angka rinci dapat dilihat di `results/benchmark_summary.csv`; setiap anggota sebaiknya memahami alasan di balik hasil bagiannya, bukan hanya menghafal angka.

| Orang | Tugas utama saat presentasi | Yang harus dikuasai (angka & konsep) | File/bukti yang dibuka saat presentasi |
|---|---|---|---|
| **Ketua — koordinator & pembuka** | Menjelaskan tujuan proyek, asal dataset, alur pembagian data train/test, serta memetakan batas tugas kita vs tugas lawan | Target kita: `airline_sentiment` (3 kelas: negative, neutral, positive). Setelah 213 teks duplikat dibuang, data dibagi 11.541 train / 2.886 test secara stratified dengan `random_state=42`. Kita mengerjakan **Sentiment Analysis**, sedangkan kelompok lawan mengerjakan **Text Classification** pada alasan keluhan (`negativereason`) dengan dataset yang sama persis. | Diagram alur eksperimen; ringkasan cepat dari `results/benchmark_summary.csv` |
| **Anggota 2 — TF-IDF & preprocessing** | Menjelaskan perbandingan representasi sparse (BoW vs TF-IDF) dan studi ablasi preprocessing | Model terbaik proyek: TF-IDF + LinearSVC + `minimal_clean` (Macro F1 **0,7495**, Akurasi **80,87%**). Pada 40 konfigurasi teruji, rata-rata `minimal_clean` unggul konsisten dibanding `full_norm` (**0,6502 vs 0,6195**, selisih +0,0307). Alasan linguistik: tanda seru/tanya (`!`, `?`) dan kontraksi asli (`can't`, `won't`) membawa muatan emosi yang hilang jika dilemmatisasi/dibersihkan agresif. | `results/plots/representations_comparison.png`; `results/plots/preprocessing_ablation.png`; sampel pembersihan dari `src/preprocessing.py` |
| **Anggota 3 — embedding** | Menjelaskan arsitektur dense embedding (Word2Vec CBOW vs Skip-gram, GloVe Twitter, fastText), pooling (Mean vs TF-IDF), dan penanganan OOV | Skip-gram (**0,6888**) unggul atas CBOW (**0,6588**) karena lebih handal untuk kata langka. GloVe Twitter 100d mencapai **0,6836** (terhambat OOV 18,51% pada domain maskapai). fastText mencatat OOV **0,00%** di metrik kode karena dukungan subword n-gram ($n=3..6$), namun secara ilmiah ini bukan bukti bahwa seluruh vektor hasil subword akurat secara semantik (bisa terdistorsi kesamaan karakter leksikal). | `results/plots/representations_comparison.png`; Tabel 15 token OOV di `results/debate_ammunition.md` bagian 3 |
| **Anggota 4 — pembanding & penanya** | Membedah kelemahan perumusan masalah kelompok lawan dan melontarkan pertanyaan teknis yang adil saat sesi debat | Tugas kita mengukur sentimen global (jelas dan ortogonal). Tugas lawan memaksakan single-label multiclass pada keluhan (`negativereason`), padahal ulasan pelanggan sering berisi **keluhan ganda** (misal: penerbangan delay + bagasi hilang + staf kasar sekaligus dalam 1 tweet). Tanyakan ke lawan bagaimana model mereka menangani ambiguitas anotasi (*annotator disagreement*) pada kasus seperti ini. | 10 sampel tweet keluhan ganda di `results/debate_ammunition.md` bagian 4; tabel perbandingan "tugas kita vs tugas lawan" |
| **Anggota 5 — audit kesalahan & penutup** | Menunjukkan evaluasi confusion matrix, membedah contoh salah prediksi dari inspeksi teks nyata, dan memberikan kesimpulan penutup | Dari 2.886 data uji, terdapat 552 kesalahan (tingkat error 19,13%). Pola error terbukti dari teks: (A) Sarkasme/ironi di mana kata positif (*great*, *proud :)*) dipakai menyindir kegagalan maskapai (33 kasus); (B) Pujian komparatif yang menyindir kompetitor; (C) Pertanyaan netral yang membawa kata sensitif (*fee*, *charge*, *cancel*). Tutup dengan pernyataan keterbatasan model n-gram linier. | `results/plots/confusion_matrix_tfidf.png`; 7 sampel audit salah prediksi terinspeksi di `results/debate_ammunition.md` bagian 5 |
| **Semua anggota — sesi Q&A / sanggahan** | Menjawab pertanyaan sanggahan dari kelompok lawan maupun dosen sesuai bidang keahlian masing-masing | • Anggota 2: pertanyaan seputar tokenisasi, stopwords negasi, dan vectorizer sparse.<br>• Anggota 3: pertanyaan seputar dimensi vektor, window size, OOV, dan teknik pooling dense.<br>• Anggota 4: pertanyaan seputar kesesuaian dataset dan metodologi perbandingan antarkelompok.<br>• Anggota 5: pertanyaan seputar penanganan sarkasme dan batas performa model.<br>• Koordinator: penengah/pengambil alih jika pertanyaan bersifat lintas topik atau situasi debat membutuhkan arahan umum. | `results/debate_ammunition.md` (dokumen lengkap) sebagai amunisi rujukan utama meja debat |

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
