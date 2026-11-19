# Bukti Debat NLP — Hasil yang Dapat Ditelusuri

Angka model dan contoh tweet pada dokumen ini dibuat dari berkas lokal, bukan diketik sebagai contoh. Pemeriksaan manual tetap diperlukan untuk menyebut sarkasme atau menentukan arti keluhan.

## Sumber dan reproduksi

- Dataset: `data/raw/Tweets.csv` (14,640 baris; SHA-256 `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`).
- Split dibangun ulang melalui `load_and_split_data` dengan `test_size=0.2`, `stratify=y`, dan `random_state=42`.
- Jalankan dari root proyek: `uv run python src/debate_prep.py`.
- `minimal_clean`, vectorizer, dan classifier mengikuti implementasi pada `src/preprocessing.py`, `src/embeddings.py`, dan `src/train_eval.py`.

## 1. Hasil terbaik per representasi

Tiap baris berikut adalah konfigurasi dengan Macro F1 tertinggi yang tercatat untuk representasi tersebut. Kolom Accuracy berasal dari baris konfigurasi yang sama; pilihan preprocessing/classifier dapat berbeda antarrepresentasi.

| Representasi | Preprocessing | Pooling | Classifier | Class weight | Accuracy | Macro F1 |
|---|---|---|---|---|---:|---:|
| TF-IDF | minimal_clean | — | LinearSVC | None | 0.8087 | 0.7495 |
| BoW | minimal_clean | — | LogisticRegression | None | 0.7983 | 0.7421 |
| Word2Vec Skip-gram | minimal_clean | MEAN | LinearSVC | balanced | 0.7561 | 0.6888 |
| GloVe Twitter | full_norm | MEAN | LinearSVC | balanced | 0.7568 | 0.6836 |
| Word2Vec CBOW | minimal_clean | MEAN | LinearSVC | balanced | 0.7335 | 0.6588 |
| fastText | minimal_clean | MEAN | LinearSVC | balanced | 0.7155 | 0.6312 |

Plot: `results/plots/representations_comparison.png`.

## 2. Ablasi preprocessing

Rerata deskriptif dihitung atas semua baris konfigurasi yang tersedia untuk tiap preprocessing. Ini bukan 40 pengulangan acak independen dan bukan uji signifikansi.

| Preprocessing | Jumlah konfigurasi | Mean Macro F1 | SD Macro F1 | Mean Accuracy |
|---|---:|---:|---:|---:|
| minimal_clean | 40 | 0.6502 | 0.0574 | 0.7342 |
| full_norm | 40 | 0.6195 | 0.0808 | 0.7204 |

Selisih rerata Macro F1 `minimal_clean - full_norm`: **+0.0307**.
Plot: `results/plots/preprocessing_ablation.png`.

## 3. FastText pada token yang tidak ada di kosakata W2V dan GloVe

Dari token yang benar-benar muncul di X_test, skrip menemukan 15 token unik (maksimal 15) yang tidak ada di indeks Word2Vec Skip-gram maupun GloVe dan tetap menghasilkan vektor melalui fastText. OOV berdasarkan kosakata pada test: Word2Vec 5.67%; GloVe 12.82%.

| Token X_test | Frekuensi di X_test | Word2Vec | GloVe | Vektor fastText | Norm | Tetangga terdekat (similarity) |
|---|---:|---|---|---|---:|---|
| `nonrefundable` | 2 | OOV | OOV | available | 1.960 | `unavailable` (0.999), `available` (0.999), `refundable` (0.998) |
| `carsl` | 2 | OOV | OOV | available | 1.691 | `car` (0.998), `cargo` (0.998), `cash` (0.996) |
| `preboard` | 2 | OOV | OOV | available | 2.538 | `richard` (0.999), `clearly` (0.999), `fiasco` (0.999) |
| `overflight` | 2 | OOV | OOV | available | 4.125 | `flighting` (0.991), `flightling` (0.986), `flightlations` (0.983) |
| `servicefail` | 2 | OOV | OOV | available | 3.575 | `services` (1.000), `poorservice` (0.999), `badservice` (0.999) |
| `carryons` | 2 | OOV | OOV | available | 2.247 | `carousel` (0.998), `huntsville` (0.997), `avail` (0.997) |
| `worstflight` | 2 | OOV | OOV | available | 3.566 | `lights` (0.994), `cheapflights` (0.984), `flightations` (0.982) |
| `jeokoo` | 2 | OOV | OOV | available | 0.296 | `cessna's` (0.997), `views` (0.997), `roc` (0.997) |
| `getmartyhome` | 1 | OOV | OOV | available | 1.032 | `toilet` (0.986), `tough` (0.985), `find` (0.984) |
| `reativation` | 1 | OOV | OOV | available | 3.236 | `reputation` (0.999), `registration` (0.999), `location` (0.998) |
| `bookofnegroes` | 1 | OOV | OOV | available | 1.326 | `connected` (0.996), `boy` (0.995), `disconnected` (0.992) |
| `goodgriefpeople` | 1 | OOV | OOV | available | 1.463 | `paperwork` (0.998), `life` (0.997), `lisa` (0.997) |
| `luvintheair` | 1 | OOV | OOV | available | 1.253 | `passport` (0.997), `north` (0.997), `northeast` (0.997) |
| `socialtantrum` | 1 | OOV | OOV | available | 1.107 | `quickly` (0.998), `lovely` (0.998), `apply` (0.998) |
| `jetblues` | 1 | OOV | OOV | available | 3.115 | `cessna's` (0.999), `blog` (0.999), `public` (0.998) |

Interpretasi: ini menunjukkan bahwa fastText menghasilkan vektor untuk token sampel yang tidak berada dalam indeks kata W2V/GloVe. Tetangga cosine hanya keluaran model; kedekatan itu tidak membuktikan makna benar. Angka OOV fastText 0% pada CSV mengikuti konvensi implementasi proyek yang menganggap token dapat ditangani lewat subword, bukan ukuran akurasi semantik.

## 4. Tweet yang memuat beberapa indikator keluhan

Skrip memilih 10 tweet berlabel `negativereason` yang memuat kata/frasa dari sedikitnya dua kategori indikator. Kategori adalah hasil pencocokan kata sederhana untuk membantu pemeriksaan; bukan anotasi manual atau bukti bahwa model lawan salah.

### Sampel 1 — source row 122 / tweet_id 569737603617943552
- Kolom airline: `Virgin America`; `negativereason` asli: `Customer Service Issue`.
- Indikator kata: Delay / disruption (cocok: `\bdelays?\b`), Service / staff (cocok: `\bcustomer service\b`).
- Teks asli: @VirginAmerica I like the customer service but a 40 min delay just for connecting passengers seems too long. VA370

### Sampel 2 — source row 251 / tweet_id 569022319437357056
- Kolom airline: `Virgin America`; `negativereason` asli: `Damaged Luggage`.
- Indikator kata: Baggage (cocok: `\bluggage\b, \bbaggage\b`), Service / staff (cocok: `\b(?:agent|staff|representative)s?\b`).
- Teks asli: @VirginAmerica luggage was severely dented/missing wheel coming off baggage claim in SAN. Luggage agent Miranda (I think) wasn't any help.

### Sampel 3 — source row 710 / tweet_id 570160120698679296
- Kolom airline: `United`; `negativereason` asli: `Customer Service Issue`.
- Indikator kata: Baggage (cocok: `\bluggage\b`), Service / staff (cocok: `\bcustomer service\b`).
- Teks asli: @united customer service is atrocious! You have disrupted my travel plans, you have lost my luggage and it is impossible to TALK TO A HUMAN

### Sampel 4 — source row 741 / tweet_id 570106573395087362
- Kolom airline: `United`; `negativereason` asli: `Customer Service Issue`.
- Indikator kata: Delay / disruption (cocok: `\bdelays?\b`), Service / staff (cocok: `\b(?:agent|staff|representative)s?\b`).
- Teks asli: @united The agent that met us at the gate said any issues w/delays on UA1116 would have to be taken up online...

### Sampel 5 — source row 4374 / tweet_id 570282347267895296
- Kolom airline: `Southwest`; `negativereason` asli: `Damaged Luggage`.
- Indikator kata: Baggage (cocok: `\bluggage\b`), Service / staff (cocok: `\brude\b`).
- Teks asli: @SouthwestAir I'm so upset that my luggage was damaged and the rude employee at the Laguardia airport didn't help!

### Sampel 6 — source row 4501 / tweet_id 570198775203016704
- Kolom airline: `Southwest`; `negativereason` asli: `Customer Service Issue`.
- Indikator kata: Baggage (cocok: `\bbags?\b`), Service / staff (cocok: `\bcustomer service\b`).
- Teks asli: @SouthwestAir customer service at FLL, BWI,and PIT have been terrible.no one knows where his bags are.his is on a job with no clothes &amp; gear

### Sampel 7 — source row 9030 / tweet_id 570276878788861953
- Kolom airline: `US Airways`; `negativereason` asli: `Lost Luggage`.
- Indikator kata: Baggage (cocok: `\bbags?\b`), Service / staff (cocok: `\bon hold\b`).
- Teks asli: @USAirways Need to talk to a person about my bag. Waiting on hold for two hours. Bag s!b delivered last night. Online says not found.

### Sampel 8 — source row 9084 / tweet_id 570239546673238017
- Kolom airline: `US Airways`; `negativereason` asli: `Customer Service Issue`.
- Indikator kata: Delay / disruption (cocok: `\bdelayed\b`), Service / staff (cocok: `\bon hold\b`).
- Teks asli: @USAirways Been on hold for 40 minutes, calling because my flight is delayed.  This type of service in not acceptable.

### Sampel 9 — source row 11883 / tweet_id 570307434113310720
- Kolom airline: `American`; `negativereason` asli: `Late Flight`.
- Indikator kata: Delay / disruption (cocok: `\bstranded\b`), Service / staff (cocok: `\b(?:agent|staff|representative)s?\b`).
- Teks asli: @AmericanAir stranded for 24 hours in MIA, Patrick casimir has been the ONLY AA staff to apologize for the great inconvenience #unreal

### Sampel 10 — source row 11994 / tweet_id 570275010759102466
- Kolom airline: `American`; `negativereason` asli: `Lost Luggage`.
- Indikator kata: Delay / disruption (cocok: `\bdelayed\b`), Baggage (cocok: `\bbags?\b`).
- Teks asli: @AmericanAir this delayed bag was for my friend Lisa Pafe. She got her bag after 3 days in Costa Rica. Issue no updates on your system.

Catatan debat: contoh ini hanya mendukung pernyataan bahwa sebagian tweet menyebut lebih dari satu aspek keluhan sementara dataset menyimpan satu nilai `negativereason`. Hal itu tidak membuktikan ketidaksepakatan anotator, tidak menunjukkan performa model lawan, dan tidak membuat kedua tugas dapat dibandingkan langsung.

## 5. Confusion matrix dan audit kesalahan

Model TF-IDF + `minimal_clean` + LinearSVC (class_weight=None) dibangun ulang pada split tetap. Akurasi **0.8087**, Macro F1 **0.7495**. Jumlah salah prediksi **552/2886 (19.13%)**.

| Label sebenarnya \ Prediksi | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1657 | 126 | 33 |
| Neutral | 207 | 362 | 43 |
| Positive | 88 | 55 | 315 |

Plot: `results/plots/confusion_matrix_tfidf.png`.

### Contoh salah prediksi: negative → positive (positive-word candidates)

Kandidat pemeriksaan manual berdasarkan adanya kata positif; belum dikonfirmasi sebagai sarkasme.

- source row 4977 / tweet_id 569588348593025024: label `negative`, prediksi `positive` — @SouthwestAir thanks for the response - flight time has passed - rebooked now
- source row 9143 / tweet_id 570125976195219456: label `negative`, prediksi `positive` — @USAirways thank you for reaching out to me. Hopefully my request will be fixed and I can remain a loyal customer.
- source row 2437 / tweet_id 569208751972065280: label `negative`, prediksi `positive` — @united What a really GREAT &amp; FLATTERING story about you! You should be very proud :) http://t.co/oKtUkjY92O (via @ParachuteGuy)
- source row 12492 / tweet_id 570153610870456321: label `negative`, prediksi `positive` — @AmericanAir Thank you for the acknowledgement. The IFE didn't work all that well anyway so maybe time to upgrade to lower profile system.
- source row 11881 / tweet_id 570307949614256128: label `negative`, prediksi `positive` — @AmericanAir thanks for getting back to me. But I will find other airlines in the future.

### Contoh salah prediksi: positive → negative

- source row 4203 / tweet_id 567780426601996288: label `positive`, prediksi `negative` — Is expiring the @united offer to buy or transfer #United #MileagePlus miles with discount? Many offers during a year http://t.co/6tz6imqZlG
- source row 13244 / tweet_id 569907544124219392: label `positive`, prediksi `negative` — @AmericanAir thanks! a response is better than nothing at all {ahem, @USAirways}
- source row 11735 / tweet_id 567788578026311680: label `positive`, prediksi `negative` — @USAirways that's why u guys are my #1 choice.
- source row 8263 / tweet_id 568547839636332544: label `positive`, prediksi `negative` — @JetBlue offers hot tea and coffee...at the gate! You guys make me wonder why I have ever flown any other airlines! ☕✈👍
- source row 11360 / tweet_id 568171538324840449: label `positive`, prediksi `negative` — @USAirways we haven't departed yet so let's not get too high hopes.  But everything has been on schedule so far

### Contoh salah prediksi: neutral → negative

- source row 10025 / tweet_id 569566323912671232: label `neutral`, prediksi `negative` — @USAirways Look into better service.
- source row 11184 / tweet_id 568450712314753024: label `neutral`, prediksi `negative` — @USAirways if you've got room on an earlier flight home, why charge me $75 to get on it? Be the hero!
- source row 11023 / tweet_id 568606704566439936: label `neutral`, prediksi `negative` — @USAirways but it's not their fault, because their policy says they at least got us to our destination. Really?
- source row 11943 / tweet_id 570290034009636865: label `neutral`, prediksi `negative` — @AmericanAir OK I will call them likely tmrw UK time to question the process getting the change charge reversed due to a bereavement
- source row 2682 / tweet_id 568957349534289920: label `neutral`, prediksi `negative` — @united How can I verify if wheelchair assistance has been requested for my next flight?

Angka kelompok salah prediksi dihitung dari prediksi model yang dibangun ulang. Contoh dipilih dari kelompok tersebut; interpretasi seperti sarkasme harus diperiksa manusia sebelum dipresentasikan.

## Catatan kualitas sumber

Berkas lokal juga mengandung beberapa frasa literal yang tidak lazim. Skrip tidak mengubah teks sumber dan mengecualikan baris berfrasa tersebut dari contoh keluhan agar kutipan mudah dibaca. Metrik tetap dihitung pada dataset persis seperti yang digunakan benchmark.

| Frasa literal dalam kolom text | Jumlah kemunculan |
|---|---:|
| `Cancelled Flighted` | 228 |
| `Cancelled Flightation` | 25 |
| `Late Flightr` | 161 |
