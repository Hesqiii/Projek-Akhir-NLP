"""Build reproducible plots and evidence notes for the debate."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath("."))

from src.embeddings import FeatureExtractor
from src.preprocessing import load_and_split_data, minimal_clean


ROOT = Path(".")
DATA_PATH = ROOT / "data/raw/Tweets.csv"
SUMMARY_PATH = ROOT / "results/benchmark_summary.csv"
PLOTS_DIR = ROOT / "results/plots"
REPORT_PATH = ROOT / "results/debate_ammunition.md"
LABELS = ["negative", "neutral", "positive"]

REPRESENTATIONS = [
    "TF-IDF",
    "BoW",
    "Word2Vec_SkipGram",
    "GloVe",
    "Word2Vec_CBOW",
    "fastText",
]
DISPLAY_NAMES = {
    "TF-IDF": "TF-IDF",
    "BoW": "BoW",
    "Word2Vec_SkipGram": "Word2Vec Skip-gram",
    "GloVe": "GloVe Twitter",
    "Word2Vec_CBOW": "Word2Vec CBOW",
    "fastText": "fastText",
}

ISSUE_PATTERNS = {
    "Delay / disruption": [
        r"\bdelays?\b",
        r"\bdelayed\b",
        r"\blate\b",
        r"\bstranded\b",
        r"\bmissed connection\b",
    ],
    "Baggage": [r"\bbags?\b", r"\bluggage\b", r"\bsuitcases?\b", r"\bbaggage\b"],
    "Service / staff": [
        r"\bcustomer service\b",
        r"\brude\b",
        r"\bunhelpful\b",
        r"\b(?:agent|staff|representative)s?\b",
        r"\bon hold\b",
    ],
    "Cancellation": [
        r"\bcancel(?:led|ed|lation|ing)?\b",
        r"\bcancelled flight(?:ed|ation)\b",
    ],
}
SOURCE_ARTIFACTS = re.compile(r"cancelled flight(?:ed|ation)?|late flight(?:r)?", re.IGNORECASE)
AIRLINE_HANDLES = {
    "American": "@americanair",
    "US Airways": "@usairways",
    "United": "@united",
    "Southwest": "@southwestair",
    "Delta": "@delta",
    "JetBlue": "@jetblue",
    "Virgin America": "@virginamerica",
}
KNOWN_HANDLES = re.compile(r"@(AmericanAir|USAirways|United|SouthwestAir|Delta|VirginAmerica|JetBlue)\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_']{2,}")

sns.set_theme(style="whitegrid", palette="muted")


def read_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary benchmark tidak ditemukan: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    required = {
        "Representation",
        "Pooling",
        "Preprocessing",
        "Classifier",
        "Class_Weight",
        "Accuracy",
        "Macro_F1",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kolom benchmark tidak lengkap: {missing}")
    for col in ("Accuracy", "Macro_F1"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if len(df) != 80:
        raise ValueError(f"Diharapkan 80 konfigurasi benchmark, ditemukan {len(df)}.")
    return df


def best_rows_by_representation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in REPRESENTATIONS:
        choices = df[df["Representation"] == name]
        if choices.empty:
            raise ValueError(f"Representasi {name} tidak ada di benchmark summary.")
        rows.append(choices.sort_values("Macro_F1", ascending=False).iloc[0])
    best = pd.DataFrame(rows).reset_index(drop=True)
    best["Display_Name"] = best["Representation"].map(DISPLAY_NAMES)
    return best


def plot_representations(best: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(best))
    width = 0.36
    bars_f1 = ax.bar(x - width / 2, best["Macro_F1"], width, label="Macro F1", color="#276FBF")
    bars_acc = ax.bar(x + width / 2, best["Accuracy"], width, label="Accuracy", color="#E07A32")
    ax.set_title("Hasil terbaik yang tercatat untuk tiap representasi")
    ax.set_ylabel("Skor pada test set")
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x, best["Display_Name"], rotation=18, ha="right")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for bars in (bars_f1, bars_acc):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.text(
        0.5,
        0.015,
        "Setiap nilai berasal dari baris dengan Macro F1 tertinggi untuk representasi itu; konfigurasi pendukungnya dapat berbeda.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOTS_DIR / "representations_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("Preprocessing", as_index=False)
        .agg(
            Mean_Macro_F1=("Macro_F1", "mean"),
            Std_Macro_F1=("Macro_F1", "std"),
            Mean_Accuracy=("Accuracy", "mean"),
            Count=("Macro_F1", "count"),
        )
    )
    expected = {"minimal_clean", "full_norm"}
    if set(agg["Preprocessing"]) != expected:
        raise ValueError(f"Skenario preprocessing tidak sesuai: {agg['Preprocessing'].tolist()}")
    if agg["Count"].nunique() != 1:
        raise ValueError("Jumlah konfigurasi per preprocessing tidak seimbang.")
    return agg.set_index("Preprocessing").loc[["minimal_clean", "full_norm"]].reset_index()


def plot_ablation(agg: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    colors = ["#4C956C", "#C8553D"]
    for ax, metric, title, ylabel in [
        (axes[0], "Mean_Macro_F1", "Rata-rata Macro F1", "Macro F1"),
        (axes[1], "Mean_Accuracy", "Rata-rata akurasi", "Accuracy"),
    ]:
        bars = ax.bar(agg["Preprocessing"], agg[metric], color=colors, width=0.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 0.9)
        ax.tick_params(axis="x", rotation=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.bar_label(bars, labels=[f"{v:.4f}\nN={n}" for v, n in zip(agg[metric], agg["Count"])], padding=3)
    fig.suptitle("Ablasi preprocessing — rerata 40 konfigurasi per skenario", y=1.02)
    fig.text(
        0.5,
        0.01,
        "Ringkasan deskriptif dari konfigurasi benchmark; bukan 40 pengulangan acak yang independen.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(PLOTS_DIR / "preprocessing_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_best_tfidf(X_train_raw, X_test_raw, y_train, y_test, summary: pd.DataFrame):
    row = summary[
        (summary["Representation"] == "TF-IDF")
        & (summary["Preprocessing"] == "minimal_clean")
        & (summary["Classifier"] == "LinearSVC")
        & (summary["Class_Weight"].isna() | (summary["Class_Weight"].astype(str) == "None"))
    ]
    if len(row) != 1:
        raise ValueError("Konfigurasi TF-IDF/minimal_clean/LinearSVC/None tidak unik atau tidak ada.")

    clean_train = X_train_raw.apply(minimal_clean)
    clean_test = X_test_raw.apply(minimal_clean)
    extractor = FeatureExtractor()
    Xtr, Xte, _ = extractor.build_tfidf(clean_train, clean_test)
    model = LinearSVC(class_weight=None, random_state=42, max_iter=2000)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    actual_acc = accuracy_score(y_test, pred)
    actual_f1 = f1_score(y_test, pred, average="macro")
    expected = row.iloc[0]
    if not (np.isclose(actual_acc, expected["Accuracy"], atol=0.00011) and np.isclose(actual_f1, expected["Macro_F1"], atol=0.00011)):
        raise ValueError(
            "Hasil model yang dibangun ulang tidak cocok dengan CSV: "
            f"accuracy={actual_acc:.4f}/{expected['Accuracy']:.4f}, "
            f"Macro F1={actual_f1:.4f}/{expected['Macro_F1']:.4f}. "
            "Periksa dataset, split, dan versi dependency sebelum membuat bukti."
        )
    return model, pred, actual_acc, actual_f1


def plot_confusion_matrix(y_test, pred, accuracy: float, macro_f1: float) -> np.ndarray:
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    row_totals = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    annotations = np.array(
        [[f"{count:,}\n({pct:.1%})" for count, pct in zip(row, pct_row)] for row, pct_row in zip(cm, row_pct)],
        dtype=object,
    )
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=[label.title() for label in LABELS],
        yticklabels=[label.title() for label in LABELS],
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"TF-IDF + minimal_clean + LinearSVC\nAccuracy {accuracy:.2%} | Macro F1 {macro_f1:.4f}")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix_tfidf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return cm


def token_oov_evidence(X_train_raw, X_test_raw, limit: int = 15) -> tuple[list[dict], dict]:
    train = X_train_raw.apply(minimal_clean)
    test = X_test_raw.apply(minimal_clean)
    extractor = FeatureExtractor()
    w2v = extractor.train_w2v(train, sg=1)
    fasttext = extractor.train_fasttext(train)
    glove = extractor.load_glove()
    w2v_vocab = w2v.wv.key_to_index
    glove_vocab = glove.key_to_index
    token_counts = Counter(token for text in test for token in TOKEN_PATTERN.findall(text))

    evidence = []
    for token, count in token_counts.most_common():
        # Waktu, nomor penerbangan, dan fragmen alfanumerik kurang berguna
        # sebagai contoh linguistik OOV untuk presentasi.
        if not token.isalpha():
            continue
        if token in w2v_vocab or token in glove_vocab:
            continue
        try:
            vector = fasttext.wv[token]
        except (KeyError, ValueError):
            continue
        try:
            neighbors = fasttext.wv.most_similar(positive=[vector], topn=8)
            neighbors = [(word, score) for word, score in neighbors if word != token][:3]
        except (KeyError, ValueError):
            neighbors = []
        evidence.append(
            {
                "token": token,
                "test_count": count,
                "w2v": "OOV",
                "glove": "OOV",
                "ft_vector": "available",
                "norm": float(np.linalg.norm(vector)),
                "neighbors": neighbors,
            }
        )
        if len(evidence) == limit:
            break

    metrics = {
        "candidate_count": len(evidence),
        "w2v_test_oov_pct": extractor.calculate_oov_rate(w2v, test),
        "glove_test_oov_pct": extractor.calculate_oov_rate(glove, test),
    }
    return evidence, metrics


def multi_complaint_examples(data_path: Path, limit: int = 10) -> list[dict]:
    df = pd.read_csv(data_path).reset_index(names="source_row")
    required = {"text", "negativereason", "airline", "tweet_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kolom dataset untuk contoh keluhan tidak ada: {missing}")
    df = df.dropna(subset=["text", "negativereason", "airline"])
    examples = []
    per_airline = Counter()
    for _, row in df.iterrows():
        text = str(row["text"])
        # Hindari contoh dengan artefak literal yang mengganggu pembacaan slide.
        if SOURCE_ARTIFACTS.search(text):
            continue
        first_handle = KNOWN_HANDLES.search(text)
        if first_handle:
            mentioned_handle = f"@{first_handle.group(1)}".lower()
            expected_handle = AIRLINE_HANDLES.get(str(row["airline"]), "").lower()
            if expected_handle and mentioned_handle != expected_handle:
                continue
        indicators = []
        for issue, patterns in ISSUE_PATTERNS.items():
            matches = [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
            if matches:
                indicators.append({"issue": issue, "matched": matches})
        if len(indicators) < 2:
            continue
        airline = str(row["airline"])
        if per_airline[airline] >= 2:
            continue
        examples.append(
            {
                "source_row": int(row["source_row"]),
                "tweet_id": str(row["tweet_id"]),
                "airline_column": airline,
                "text": text,
                "negative_reason": str(row["negativereason"]),
                "indicators": indicators,
            }
        )
        per_airline[airline] += 1
        if len(examples) >= limit:
            break
    return examples


def error_examples(X_test_raw, y_test, pred, row_lookup: dict[str, tuple[int, str]], limit_each: int = 5) -> dict:
    errors = pd.DataFrame(
        {
            "source_row": y_test.index,
            "text": X_test_raw.to_numpy(),
            "true": y_test.to_numpy(),
            "predicted": pred,
        }
    )
    errors = errors[errors["true"] != errors["predicted"]].copy()
    source_rows = errors["text"].map(lambda text: row_lookup.get(str(text), (-1, "unknown")))
    errors["source_row"] = source_rows.map(lambda item: item[0])
    errors["tweet_id"] = source_rows.map(lambda item: item[1])
    groups = {
        "negative → positive (positive-word candidates)": errors[
            (errors["true"] == "negative")
            & (errors["predicted"] == "positive")
            & errors["text"].str.contains(r"\b(?:thanks|thank you|great|love|happy|proud|excellent)\b", case=False, regex=True)
        ],
        "positive → negative": errors[(errors["true"] == "positive") & (errors["predicted"] == "negative")],
        "neutral → negative": errors[(errors["true"] == "neutral") & (errors["predicted"] == "negative")],
    }
    examples = {}
    for name, group in groups.items():
        examples[name] = group.head(limit_each).to_dict("records")
    counts = Counter(f"{r} → {p}" for r, p in zip(errors["true"], errors["predicted"]))
    return {
        "test_count": len(y_test),
        "error_count": len(errors),
        "error_pct": len(errors) / len(y_test) * 100,
        "confusion_counts": counts,
        "examples": examples,
    }


def format_neighbors(items: list[tuple[str, float]]) -> str:
    if not items:
        return "(tidak tersedia)"
    return ", ".join(f"`{word}` ({score:.3f})" for word, score in items)


def write_report(
    best: pd.DataFrame,
    ablation: pd.DataFrame,
    oov_rows: list[dict],
    oov_metrics: dict,
    complaint_rows: list[dict],
    errors: dict,
    accuracy: float,
    macro_f1: float,
    cm: np.ndarray,
) -> None:
    raw = pd.read_csv(DATA_PATH)
    file_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    artifact_counts = {
        term: int(raw["text"].fillna("").str.count(re.escape(term)).sum())
        for term in ("Cancelled Flighted", "Cancelled Flightation", "Late Flightr")
    }
    lines = [
        "# Bukti Debat NLP — Hasil yang Dapat Ditelusuri",
        "",
        "Angka model dan contoh tweet pada dokumen ini dibuat dari berkas lokal, bukan diketik sebagai contoh. Pemeriksaan manual tetap diperlukan untuk menyebut sarkasme atau menentukan arti keluhan.",
        "",
        "## Sumber dan reproduksi",
        "",
        f"- Dataset: `{DATA_PATH.as_posix()}` ({len(raw):,} baris; SHA-256 `{file_hash}`).",
        "- Split dibangun ulang melalui `load_and_split_data` dengan `test_size=0.2`, `stratify=y`, dan `random_state=42`.",
        "- Jalankan dari root proyek: `uv run python src/debate_prep.py`.",
        "- `minimal_clean`, vectorizer, dan classifier mengikuti implementasi pada `src/preprocessing.py`, `src/embeddings.py`, dan `src/train_eval.py`.",
        "",
        "## 1. Hasil terbaik per representasi",
        "",
        "Tiap baris berikut adalah konfigurasi dengan Macro F1 tertinggi yang tercatat untuk representasi tersebut. Kolom Accuracy berasal dari baris konfigurasi yang sama; pilihan preprocessing/classifier dapat berbeda antarrepresentasi.",
        "",
        "| Representasi | Preprocessing | Pooling | Classifier | Class weight | Accuracy | Macro F1 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for _, row in best.iterrows():
        pooling = "—" if pd.isna(row["Pooling"]) or str(row["Pooling"]) == "None" else str(row["Pooling"])
        cw = "None" if pd.isna(row["Class_Weight"]) or str(row["Class_Weight"]) == "None" else str(row["Class_Weight"])
        lines.append(
            f"| {DISPLAY_NAMES[row['Representation']]} | {row['Preprocessing']} | {pooling} | {row['Classifier']} | {cw} | {row['Accuracy']:.4f} | {row['Macro_F1']:.4f} |"
        )
    lines += ["", "Plot: `results/plots/representations_comparison.png`.", "", "## 2. Ablasi preprocessing", ""]
    lines += [
        "Rerata deskriptif dihitung atas semua baris konfigurasi yang tersedia untuk tiap preprocessing. Ini bukan 40 pengulangan acak independen dan bukan uji signifikansi.",
        "",
        "| Preprocessing | Jumlah konfigurasi | Mean Macro F1 | SD Macro F1 | Mean Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in ablation.iterrows():
        lines.append(
            f"| {row['Preprocessing']} | {int(row['Count'])} | {row['Mean_Macro_F1']:.4f} | {row['Std_Macro_F1']:.4f} | {row['Mean_Accuracy']:.4f} |"
        )
    delta = float(ablation.loc[ablation["Preprocessing"] == "minimal_clean", "Mean_Macro_F1"].iloc[0] - ablation.loc[ablation["Preprocessing"] == "full_norm", "Mean_Macro_F1"].iloc[0])
    lines += [
        "",
        f"Selisih rerata Macro F1 `minimal_clean - full_norm`: **{delta:+.4f}**.",
        "Plot: `results/plots/preprocessing_ablation.png`.",
        "",
        "## 3. FastText pada token yang tidak ada di kosakata W2V dan GloVe",
        "",
        f"Dari token yang benar-benar muncul di X_test, skrip menemukan {len(oov_rows)} token unik (maksimal 15) yang tidak ada di indeks Word2Vec Skip-gram maupun GloVe dan tetap menghasilkan vektor melalui fastText. OOV berdasarkan kosakata pada test: Word2Vec {oov_metrics['w2v_test_oov_pct']:.2f}%; GloVe {oov_metrics['glove_test_oov_pct']:.2f}%.",
        "",
        "| Token X_test | Frekuensi di X_test | Word2Vec | GloVe | Vektor fastText | Norm | Tetangga terdekat (similarity) |",
        "|---|---:|---|---|---|---:|---|",
    ]
    if oov_rows:
        for row in oov_rows:
            lines.append(
                f"| `{row['token']}` | {row['test_count']} | OOV | OOV | {row['ft_vector']} | {row['norm']:.3f} | {format_neighbors(row['neighbors'])} |"
            )
    else:
        lines.append("| — | — | — | — | Tidak ada token yang memenuhi syarat | — | — |")
    lines += [
        "",
        "Interpretasi: ini menunjukkan bahwa fastText menghasilkan vektor untuk token sampel yang tidak berada dalam indeks kata W2V/GloVe. Tetangga cosine hanya keluaran model; kedekatan itu tidak membuktikan makna benar. Angka OOV fastText 0% pada CSV mengikuti konvensi implementasi proyek yang menganggap token dapat ditangani lewat subword, bukan ukuran akurasi semantik.",
        "",
        "## 4. Tweet yang memuat beberapa indikator keluhan",
        "",
        f"Skrip memilih {len(complaint_rows)} tweet berlabel `negativereason` yang memuat kata/frasa dari sedikitnya dua kategori indikator. Kategori adalah hasil pencocokan kata sederhana untuk membantu pemeriksaan; bukan anotasi manual atau bukti bahwa model lawan salah.",
        "",
    ]
    if complaint_rows:
        for i, row in enumerate(complaint_rows, 1):
            indicators = ", ".join(
                f"{item['issue']} (cocok: `{', '.join(item['matched'])}`)" for item in row["indicators"]
            )
            lines += [
                f"### Sampel {i} — source row {row['source_row']} / tweet_id {row['tweet_id']}",
                f"- Kolom airline: `{row['airline_column']}`; `negativereason` asli: `{row['negative_reason']}`.",
                f"- Indikator kata: {indicators}.",
                f"- Teks asli: {row['text']}",
                "",
            ]
    else:
        lines.append("Belum ditemukan contoh yang memenuhi aturan pencocokan kata.")
    lines += [
        "Catatan debat: contoh ini hanya mendukung pernyataan bahwa sebagian tweet menyebut lebih dari satu aspek keluhan sementara dataset menyimpan satu nilai `negativereason`. Hal itu tidak membuktikan ketidaksepakatan anotator, tidak menunjukkan performa model lawan, dan tidak membuat kedua tugas dapat dibandingkan langsung.",
        "",
        "## 5. Confusion matrix dan audit kesalahan",
        "",
        f"Model TF-IDF + `minimal_clean` + LinearSVC (class_weight=None) dibangun ulang pada split tetap. Akurasi **{accuracy:.4f}**, Macro F1 **{macro_f1:.4f}**. Jumlah salah prediksi **{errors['error_count']}/{errors['test_count']} ({errors['error_pct']:.2f}%)**.",
        "",
        "| Label sebenarnya \\ Prediksi | Negative | Neutral | Positive |",
        "|---|---:|---:|---:|",
    ]
    for label, row in zip(LABELS, cm):
        lines.append(f"| {label.title()} | {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", "Plot: `results/plots/confusion_matrix_tfidf.png`.", ""]
    for group_name, rows in errors["examples"].items():
        lines += [f"### Contoh salah prediksi: {group_name}", ""]
        if not rows:
            lines += ["Tidak ditemukan contoh pada kelompok ini.", ""]
            continue
        if group_name.startswith("negative → positive"):
            lines.append("Kandidat pemeriksaan manual berdasarkan adanya kata positif; belum dikonfirmasi sebagai sarkasme.")
            lines.append("")
        for row in rows:
            text = str(row["text"]).replace("\n", " ")
            lines.append(
                f"- source row {row['source_row']} / tweet_id {row['tweet_id']}: label `{row['true']}`, prediksi `{row['predicted']}` — {text}"
            )
        lines.append("")
    lines += [
        "Angka kelompok salah prediksi dihitung dari prediksi model yang dibangun ulang. Contoh dipilih dari kelompok tersebut; interpretasi seperti sarkasme harus diperiksa manusia sebelum dipresentasikan.",
        "",
        "## Catatan kualitas sumber",
        "",
        "Berkas lokal juga mengandung beberapa frasa literal yang tidak lazim. Skrip tidak mengubah teks sumber dan mengecualikan baris berfrasa tersebut dari contoh keluhan agar kutipan mudah dibaca. Metrik tetap dihitung pada dataset persis seperti yang digunakan benchmark.",
        "",
        "| Frasa literal dalam kolom text | Jumlah kemunculan |",
        "|---|---:|",
    ]
    for term, count in artifact_counts.items():
        lines.append(f"| `{term}` | {count} |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_summary()
    best = best_rows_by_representation(summary)
    ablation = summarize_ablation(summary)

    split = load_and_split_data(str(DATA_PATH), test_size=0.2, random_state=42)
    model, pred, accuracy, macro_f1 = train_best_tfidf(
        split["X_train"], split["X_test"], split["y_train"], split["y_test"], summary
    )
    cm = plot_confusion_matrix(split["y_test"], pred, accuracy, macro_f1)

    plot_representations(best)
    plot_ablation(ablation, summary)
    oov_rows, oov_metrics = token_oov_evidence(split["X_train"], split["X_test"])
    complaints = multi_complaint_examples(DATA_PATH)
    raw = pd.read_csv(DATA_PATH).reset_index(names="source_row")
    row_lookup = {
        str(row["text"]): (int(row["source_row"]), str(row["tweet_id"]))
        for _, row in raw.dropna(subset=["text", "tweet_id"]).drop_duplicates(subset=["text"]).iterrows()
    }
    errors = error_examples(split["X_test"], split["y_test"], pred, row_lookup)
    write_report(best, ablation, oov_rows, oov_metrics, complaints, errors, accuracy, macro_f1, cm)

    print(f"Confusion matrix model rebuilt: accuracy={accuracy:.4f}, Macro F1={macro_f1:.4f}")
    print(f"Evidence: {REPORT_PATH}; plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
