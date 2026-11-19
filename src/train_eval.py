import os
import sys
import time

sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import load_and_split_data, minimal_clean, full_norm
from src.embeddings import FeatureExtractor

def run_benchmarks(data_path: str = "data/raw/Tweets.csv", results_csv: str = "results/benchmark_summary.csv"):
    print("=" * 85)
    print("MEMULAI BENCHMARK MODULAR SENTIMENT ANALYSIS (AIRLINE SENTIMENT)")
    print("=" * 85)
    
    # 1. Load data dan split train-test sebelum proses apapun (No Data Leakage)
    split_info = load_and_split_data(data_path=data_path, test_size=0.2, random_state=42)
    X_train_raw = split_info["X_train"]
    X_test_raw = split_info["X_test"]
    y_train = split_info["y_train"]
    y_test = split_info["y_test"]
    
    print(f"Data Split: Train = {len(X_train_raw):,} baris | Test = {len(X_test_raw):,} baris")
    print(f"Distribusi Target Train:\n{y_train.value_counts(normalize=True).to_dict()}")
    
    # Preprocessing Cache
    prep_data = {}
    for prep_name, prep_func in [("minimal_clean", minimal_clean), ("full_norm", full_norm)]:
        print(f"\nMenjalankan preprocessing: '{prep_name}'...")
        t0 = time.time()
        X_tr = X_train_raw.apply(prep_func)
        X_te = X_test_raw.apply(prep_func)
        prep_data[prep_name] = {
            "X_train": X_tr,
            "X_test": X_te,
            "runtime": time.time() - t0
        }
        print(f"Selesai preprocessing '{prep_name}' dalam {prep_data[prep_name]['runtime']:.2f} detik.")
        
    extractor = FeatureExtractor(vector_size=100, window=5, min_count=3, seed=42)
    print("\nMemuat model pre-trained GloVe Twitter 100d...")
    glove_model = extractor.load_glove()
    print("Model GloVe siap digunakan!")

    results = []
    
    # Loop 2 skenario Preprocessing
    for prep_name, data in prep_data.items():
        X_tr = data["X_train"]
        X_te = data["X_test"]
        
        print(f"\n" + "-" * 70)
        print(f"EKSPERIMEN PREPROCESSING: {prep_name.upper()}")
        print("-" * 70)
        
        # 1. Bag-of-Words
        print("  -> Ekstraksi Fitur: Bag-of-Words (ngram 1-2)...")
        t_start = time.time()
        X_tr_bow, X_te_bow, oov_bow = extractor.build_bow(X_tr, X_te)
        t_feat = time.time() - t_start
        
        for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
            for cw in [None, "balanced"]:
                t0 = time.time()
                clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                clf.fit(X_tr_bow, y_train)
                y_pred = clf.predict(X_te_bow)
                run_sec = t_feat + (time.time() - t0)
                
                results.append({
                    "Representation": "BoW",
                    "Pooling": "None",
                    "Preprocessing": prep_name,
                    "Classifier": clf_name,
                    "Class_Weight": str(cw),
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Test_OOV_Pct": round(oov_bow, 2),
                    "Runtime_Sec": round(run_sec, 3)
                })

        # 2. TF-IDF
        print("  -> Ekstraksi Fitur: TF-IDF (ngram 1-2)...")
        t_start = time.time()
        X_tr_tfidf, X_te_tfidf, oov_tfidf = extractor.build_tfidf(X_tr, X_te)
        t_feat = time.time() - t_start
        
        for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
            for cw in [None, "balanced"]:
                t0 = time.time()
                clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                clf.fit(X_tr_tfidf, y_train)
                y_pred = clf.predict(X_te_tfidf)
                run_sec = t_feat + (time.time() - t0)
                
                results.append({
                    "Representation": "TF-IDF",
                    "Pooling": "None",
                    "Preprocessing": prep_name,
                    "Classifier": clf_name,
                    "Class_Weight": str(cw),
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                    "Test_OOV_Pct": round(oov_tfidf, 2),
                    "Runtime_Sec": round(run_sec, 3)
                })

        # Fit TF-IDF Vectorizer untuk weighted pooling dense
        tfidf_vec = TfidfVectorizer(min_df=3)
        tfidf_vec.fit(X_tr)

        # 3. Word2Vec CBOW
        print("  -> Melatih Word2Vec CBOW (sg=0) pada X_train...")
        t_start = time.time()
        w2v_cbow = extractor.train_w2v(X_tr, sg=0)
        t_train_cbow = time.time() - t_start
        oov_cbow = extractor.calculate_oov_rate(w2v_cbow, X_te)
        
        for pool in ["mean", "tfidf"]:
            t0 = time.time()
            X_tr_p = extractor.pool_dense(w2v_cbow, X_tr, pooling=pool, tfidf_vec=tfidf_vec)
            X_te_p = extractor.pool_dense(w2v_cbow, X_te, pooling=pool, tfidf_vec=tfidf_vec)
            t_pool = time.time() - t0
            
            for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
                for cw in [None, "balanced"]:
                    t0 = time.time()
                    clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                    clf.fit(X_tr_p, y_train)
                    y_pred = clf.predict(X_te_p)
                    run_sec = t_train_cbow + t_pool + (time.time() - t0)
                    
                    results.append({
                        "Representation": "Word2Vec_CBOW",
                        "Pooling": pool.upper(),
                        "Preprocessing": prep_name,
                        "Classifier": clf_name,
                        "Class_Weight": str(cw),
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Test_OOV_Pct": round(oov_cbow, 2),
                        "Runtime_Sec": round(run_sec, 3)
                    })

        # 4. Word2Vec Skip-gram
        print("  -> Melatih Word2Vec Skip-gram (sg=1) pada X_train...")
        t_start = time.time()
        w2v_sg = extractor.train_w2v(X_tr, sg=1)
        t_train_sg = time.time() - t_start
        oov_sg = extractor.calculate_oov_rate(w2v_sg, X_te)
        
        for pool in ["mean", "tfidf"]:
            t0 = time.time()
            X_tr_p = extractor.pool_dense(w2v_sg, X_tr, pooling=pool, tfidf_vec=tfidf_vec)
            X_te_p = extractor.pool_dense(w2v_sg, X_te, pooling=pool, tfidf_vec=tfidf_vec)
            t_pool = time.time() - t0
            
            for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
                for cw in [None, "balanced"]:
                    t0 = time.time()
                    clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                    clf.fit(X_tr_p, y_train)
                    y_pred = clf.predict(X_te_p)
                    run_sec = t_train_sg + t_pool + (time.time() - t0)
                    
                    results.append({
                        "Representation": "Word2Vec_SkipGram",
                        "Pooling": pool.upper(),
                        "Preprocessing": prep_name,
                        "Classifier": clf_name,
                        "Class_Weight": str(cw),
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Test_OOV_Pct": round(oov_sg, 2),
                        "Runtime_Sec": round(run_sec, 3)
                    })

        # 5. GloVe Twitter 100d
        print("  -> Ekstraksi Fitur Pre-trained GloVe Twitter 100d...")
        oov_glove = extractor.calculate_oov_rate(glove_model, X_te)
        
        for pool in ["mean", "tfidf"]:
            t0 = time.time()
            X_tr_p = extractor.pool_dense(glove_model, X_tr, pooling=pool, tfidf_vec=tfidf_vec)
            X_te_p = extractor.pool_dense(glove_model, X_te, pooling=pool, tfidf_vec=tfidf_vec)
            t_pool = time.time() - t0
            
            for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
                for cw in [None, "balanced"]:
                    t0 = time.time()
                    clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                    clf.fit(X_tr_p, y_train)
                    y_pred = clf.predict(X_te_p)
                    run_sec = t_pool + (time.time() - t0)
                    
                    results.append({
                        "Representation": "GloVe",
                        "Pooling": pool.upper(),
                        "Preprocessing": prep_name,
                        "Classifier": clf_name,
                        "Class_Weight": str(cw),
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Test_OOV_Pct": round(oov_glove, 2),
                        "Runtime_Sec": round(run_sec, 3)
                    })

        # 6. fastText
        print("  -> Melatih fastText (min_n=3, max_n=6) pada X_train...")
        t_start = time.time()
        ft_model = extractor.train_fasttext(X_tr)
        t_train_ft = time.time() - t_start
        oov_ft = extractor.calculate_oov_rate(ft_model, X_te, is_fasttext=True)
        
        for pool in ["mean", "tfidf"]:
            t0 = time.time()
            X_tr_p = extractor.pool_dense(ft_model, X_tr, pooling=pool, tfidf_vec=tfidf_vec, is_fasttext=True)
            X_te_p = extractor.pool_dense(ft_model, X_te, pooling=pool, tfidf_vec=tfidf_vec, is_fasttext=True)
            t_pool = time.time() - t0
            
            for clf_name, clf_cls in [("LinearSVC", LinearSVC), ("LogisticRegression", LogisticRegression)]:
                for cw in [None, "balanced"]:
                    t0 = time.time()
                    clf = clf_cls(class_weight=cw, random_state=42, max_iter=2000)
                    clf.fit(X_tr_p, y_train)
                    y_pred = clf.predict(X_te_p)
                    run_sec = t_train_ft + t_pool + (time.time() - t0)
                    
                    results.append({
                        "Representation": "fastText",
                        "Pooling": pool.upper(),
                        "Preprocessing": prep_name,
                        "Classifier": clf_name,
                        "Class_Weight": str(cw),
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "Macro_Precision": round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_Recall": round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Macro_F1": round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
                        "Test_OOV_Pct": round(oov_ft, 2),
                        "Runtime_Sec": round(run_sec, 3)
                    })

    # Simpan hasil ke CSV
    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df_res.to_csv(results_csv, index=False)
    print(f"\nSemua metrik ({len(df_res)} konfigurasi) berhasil disimpan ke: {results_csv}")
    
    # Tampilkan Ringkasan Top 15 Berdasarkan Macro F1
    df_res_sorted = df_res.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 100)
    print("TOP 15 KONFIGURASI TERBAIK BERDASARKAN MACRO F1-SCORE")
    print("=" * 100)
    cols_show = ["Representation", "Pooling", "Preprocessing", "Classifier", "Class_Weight", "Accuracy", "Macro_F1", "Test_OOV_Pct", "Runtime_Sec"]
    print(tabulate(df_res_sorted[cols_show].head(15), headers=cols_show, tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_benchmarks()
