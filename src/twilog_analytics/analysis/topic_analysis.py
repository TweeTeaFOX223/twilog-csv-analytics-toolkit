from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import polars as pl

from .text_analysis import TextAnalyzer

__all__ = ["monthly_tfidf_top", "kmeans_cluster_summary", "topic_word_trend"]


def monthly_tfidf_top(
    frame: pl.DataFrame, analyzer: TextAnalyzer, top_n: int = 5
) -> pl.DataFrame:
    """月ごとの代表語（TF-IDF上位）を返す。"""

    if "text" not in frame.columns:
        return pl.DataFrame()
    if "year" in frame.columns and "month" in frame.columns:
        df = frame.select(
            pl.col("text").cast(pl.Utf8).fill_null("").alias("text"),
            pl.col("year").cast(pl.Int32).alias("year"),
            pl.col("month").cast(pl.Int8).alias("month"),
        )
    elif "created_at" in frame.columns:
        df = frame.select(
            pl.col("text").cast(pl.Utf8).fill_null("").alias("text"),
            pl.col("created_at").dt.year().alias("year"),
            pl.col("created_at").dt.month().alias("month"),
        )
    else:
        return pl.DataFrame()

    if df.is_empty():
        return pl.DataFrame()

    doc_terms: Dict[Tuple[int, int], Dict[str, int]] = {}
    df_counts: Dict[str, int] = {}
    for row in df.to_dicts():
        key = (int(row["year"]), int(row["month"]))
        terms = analyzer.extract_words_from_text(row["text"])
        if not terms:
            continue
        bucket = doc_terms.setdefault(key, {})
        for term in terms:
            bucket[term] = bucket.get(term, 0) + 1
        for term in set(terms):
            df_counts[term] = df_counts.get(term, 0) + 1

    if not doc_terms:
        return pl.DataFrame()

    doc_count = max(len(doc_terms), 1)
    rows: List[Dict[str, object]] = []
    for (year, month), tf_counts in doc_terms.items():
        scored = []
        for term, tf in tf_counts.items():
            idf = math.log((1 + doc_count) / (1 + df_counts.get(term, 0))) + 1
            scored.append((term, tf * idf))
        scored.sort(key=lambda x: x[1], reverse=True)
        for term, score in scored[:top_n]:
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "year_month": f"{year}-{month:02d}",
                    "word": term,
                    "score": round(score, 4),
                }
            )
    rows.sort(key=lambda r: (r["year"], r["month"], -float(r["score"])))
    return pl.DataFrame(rows)


def kmeans_cluster_summary(
    frame: pl.DataFrame,
    analyzer: TextAnalyzer,
    k: int = 4,
    max_features: int = 200,
    max_iter: int = 15,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """TF-IDFベクトルを簡易KMeansでクラスタ化し、要約を返す。"""

    if "text" not in frame.columns:
        return pl.DataFrame(), pl.DataFrame()

    texts = frame.select(pl.col("text").cast(pl.Utf8).fill_null("")).to_series().to_list()
    if not texts:
        return pl.DataFrame(), pl.DataFrame()

    doc_terms: List[List[str]] = []
    df_counts: Dict[str, int] = {}
    for text in texts:
        terms = analyzer.extract_words_from_text(text)
        doc_terms.append(terms)
        for term in set(terms):
            df_counts[term] = df_counts.get(term, 0) + 1

    if not df_counts:
        return pl.DataFrame(), pl.DataFrame()

    vocab = sorted(df_counts.items(), key=lambda x: x[1], reverse=True)[:max_features]
    vocab_terms = [term for term, _ in vocab]
    vocab_index = {term: idx for idx, term in enumerate(vocab_terms)}
    doc_count = len(doc_terms)

    vectors: List[List[float]] = []
    for terms in doc_terms:
        tf: Dict[str, int] = {}
        for term in terms:
            if term in vocab_index:
                tf[term] = tf.get(term, 0) + 1
        vec = [0.0] * len(vocab_terms)
        for term, count in tf.items():
            idx = vocab_index[term]
            idf = math.log((1 + doc_count) / (1 + df_counts.get(term, 0))) + 1
            vec[idx] = count * idf
        vectors.append(vec)

    if not vectors:
        return pl.DataFrame(), pl.DataFrame()

    k = max(2, min(k, len(vectors)))
    centroids = random.sample(vectors, k)

    def _dist_sq(a: List[float], b: List[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    assignments = [0] * len(vectors)
    for _ in range(max_iter):
        changed = False
        for i, vec in enumerate(vectors):
            distances = [_dist_sq(vec, c) for c in centroids]
            new_idx = distances.index(min(distances))
            if assignments[i] != new_idx:
                assignments[i] = new_idx
                changed = True
        if not changed:
            break
        for idx in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == idx]
            if not members:
                continue
            centroid = [0.0] * len(vocab_terms)
            for vec in members:
                for j, val in enumerate(vec):
                    centroid[j] += val
            centroids[idx] = [val / len(members) for val in centroid]

    cluster_sizes = [0] * k
    cluster_sums = [[0.0] * len(vocab_terms) for _ in range(k)]
    for vec, cluster_id in zip(vectors, assignments):
        cluster_sizes[cluster_id] += 1
        for idx, val in enumerate(vec):
            cluster_sums[cluster_id][idx] += val

    summary_rows: List[Dict[str, object]] = []
    for cluster_id in range(k):
        sums = cluster_sums[cluster_id]
        top_indices = sorted(range(len(sums)), key=lambda i: sums[i], reverse=True)[:8]
        top_terms = [vocab_terms[i] for i in top_indices if sums[i] > 0]
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_sizes[cluster_id],
                "top_terms": ", ".join(top_terms),
            }
        )

    summary_rows.sort(key=lambda r: int(r["cluster_id"]))
    summary_df = pl.DataFrame(summary_rows)
    counts_df = summary_df.select(["cluster_id", "size"])
    return summary_df, counts_df


def topic_word_trend(
    frame: pl.DataFrame, analyzer: TextAnalyzer, top_n: int = 5
) -> pl.DataFrame:
    """代表語の月次推移（投稿数）を返す。"""

    if "text" not in frame.columns:
        return pl.DataFrame()
    tfidf_top = analyzer.get_tfidf_ranking(frame, text_column="text", top_n=top_n)
    if tfidf_top.is_empty():
        return pl.DataFrame()
    from . import text_analysis  # 循環参照回避のため遅延import

    frames: List[pl.DataFrame] = []
    for term in tfidf_top["word"].to_list():
        term_counts = text_analysis.word_monthly_counts(frame, str(term), analyzer)
        if not term_counts.is_empty():
            frames.append(
                term_counts.select(
                    pl.col("year_month"),
                    pl.col("posts"),
                ).with_columns(pl.lit(str(term)).alias("word"))
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical").sort(["year_month", "word"])
