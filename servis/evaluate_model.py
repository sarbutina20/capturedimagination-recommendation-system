# evaluate_model.py

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pymongo import MongoClient
from dotenv import load_dotenv
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import model_loader


K_FOR_RECOMMENDATION = 20
GAMMA = 0.7

def load_raw_interactions(db_client):
    db = db_client[os.getenv("DB_NAME")]
    interactions_cursor = db.interakcije.find(
        {"tip_interakcije": {"$in": ["like", "cart", "order"]}},
        {"korisnik_id": 1, "isbn": 1, "vrijeme": 1, "_id": 0}
    )
    df = pd.DataFrame(list(interactions_cursor))
    df['user_id'] = df['korisnik_id'].astype(str)
    df.rename(columns={'vrijeme': 'timestamp', 'isbn': 'isbn'}, inplace=True)
    return df[['user_id', 'isbn', 'timestamp']]

def perform_temporal_split(df, train_ratio=0.8):
    df = df.sort_values(by='timestamp').groupby('user_id')
    user_train_history = defaultdict(set)
    user_test_ground_truth = defaultdict(set)
    for user_id, user_data in tqdm(df, desc="Obrađujem korisnike"):
        if len(user_data) < 5:
            continue
        split_point = int(len(user_data) * train_ratio)
        train_part = user_data.iloc[:split_point]
        test_part = user_data.iloc[split_point:]
        user_train_history[user_id] = set(train_part['isbn'])
        user_test_ground_truth[user_id] = set(test_part['isbn'])
    return user_train_history, user_test_ground_truth

def get_recommendations_for_user(user_id, user_history_isbns, model_components, all_isbns_set, k):
    user_tower = model_components["user_tower"]
    user_map = model_components["user_map"]
    location_map = model_components["location_map"]
    all_book_embeddings = model_components["all_book_embeddings"]
    unique_isbns = model_components["unique_isbns"]
    isbn_to_idx = {isbn: i for i, isbn in enumerate(unique_isbns)}
    history_indices = [isbn_to_idx[isbn] for isbn in user_history_isbns if isbn in isbn_to_idx]
    if history_indices:
        history_embeddings = tf.gather(all_book_embeddings, tf.constant(history_indices, dtype=tf.int64))
        content_profile = tf.nn.l2_normalize(tf.reduce_mean(history_embeddings, axis=0), axis=-1)
        content_scores = tf.linalg.matvec(all_book_embeddings, content_profile)
    else:
        content_scores = tf.zeros(len(unique_isbns), dtype=tf.float32)
    if user_tower is not None and user_id in user_map:
        uid = tf.constant([user_map[user_id]], dtype=tf.int64)
        loc = tf.constant([location_map.get("Nepoznato", 0)], dtype=tf.int64)
        user_embedding = tf.nn.l2_normalize(user_tower([uid, loc], training=False), axis=-1)
        collaborative_scores = tf.squeeze(tf.matmul(all_book_embeddings, tf.transpose(user_embedding)), axis=-1)
    else:
        collaborative_scores = tf.zeros(len(unique_isbns), dtype=tf.float32)
    final_scores = GAMMA * content_scores + (1.0 - GAMMA) * collaborative_scores
    scores_np = final_scores.numpy()
    history_indices_to_filter = [isbn_to_idx[isbn] for isbn in user_history_isbns if isbn in isbn_to_idx]
    scores_np[history_indices_to_filter] = -np.inf
    top_indices = np.argsort(-scores_np)[:k]
    recommended_isbns = [unique_isbns[i] for i in top_indices]
    return recommended_isbns

def calculate_metrics(recommendations, ground_truth):
    hits = len(set(recommendations) & set(ground_truth))
    precision = hits / len(recommendations) if recommendations else 0
    recall = hits / len(ground_truth) if ground_truth else 0
    return precision, recall

def calculate_ndcg(recommendations, ground_truth):
    k = len(recommendations)
    dcg = 0
    for i, item in enumerate(recommendations):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(ground_truth)))])
    return dcg / idcg if idcg > 0 else 0

def calculate_diversity(recommendations, book_embeddings_map, unique_isbns):
    isbn_to_idx = {isbn: i for i, isbn in enumerate(unique_isbns)}
    rec_indices = [isbn_to_idx[isbn] for isbn in recommendations if isbn in isbn_to_idx]
    if len(rec_indices) < 2:
        return 0.0
    rec_embeddings = tf.gather(book_embeddings_map, rec_indices).numpy()
    similarity_matrix = cosine_similarity(rec_embeddings)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    avg_similarity = np.mean(similarity_matrix[upper_triangle_indices])
    return 1.0 - avg_similarity

def main():
    load_dotenv()
    mongo_client = MongoClient(os.getenv("MONGO_URL"))
    raw_interactions_df = load_raw_interactions(mongo_client)
    model_components = model_loader.load_all()
    user_train, user_test = perform_temporal_split(raw_interactions_df)
    all_isbns_in_catalog = set(model_components["unique_isbns"])
    all_metrics = defaultdict(list)
    all_recommended_items = set()
    for user_id, ground_truth_items in tqdm(user_test.items(), desc="Evaluacija korisnika"):
        user_history = user_train.get(user_id, set())
        recommendations = get_recommendations_for_user(
            user_id, user_history, model_components, all_isbns_in_catalog, K_FOR_RECOMMENDATION
        )
        all_recommended_items.update(recommendations)
        precision, recall = calculate_metrics(recommendations, ground_truth_items)
        ndcg = calculate_ndcg(recommendations, ground_truth_items)
        all_metrics["precision_at_k"].append(precision)
        all_metrics["recall_at_k"].append(recall)
        all_metrics["ndcg_at_k"].append(ndcg)
        diversity = calculate_diversity(recommendations, model_components["all_book_embeddings"], model_components["unique_isbns"])
        all_metrics["diversity"].append(diversity)
    catalog_coverage = len(all_recommended_items) / len(all_isbns_in_catalog)
    print("\n--- REZULTATI EVALUACIJE ---")
    print(f"Metrike izračunate za k = {K_FOR_RECOMMENDATION}")
    print("-" * 30)
    print(f"Prosječna Preciznost (Precision@{K_FOR_RECOMMENDATION}): {np.mean(all_metrics['precision_at_k']):.4f}")
    print(f"Prosječan Opoziv (Recall@{K_FOR_RECOMMENDATION}):    {np.mean(all_metrics['recall_at_k']):.4f}")
    print(f"Prosječan NDCG@{K_FOR_RECOMMENDATION}:                 {np.mean(all_metrics['ndcg_at_k']):.4f}")
    print("-" * 30)
    print(f"Prosječna Raznolikost (Diversity):          {np.mean(all_metrics['diversity']):.4f}")
    print(f"Pokrivenost Kataloga (Catalog Coverage):    {catalog_coverage:.4f}")
    print("-" * 30)
    mongo_client.close()

if __name__ == "__main__":
    main()