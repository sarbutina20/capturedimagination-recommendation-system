import os, json, numpy as np, tensorflow as tf
from dotenv import load_dotenv
from pymongo import MongoClient

def load_all():
    print("Pokretanje učitavanja poboljšanih modela i podataka...")
    load_dotenv()

    user_tower = tf.keras.models.load_model("data/improved_user_tower.keras") if os.path.exists("data/improved_user_tower.keras") else None
    book_tower = tf.keras.models.load_model("data/improved_book_tower.keras") if os.path.exists("data/improved_book_tower.keras") else None

    with open("data/improved_unique_isbns.json", "r", encoding="utf-8") as f:
        unique_isbns = json.load(f)
    with open("data/improved_location_map.json", "r", encoding="utf-8") as f:
        location_map = json.load(f)
    with open("data/improved_user_map.json", "r", encoding="utf-8") as f:
        user_map = json.load(f)

    trained_path = "data/trained_book_embeddings.npy"
    if os.path.exists(trained_path):
        all_book_embeddings = tf.constant(np.load(trained_path), dtype=tf.float32)
    else:
        sbert_matrix = np.load("data/improved_sbert_matrix.npy")
        if book_tower is not None:
            book_indices = tf.constant(list(range(len(unique_isbns))), dtype=tf.int64)
            all_book_embeddings = tf.nn.l2_normalize(book_tower.predict(book_indices, verbose=0), axis=1)
        else:
            all_book_embeddings = tf.nn.l2_normalize(tf.constant(sbert_matrix, dtype=tf.float32), axis=1)

    book_categories = {}
    cats_idx_path = "data/improved_book_categories.json"
    cat_vocab_path = "data/improved_category_vocab.json"
    if os.path.exists(cats_idx_path) and os.path.exists(cat_vocab_path):
        with open(cats_idx_path, "r", encoding="utf-8") as f:
            isbn_to_cat_idx = json.load(f) or {}
        with open(cat_vocab_path, "r", encoding="utf-8") as f:
            cat_vocab = json.load(f) or []
        for isbn in unique_isbns:
            idxs = isbn_to_cat_idx.get(isbn, [])
            book_categories[isbn] = [cat_vocab[i] for i in idxs if 0 <= i < len(cat_vocab)]
    if not book_categories or all(len(v) == 0 for v in book_categories.values()):
        mongo_url, db_name = os.getenv("MONGO_URL"), os.getenv("DB_NAME")
        if mongo_url and db_name:
            client = MongoClient(mongo_url)
            db = client[db_name]
            for doc in db.knjige.find({"isbn": {"$in": unique_isbns}}, {"isbn": 1, "kategorije": 1}):
                book_categories[doc["isbn"]] = doc.get("kategorije", []) or []

    return {
        "user_tower": user_tower,
        "user_map": user_map,
        "location_map": location_map,
        "unique_isbns": unique_isbns,
        "all_book_embeddings": all_book_embeddings,
        "book_categories": book_categories  # <-- added
    }