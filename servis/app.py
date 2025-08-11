import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify
from model_loader import load_all
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

loaded_components = load_all()
user_tower = loaded_components["user_tower"]
user_map = loaded_components["user_map"]
location_map = loaded_components["location_map"]
unique_isbns = loaded_components["unique_isbns"]
all_book_embeddings = loaded_components["all_book_embeddings"]
book_categories = loaded_components["book_categories"]

category_vocab = loaded_components.get("category_vocab", None)
if category_vocab is None:
    try:
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, "data", "improved_category_vocab.json"), encoding="utf-8") as f:
            category_vocab = json.load(f)
    except Exception:
        category_vocab = None

def _normalize_book_categories(book_cats, vocab):
    if not book_cats:
        return {}
    sample = next((v for v in book_cats.values() if isinstance(v, list) and v), [])
    converted = book_cats
    if sample and isinstance(sample[0], int) and vocab:
        idx_to_cat = {i: vocab[i] for i in range(len(vocab))}
        converted = {isbn: [idx_to_cat.get(i) for i in idxs if i in idx_to_cat] for isbn, idxs in book_cats.items()}
    return {isbn: [str(c).lower().strip() for c in (cats or []) if c is not None] for isbn, cats in converted.items()}

book_categories = _normalize_book_categories(book_categories, category_vocab)
isbn_to_idx = {isbn: i for i, isbn in enumerate(unique_isbns)}
GAMMA = 0.7 

def build_user_content_profile(user_oid, top_cats=None):
    interactions = list(db.interakcije.find(
        {"korisnik_id": user_oid, "tip_interakcije": {"$in": ["like", "cart", "order"]}},
        {"isbn": 1}
    ).limit(300))
    idx = [isbn_to_idx[x["isbn"]] for x in interactions if x.get("isbn") in isbn_to_idx]
    if not idx and top_cats:
        in_cat_idx = [i for i, isbn in enumerate(unique_isbns)
                      if any(cat in set(top_cats) for cat in book_categories.get(isbn, []))]
        if in_cat_idx:
            emb = tf.gather(all_book_embeddings, tf.constant(in_cat_idx, dtype=tf.int64))
            return tf.nn.l2_normalize(tf.reduce_mean(emb, axis=0), axis=-1)
        return None
    if not idx:
        return None
    emb = tf.gather(all_book_embeddings, tf.constant(idx, dtype=tf.int64))
    return tf.nn.l2_normalize(tf.reduce_mean(emb, axis=0), axis=-1)

load_dotenv()
mongo_client = MongoClient(os.getenv("MONGO_URL"))
db = mongo_client[os.getenv("DB_NAME")]
bestseller_isbns = set(b['isbn'] for b in db.knjige.find({"bestseller_lists.0": {"$exists": True}}, {"isbn": 1}))

app = Flask(__name__)

def user_top_categories(db, user_oid, topk=3):
    inter = list(db.interakcije.aggregate([
        {"$match": {"korisnik_id": user_oid, "tip_interakcije": {"$in": ["like","cart","order"]}}}, 
        {"$lookup": {"from": "knjige", "localField": "isbn", "foreignField": "isbn", "as": "book"}},
        {"$unwind": "$book"},
        {"$unwind": {"path": "$book.kategorije", "preserveNullAndEmptyArrays": False}},
        {"$group": {"_id": "$book.kategorije", "cnt": {"$sum": 1}}},
        {"$sort": {"cnt": -1}}, {"$limit": topk}
    ]))
    return [str(d["_id"]).lower().strip() for d in inter]

@app.route("/recommendations/<user_id>", methods=['GET'])
def get_recommendations(user_id):
    try:
        user_oid = ObjectId(user_id)
    except Exception:
        return jsonify({"error": "invalid user id"}), 400

    last_order = db.narudzbe.find_one({"korisnik_id": user_oid}, sort=[("vrijeme", -1)])
    user_location = last_order["adresa"]["drzava"] if last_order and "adresa" in last_order and "drzava" in last_order["adresa"] else "Nepoznato"

    user_doc = db.korisnici.find_one({"_id": user_oid}, {"favoriti": 1})
    favorites_isbns = set(map(str, (user_doc or {}).get("favoriti", [])))

    banned_isbns = bestseller_isbns | favorites_isbns
    banned_idx = {isbn_to_idx[isbn] for isbn in banned_isbns if isbn in isbn_to_idx}
    allowed_mask = np.ones(len(unique_isbns), dtype=bool)
    if banned_idx:
        idx_arr = np.fromiter(banned_idx, dtype=np.int64, count=len(banned_idx))
        allowed_mask[idx_arr] = False

    top_cats = user_top_categories(db, user_oid, topk=3)
    print(f"Top categories for user {user_id}: {top_cats}")
    top_cats_set = set(top_cats)

    profile = build_user_content_profile(user_oid, top_cats=top_cats)
    content_scores = tf.linalg.matvec(all_book_embeddings, profile) if profile is not None else tf.zeros(len(unique_isbns), dtype=tf.float32)

    if user_tower is not None and user_id in user_map:
        uid = tf.constant([user_map[user_id]], dtype=tf.int64)
        loc = tf.constant([location_map.get(user_location, 0)], dtype=tf.int64)
        u = tf.nn.l2_normalize(user_tower([uid, loc], training=False), axis=-1) 
        user_scores = tf.squeeze(tf.matmul(all_book_embeddings, tf.transpose(u)), axis=-1)
    else:
        user_scores = tf.zeros(len(unique_isbns), dtype=tf.float32)

    final_scores = GAMMA * content_scores + (1.0 - GAMMA) * user_scores

    topk = 20
    if top_cats_set:
        in_cat = [i for i, isbn in enumerate(unique_isbns)
                  if allowed_mask[i] and any(cat in top_cats_set for cat in book_categories.get(isbn, []))]
        in_cat = tf.constant(in_cat, dtype=tf.int64)
        if tf.size(in_cat) > 0:
            primary_k = max(1, int(0.8 * topk))
            in_cat_scores = tf.gather(final_scores, in_cat)
            in_cat_top = tf.gather(in_cat, tf.argsort(in_cat_scores, direction='DESCENDING')[:primary_k]).numpy().tolist()
            mask = allowed_mask.copy()
            if in_cat_top:
                mask[np.array(in_cat_top, dtype=np.int64)] = False
            backfill_scores = final_scores.numpy()
            backfill_scores[~mask] = -1e9
            need = topk - len(in_cat_top)
            backfill = np.argsort(-backfill_scores)[:max(0, need)].tolist()
            top_idx = in_cat_top + backfill
        else:
            scores = final_scores.numpy()
            scores[~allowed_mask] = -1e9
            top_idx = np.argsort(-scores)[:topk].tolist()
    else:
        scores = final_scores.numpy()
        scores[~allowed_mask] = -1e9
        top_idx = np.argsort(-scores)[:topk].tolist()

    rec_isbns = [unique_isbns[i] for i in top_idx]
    return jsonify({"user_id": user_id, "location": user_location, "top_categories": list(top_cats_set), "recommendations": rec_isbns})

@app.route("/debug/<user_id>", methods=['GET'])
def debug_user(user_id):
    """Debug endpoint to analyze user preferences"""
    try:
        user_object_id = ObjectId(user_id)
        interactions = list(db.interakcije.find({"korisnik_id": user_object_id}))
        
        category_counts = {}
        for interaction in interactions:
            book = db.knjige.find_one({"isbn": interaction['isbn']}, {"kategorije": 1, "naslov": 1})
            if book:
                for category in book.get('kategorije', []):
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "user_id": user_id,
            "total_interactions": len(interactions),
            "top_categories": sorted_categories[:10],
            "in_training_data": user_id in user_map
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5001, debug=False)