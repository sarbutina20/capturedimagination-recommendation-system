import os
import random
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json

def prepare_improved_data():
    load_dotenv()
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DB_NAME")
    client = MongoClient(mongo_url)
    db = client[db_name]

    interaction_weights = {
        'like': 1.0,
        'cart': 2.0,
        'order': 5.0
    }
    
    books_with_embeddings = list(db.knjige.find(
        {"sbert_embedding": {"$exists": True}}, 
        {"isbn": 1, "sbert_embedding": 1, "kategorije": 1}
    ))
    
    all_isbns = set([book['isbn'] for book in books_with_embeddings])
    all_users = list(db.korisnici.find({}, {"_id": 1}))
    
    isbn_to_embedding = {book['isbn']: np.array(book['sbert_embedding']) 
                        for book in books_with_embeddings}
    book_categories_map = {book['isbn']: book.get('kategorije', []) 
                           for book in books_with_embeddings}

    all_categories = sorted({cat for cats in book_categories_map.values() for cat in cats})
    cat_to_idx = {c: i for i, c in enumerate(all_categories)}
    book_cats_idx = {isbn: [cat_to_idx[c] for c in book_categories_map[isbn] if c in cat_to_idx]
                     for isbn in book_categories_map}

    os.makedirs("data", exist_ok=True)
    with open("data/improved_category_vocab.json", "w", encoding="utf-8") as f:
        json.dump(all_categories, f, ensure_ascii=False)
    with open("data/improved_book_categories.json", "w", encoding="utf-8") as f:
        json.dump(book_cats_idx, f, ensure_ascii=False)
    ordered_isbns = list(isbn_to_embedding.keys())
    embedding_matrix = np.array([isbn_to_embedding[isbn] for isbn in ordered_isbns])

    cosine_sim_matrix = cosine_similarity(embedding_matrix)
    
    similar_books_map = {}
    for i, isbn in enumerate(ordered_isbns):
        similar_indices = np.argsort(cosine_sim_matrix[i])[::-1]
        top_similar_isbns = [
            ordered_isbns[j] for j in similar_indices if ordered_isbns[j] != isbn
        ][:50]
        similar_books_map[isbn] = top_similar_isbns

    user_location_map = {}
    for user in all_users:
        user_id = user["_id"]
        last_order = db.narudzbe.find_one({"korisnik_id": user_id}, sort=[("vrijeme", -1)])
        if last_order and "adresa" in last_order and "drzava" in last_order["adresa"]:
            user_location_map[str(user_id)] = last_order["adresa"]["drzava"]
        else:
            user_location_map[str(user_id)] = "Nepoznato"

    user_pos_interactions = defaultdict(set)
    user_categories = defaultdict(set)
    interaction_query = {"tip_interakcije": {"$in": list(interaction_weights.keys())}}
    
    for interaction in db.interakcije.find(interaction_query):
        user_id = str(interaction['korisnik_id'])
        isbn = interaction['isbn']
        if isbn in all_isbns:
            user_pos_interactions[user_id].add(isbn)
            for category in book_categories_map.get(isbn, []):
                user_categories[user_id].add(category)

    training_data_with_weights = []
    
    for interaction in db.interakcije.find(interaction_query):
        user_id = str(interaction['korisnik_id'])
        positive_isbn = interaction['isbn']
        
        if positive_isbn not in all_isbns:
            continue
            
        interaction_type = interaction['tip_interakcije']
        weight = interaction_weights.get(interaction_type, 1.0)
        user_location = user_location_map.get(user_id, "Nepoznato")
        
        negative_strategies = []
        
        if positive_isbn in similar_books_map:
            hard_candidates = similar_books_map[positive_isbn]
            hard_negatives = [neg for neg in hard_candidates if neg not in user_pos_interactions[user_id]]
            negative_strategies.extend([("hard", neg) for neg in hard_negatives[:3]])
        
        user_cats = user_categories.get(user_id, set())
        if user_cats:
            medium_candidates = [
                book['isbn'] for book in books_with_embeddings 
                if book['isbn'] not in user_pos_interactions[user_id] and not set(book_categories_map.get(book['isbn'], [])).intersection(user_cats)
            ]
            if medium_candidates:
                medium_negatives = random.sample(medium_candidates, min(2, len(medium_candidates)))
                negative_strategies.extend([("medium", neg) for neg in medium_negatives])
        
        easy_candidates = list(all_isbns - user_pos_interactions[user_id])
        if easy_candidates:
            easy_negatives = random.sample(easy_candidates, min(1, len(easy_candidates)))
            negative_strategies.extend([("easy", neg) for neg in easy_negatives])
        
        strategy_weights = {"hard": weight * 2.0, "medium": weight * 1.25, "easy": weight}
        
        for strategy, negative_isbn in negative_strategies:
            final_weight = strategy_weights[strategy]
            training_data_with_weights.append({
                "user_id": user_id,
                "location": user_location,
                "positive_isbn": positive_isbn,
                "negative_isbn": negative_isbn,
                "weight": final_weight,
                "strategy": strategy
            })

    df = pd.DataFrame(training_data_with_weights)

    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(os.path.join(output_dir, "improved_training_data.csv"), index=False)

    np.save(os.path.join(output_dir, "book_embeddings_map.npy"), isbn_to_embedding)

    print(f"Distribucija strategija:\n{df['strategy'].value_counts()}")
    client.close()

if __name__ == "__main__":
    prepare_improved_data()