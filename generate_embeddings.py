import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer

def generate_and_store_embeddings():
    load_dotenv()
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DB_NAME")

    client = MongoClient(mongo_url)
    db = client[db_name]
    knjige_collection = db.knjige
    
    books_to_process = list(knjige_collection.find(
        {"sbert_embedding": {"$exists": False}},
        {"_id": 1, "naslov": 1, "opis": 1, "kategorije": 1}
    ))

    if not books_to_process:
        print("Sve knjige već imaju embedding. Izlazim.")
        client.close()
        return

    print(f"Pronađeno {len(books_to_process)} knjiga za obradu.")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    book_texts = []
    for book in books_to_process:
        categories = book.get("kategorije", [])
        category_string = ". ".join(categories)
        full_text = f"{category_string}. {book.get('naslov', '')}. {book.get('opis', '')}"
        book_texts.append(full_text)

    print("Generiranje embeddinga... (ovo može potrajati)")
    embeddings = model.encode(book_texts, show_progress_bar=True)
    
    print("Priprema operacija za upis u bazu...")
    update_operations = []
    for book, embedding in zip(books_to_process, embeddings):
        operation = UpdateOne(
            {"_id": book["_id"]},
            {"$set": {"sbert_embedding": embedding.tolist()}}
        )
        update_operations.append(operation)

    print("Upisivanje embeddinga u bazu podataka (bulk write)...")
    result = knjige_collection.bulk_write(update_operations)

    print(f"Operacija završena. Ažurirano {result.modified_count} dokumenata.")
    
    client.close()

if __name__ == "__main__":
    generate_and_store_embeddings()