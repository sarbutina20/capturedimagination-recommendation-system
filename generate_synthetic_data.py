import os
import random
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from collections import defaultdict
from random import choices

CROATIAN_NAMES = [
    "Luka", "Ivan", "Marko", "Josip", "Filip", "Petar", "David", "Ante",
    "Karlo", "Dominik", "Lovro", "Fran", "Bruno", "Mateo", "Noa", "Jakov",
    "Matej", "Roko", "Leon", "Toma", "Hrvoje", "Dario", "Antonio", "Marijan",
    "Stjepan", "Nikola", "Mario", "Kristijan", "Viktor", "Andrija", "Tomislav",
    "Damir", "Zvonimir", "Miroslav", "Branimir", "Domagoj", "Vedran", "Goran",
    "Ana", "Marija", "Mia", "Lucija", "Ema", "Nika", "Lana", "Petra",
    "Sara", "Elena", "Klara", "Magdalena", "Nora", "Tara", "Iva", "Karla",
    "Laura", "Tea", "Anja", "Dora", "Luna", "Valentina", "Marta", "Nina",
    "Katarina", "Ivana", "Kristina", "Maja", "Nikolina", "Anđela", "Barbara",
    "Gabriela", "Ines", "Jelena", "Ksenija", "Mirela", "Petra", "Renata",
    "Sanja", "Tatjana", "Vesna", "Zora", "Božica", "Slavica"
    ]

PERSONAS = {
    "business_pro": ["business", "economics", "management", "finance"],
    "romance_reader": ["romance", "contemporary", "love"],
    "graphic_novel_fan": ["graphic novels", "comics", "manga", "art"],
    "history_buff": ["history", "biography", "war", "ancient civilisations", "archaeology"],
    "philosophy_student": ["philosophy", "psychology", "political science", "social science"],
    "sci_fi_fan": ["science fiction", "dystopian", "space opera", "cyberpunk"],
    "fantasy_lover": ["fantasy", "magic", "mythology", "juvenile fiction"],
    "thriller_mystery_fan": ["thriller", "mystery", "suspense", "crime", "detective"],
    "young_adult_reader": ["young adult", "juvenile fiction"],
    "wellness_seeker": ["self-help", "health", "mind", "body", "spirituality", "psychology"],
    "horror_fan": ["horror", "supernatural", "ghosts"],
    "classic_literature": ["classics", "literary fiction"],
    "poetry_lover": ["poetry", "verse"],
    "art_and_design": ["art", "design", "photography", "architecture", "fashion"],
    "music_enthusiast": ["music", "musicians", "biography"],
    "traveler": ["travel", "adventure", "nature"],
    "foodie": ["cooking", "cookbooks", "food", "wine"],
    "tech_guru": ["technology", "computers", "programming", "science"],
    "sports_fan": ["sports", "recreation", "biography"],
    "true_crime_junkie": ["true crime", "crime", "biography"],
    "science_enthusiast": ["science", "physics", "biology", "astronomy"],
    "historical_fiction": ["historical fiction", "fiction", "history"],
    "adventure_seeker": ["adventure", "action", "fiction"],
    "political_junkie": ["political science", "government", "current events"],
    "gardening_expert": ["gardening", "house plants", "nature"],
    "religion_scholar": ["religion", "spirituality", "theology"],
    "education_professional": ["education", "study aids", "reference"],
    "law_and_order": ["law", "true crime", "government"],
    "crafts_and_hobbies": ["crafts & hobbies", "antiques & collectibles"],
    "performing_arts": ["performing arts", "drama", "theater"]
}

NUM_NEW_SYNTHETIC_USERS = 500
COUNTRIES = ["Hrvatska", "Srbija", "Slovenija", "BiH", "Austrija", "Njemačka", "Italija"]

SYNTHETIC_FIELD = "synthetic"

def generate_data():
    load_dotenv()
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("DB_NAME")

    client = MongoClient(mongo_url)
    db = client[db_name]

    korisnici_collection = db.korisnici
    new_users = []
    newly_inserted_ids = []

    existing_usernames = set(user['KorisnickoIme'] for user in korisnici_collection.find({}, {"KorisnickoIme": 1}))
    for _ in range(NUM_NEW_SYNTHETIC_USERS):
        while True:
            username = f"{random.choice(CROATIAN_NAMES).lower()}{random.randint(1, 999)}"
            if username not in existing_usernames:
                existing_usernames.add(username)
                break
        
        new_user = {
            "KorisnickoIme": username,
            "Email": f"{username}@example.com",
            "Lozinka": "placeholder_hash",
            "Uloga_ID": ObjectId("64e22057f9497eba62ed9513"),
            "favoriti": [],
            SYNTHETIC_FIELD: True,
        }
        new_users.append(new_user)
    
    if new_users:
        result = korisnici_collection.insert_many(new_users)
        newly_inserted_ids = result.inserted_ids

    if not newly_inserted_ids:
        print("Nije dodan nijedan novi korisnik. Izlazim.")
        client.close()
        return

    users_to_process = list(db.korisnici.find({"_id": {"$in": newly_inserted_ids}}))
    books = list(db.knjige.find({}, {"isbn": 1, "kategorije": 1, "naslov": 1, "opis": 1}))
    
    persona_book_isbns = defaultdict(list)
    for book in books:
        book_text = " ".join(book.get("kategorije", [])).lower() + " " + book.get("opis", "").lower()
        for persona, keywords in PERSONAS.items():
            if any(keyword in book_text for keyword in keywords):
                persona_book_isbns[persona].append(book['isbn'])

    user_personas = {}
    persona_names = list(PERSONAS.keys())
    for user in users_to_process:
        num_personas = random.choices([1, 2], weights=[0.7, 0.3], k=1)[0]
        user_personas[str(user["_id"])] = random.sample(persona_names, num_personas)

    all_new_interactions = []
    
    for user_id_str, assigned_personas in user_personas.items():
        user_id = ObjectId(user_id_str)
        
        possible_isbns = set()
        for persona in assigned_personas:
            possible_isbns.update(persona_book_isbns.get(persona, []))
        
        if not possible_isbns: continue

        possible_isbns = list(possible_isbns)
        num_likes = min(random.randint(10, 25), len(possible_isbns))
        liked_books = random.sample(possible_isbns, num_likes)
        
        num_cart_adds = min(random.randint(int(num_likes * 0.3), int(num_likes * 0.5)), len(liked_books))
        cart_books = random.sample(liked_books, num_cart_adds)
        
        num_orders = min(random.randint(int(num_cart_adds * 0.7), num_cart_adds), len(cart_books))
        ordered_books = random.sample(cart_books, num_orders)
        
        for isbn in liked_books:
            all_new_interactions.append({
                "korisnik_id": user_id,
                "isbn": isbn,
                "tip_interakcije": 'like',
                "vrijeme": datetime.utcnow(),
                SYNTHETIC_FIELD: True,
            })
        for isbn in cart_books:
            all_new_interactions.append({
                "korisnik_id": user_id,
                "isbn": isbn,
                "tip_interakcije": 'cart',
                "vrijeme": datetime.utcnow(),
                SYNTHETIC_FIELD: True,
            })
        for isbn in ordered_books:
            adresa = None
            if random.random() < 0.7:
                adresa = {"drzava": choices(COUNTRIES, weights=[60,8,8,8,6,6,4], k=1)[0]}
            all_new_interactions.append({
                "korisnik_id": user_id,
                "isbn": isbn,
                "tip_interakcije": 'order',
                "vrijeme": datetime.utcnow(),
                SYNTHETIC_FIELD: True,
            })
            db.narudzbe.insert_one({
                "korisnik_id": user_id,
                "stavke": [{"isbn": isbn, "kolicina": 1}],
                "vrijeme": datetime.utcnow(),
                "adresa": adresa or {"drzava": "Hrvatska"},
                SYNTHETIC_FIELD: True,
            })

    if all_new_interactions:
        db.interakcije.insert_many(all_new_interactions)
        print("Sintetički podaci za nove korisnike uspješno generirani.")
    else:
        print("Nije generirana nijedna nova interakcija.")

    client.close()

if __name__ == "__main__":
    generate_data()