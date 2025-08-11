import os, json
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import AdamW

def load_data_and_embeddings():
    df = pd.read_csv("data/improved_training_data.csv", dtype={
        'user_id': str, 'location': str, 'positive_isbn': str, 
        'negative_isbn': str, 'strategy': str
    })
    df['weight'] = df['weight'].astype(np.float32)
    book_embeddings = np.load("data/book_embeddings_map.npy", allow_pickle=True).item()
    with open("data/improved_book_categories.json", "r", encoding="utf-8") as f:
        book_cats_idx = json.load(f)
    with open("data/improved_category_vocab.json", "r", encoding="utf-8") as f:
        category_vocab = json.load(f)
    return df, book_embeddings, {"book_cats_idx": book_cats_idx, "category_vocab": category_vocab}

class ImprovedRecommenderModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_locations, unique_isbns, 
                 book_embeddings_dict, category_vocab, book_cats_idx,
                 embedding_dim=48, temperature=0.05, alpha=3.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        user_keys = tf.constant(list(unique_user_ids))
        user_values = tf.constant(list(range(len(unique_user_ids))), dtype=tf.int64)
        loc_keys = tf.constant(list(unique_locations))
        loc_values = tf.constant(list(range(len(unique_locations))), dtype=tf.int64)
        isbn_keys = tf.constant(list(unique_isbns))
        isbn_values = tf.constant(list(range(len(unique_isbns))), dtype=tf.int64)

        self.user_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(user_keys, user_values), default_value=-1)
        self.loc_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(loc_keys, loc_values), default_value=-1)
        self.isbn_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(isbn_keys, isbn_values), default_value=-1)

        sbert_dim = len(next(iter(book_embeddings_dict.values())))
        sbert_matrix = np.zeros((len(unique_isbns), sbert_dim), dtype=np.float32)
        isbn_to_idx = {isbn: i for i, isbn in enumerate(unique_isbns)}
        for isbn, idx in isbn_to_idx.items():
            vec = book_embeddings_dict.get(isbn)
            if vec is not None:
                sbert_matrix[idx] = np.asarray(vec, dtype=np.float32)
        self.sbert_embedding_matrix = tf.constant(sbert_matrix)

        num_cats = len(category_vocab)
        book_cat = np.zeros((len(unique_isbns), num_cats), dtype=np.float32)
        for isbn, idx in isbn_to_idx.items():
            for cidx in book_cats_idx.get(isbn, []):
                if 0 <= cidx < num_cats:
                    book_cat[idx, cidx] = 1.0
        self.book_category_multi_hot = tf.constant(book_cat)

        self.user_cat_emb = tf.keras.layers.Embedding(len(unique_user_ids), embedding_dim)
        self.user_txt_emb = tf.keras.layers.Embedding(len(unique_user_ids), embedding_dim)
        self.loc_proj_cat = tf.keras.layers.Embedding(len(unique_locations), embedding_dim)
        self.loc_proj_txt = tf.keras.layers.Embedding(len(unique_locations), embedding_dim)
        self.user_dropout = tf.keras.layers.Dropout(0.3)

        self.book_cat_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.book_txt_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(embedding_dim)
        ])

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.easy_loss_tracker = tf.keras.metrics.Mean(name="easy_loss")
        self.medium_loss_tracker = tf.keras.metrics.Mean(name="medium_loss")
        self.hard_loss_tracker = tf.keras.metrics.Mean(name="hard_loss")

    def encode_user(self, user_ids, locations):
        ui = self.user_table.lookup(user_ids)
        li = self.loc_table.lookup(locations)
        u_cat = self.user_cat_emb(ui)
        u_txt = self.user_txt_emb(ui)
        l_cat = self.loc_proj_cat(li)
        l_txt = self.loc_proj_txt(li)
        u = self.alpha * (u_cat + l_cat) + self.beta * (u_txt + l_txt)
        u = tf.nn.l2_normalize(self.user_dropout(u), axis=-1)
        return u

    def encode_book(self, isbn_ids):
        bi = self.isbn_table.lookup(isbn_ids)
        b_txt_in = tf.gather(self.sbert_embedding_matrix, bi)
        b_cat_in = tf.gather(self.book_category_multi_hot, bi)
        b_cat = self.book_cat_dense(b_cat_in)
        b_txt = self.book_txt_dense(b_txt_in)
        b = self.alpha * b_cat + self.beta * b_txt
        b = tf.nn.l2_normalize(b, axis=-1)
        return b

    def call(self, inputs, training=False):
        user_id, location, pos_isbn, neg_isbn = inputs
        u = self.encode_user(user_id, location)
        p = self.encode_book(pos_isbn)
        n = self.encode_book(neg_isbn)
        pos_sim = tf.reduce_sum(u * p, axis=-1) / self.temperature
        neg_sim = tf.reduce_sum(u * n, axis=-1) / self.temperature
        return tf.stack([pos_sim, neg_sim], axis=-1)

    @property
    def metrics(self):
        return [self.loss_tracker, self.easy_loss_tracker,
                self.medium_loss_tracker, self.hard_loss_tracker]

    def train_step(self, data):
        user_id, location, pos_isbn, neg_isbn, weight, strategy = data
        strategy = tf.cast(strategy, tf.string)
        with tf.GradientTape() as tape:
            u = self.encode_user(user_id, location)
            p = self.encode_book(pos_isbn)
            n = self.encode_book(neg_isbn)
            pos_sim = tf.reduce_sum(u * p, axis=-1) / self.temperature
            neg_sim = tf.reduce_sum(u * n, axis=-1) / self.temperature
            pair_loss = tf.nn.softplus(-(pos_sim - neg_sim))
            wloss = pair_loss * tf.cast(weight, tf.float32)
            loss = tf.reduce_mean(wloss)

            easy_mask = tf.equal(strategy, tf.constant('easy'))
            med_mask = tf.equal(strategy, tf.constant('medium'))
            hard_mask = tf.equal(strategy, tf.constant('hard'))
            easy_loss = tf.reduce_mean(tf.boolean_mask(wloss, easy_mask))
            med_loss = tf.reduce_mean(tf.boolean_mask(wloss, med_mask))
            hard_loss = tf.reduce_mean(tf.boolean_mask(wloss, hard_mask))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.easy_loss_tracker.update_state(easy_loss)
        self.medium_loss_tracker.update_state(med_loss)
        self.hard_loss_tracker.update_state(hard_loss)
        return {"loss": self.loss_tracker.result(),
                "easy_loss": self.easy_loss_tracker.result(),
                "medium_loss": self.medium_loss_tracker.result(),
                "hard_loss": self.hard_loss_tracker.result()}

    def test_step(self, data):
        user_id, location, pos_isbn, neg_isbn, weight, strategy = data
        strategy = tf.cast(strategy, tf.string)
        u = self.encode_user(user_id, location)
        p = self.encode_book(pos_isbn)
        n = self.encode_book(neg_isbn)
        pos_sim = tf.reduce_sum(u * p, axis=-1) / self.temperature
        neg_sim = tf.reduce_sum(u * n, axis=-1) / self.temperature
        pair_loss = tf.nn.softplus(-(pos_sim - neg_sim))
        wloss = pair_loss * tf.cast(weight, tf.float32)
        loss = tf.reduce_mean(wloss)

        easy_mask = tf.equal(strategy, tf.constant('easy'))
        med_mask = tf.equal(strategy, tf.constant('medium'))
        hard_mask = tf.equal(strategy, tf.constant('hard'))
        easy_loss = tf.reduce_mean(tf.boolean_mask(wloss, easy_mask))
        med_loss = tf.reduce_mean(tf.boolean_mask(wloss, med_mask))
        hard_loss = tf.reduce_mean(tf.boolean_mask(wloss, hard_mask))

        self.loss_tracker.update_state(loss)
        self.easy_loss_tracker.update_state(easy_loss)
        self.medium_loss_tracker.update_state(med_loss)
        self.hard_loss_tracker.update_state(hard_loss)

        return {"loss": self.loss_tracker.result(),
                "easy_loss": self.easy_loss_tracker.result(),
                "medium_loss": self.medium_loss_tracker.result(),
                "hard_loss": self.hard_loss_tracker.result()}

def main():
    print("Učitavanje podataka...")
    df, book_embeddings, extra = load_data_and_embeddings()
    unique_users = sorted(df['user_id'].unique().tolist())
    unique_locations = sorted([str(loc) for loc in df['location'].unique()])
    unique_isbns = sorted(pd.concat([df['positive_isbn'], df['negative_isbn']]).unique().tolist())
    print(f"Korisnici: {len(unique_users)}, Lokacije: {len(unique_locations)}, Knjige: {len(unique_isbns)}")
    print(f"Training uzoraka: {len(df)}")

    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['strategy'])
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train['user_id'].values, train['location'].values, 
        train['positive_isbn'].values, train['negative_isbn'].values,
        train['weight'].values, train['strategy'].values
    )).shuffle(len(train)).batch(128).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((
        val['user_id'].values, val['location'].values, 
        val['positive_isbn'].values, val['negative_isbn'].values,
        val['weight'].values, val['strategy'].values
    )).batch(128).prefetch(tf.data.AUTOTUNE)

    checkpoint_dir = 'data'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nInicijalizacija modela...")
    model = ImprovedRecommenderModel(
        unique_users, unique_locations, unique_isbns,
        book_embeddings,
        category_vocab=extra["category_vocab"],
        book_cats_idx=extra["book_cats_idx"],
        embedding_dim=48, temperature=0.05, alpha=3.0, beta=1.0
    )

    initial_learning_rate = 3e-4
    model.compile(optimizer=AdamW(learning_rate=initial_learning_rate, weight_decay=0.004),
                  jit_compile=False)

    for batch in train_dataset.take(1):
        u_id, loc, pos_isbn, neg_isbn, _, _ = batch
        _ = model([u_id[:2], loc[:2], pos_isbn[:2], neg_isbn[:2]], training=False)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, min_delta=0.01, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'best_model_checkpoint.weights.h5'),
        monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
    )

    print("\nPočetak treniranja modela...")
    history = model.fit(train_dataset, epochs=40, validation_data=val_dataset,
                        callbacks=[early_stopping, checkpoint, reduce_lr])

    print("\nSpremanje obučenih modela...")
    trained_vecs = []
    batch = 2048
    for i in range(0, len(unique_isbns), batch):
        batch_isbns = tf.constant(unique_isbns[i:i+batch])
        vecs = model.encode_book(batch_isbns)
        trained_vecs.append(vecs.numpy())
    trained_book_embeddings = tf.nn.l2_normalize(
        tf.concat([tf.constant(v) for v in trained_vecs], axis=0), axis=1
    ).numpy()
    np.save("data/trained_book_embeddings.npy", trained_book_embeddings)

    np.save("data/improved_sbert_matrix.npy", model.sbert_embedding_matrix.numpy())
    with open("data/improved_unique_isbns.json", "w") as f:
        json.dump(unique_isbns, f)
    location_map = {loc: i for i, loc in enumerate(unique_locations)}
    with open("data/improved_location_map.json", "w", encoding='utf-8') as f:
        json.dump(location_map, f, ensure_ascii=False)
    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    with open("data/improved_user_map.json", "w") as f:
        json.dump(user_map, f)

    print("\nModeli i mape su uspješno spremljeni.")
    print("Training history:")
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), start=1):
        print(f"Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}")

if __name__ == "__main__":
    main()