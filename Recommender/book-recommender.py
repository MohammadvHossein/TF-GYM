import tensorflow_ranking as tfr
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Dict, Tuple


dataset_ratings = tfds.load('movielens/100k-ratings', split="train")
dataset_movies = tfds.load('movielens/100k-movies', split="train")

dataset_ratings = dataset_ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

dataset_movies = dataset_movies.map(lambda x: x["movie_title"])
dataset_users = dataset_ratings.map(lambda x: x["user_id"])

user_ids_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
user_ids_vocab.adapt(dataset_users.batch(1000))

movie_titles_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
movie_titles_vocab.adapt(dataset_movies.batch(1000))

def key_function(x):
    return user_ids_vocab(x["user_id"])

def reduce_function(_, dataset):
    return dataset.batch(100)

dataset_train = dataset_ratings.group_by_window(
    key_func=key_function, reduce_func=reduce_function, window_size=100)

def features_and_labels(x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    labels = x.pop("user_rating")
    return x, labels

dataset_train = dataset_train.map(features_and_labels)
dataset_train = dataset_train.ragged_batch(batch_size=32)

class MovieLensRankingModel(tf.keras.Model):
    def __init__(self, user_vocab, movie_vocab):
        super().__init__()
        self.user_vocab = user_vocab
        self.movie_vocab = movie_vocab
        self.user_embedding = tf.keras.layers.Embedding(
            user_vocab.vocabulary_size(), 64)
        self.movie_embedding = tf.keras.layers.Embedding(
            movie_vocab.vocabulary_size(), 64)

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.user_embedding(self.user_vocab(features["user_id"]))
        movie_embeddings = self.movie_embedding(
            self.movie_vocab(features["movie_title"]))
        return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)

model = MovieLensRankingModel(user_ids_vocab, movie_titles_vocab)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss_function = tfr.keras.losses.get(
    loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)

evaluation_metrics = [
    tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True)
]

model.compile(optimizer=optimizer, loss=loss_function, metrics=evaluation_metrics)

model.fit(dataset_train, epochs=2)

for movie_titles in dataset_movies.batch(2000):
    break

user = "42"

inputs = {
    "user_id": tf.expand_dims(tf.repeat(user, repeats=movie_titles.shape[0]), axis=0),
    "movie_title": tf.expand_dims(movie_titles, axis=0)
}
scores = model(inputs)
sorted_titles = tfr.utils.sort_by_scores(
    scores, [tf.expand_dims(movie_titles, axis=0)])[0][0, :5]

print(f"Top 5 recommendations for user {user}:")
for name in sorted_titles:
    print(str(name.numpy()).replace("b'","").replace("'",""))