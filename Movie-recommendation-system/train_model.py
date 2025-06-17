import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Load ratings and movies
ratings = pd.read_csv('dataset/ratings.csv')
movies = pd.read_csv('dataset/movies.csv')

# Merge ratings with movie titles
df = pd.merge(ratings, movies, on='movieId')

# Create user-item matrix (rows = users, columns = movie titles)
pivot_table = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute cosine similarity between users
similarity = cosine_similarity(pivot_table)

# Save model artifacts
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(similarity, 'model/similarity_matrix.joblib')
pivot_table.to_csv('model/user_item_matrix.csv', index=True)

print("Training complete. Matrix and similarity saved.")
