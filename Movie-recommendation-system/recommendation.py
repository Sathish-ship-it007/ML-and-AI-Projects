import pandas as pd
import joblib

# Load model artifacts
similarity = joblib.load('model/similarity_matrix.joblib')
pivot_table = pd.read_csv('model/user_item_matrix.csv', index_col=0)

# Ensure index is integer type
pivot_table.index = pivot_table.index.astype(int)

def recommend_movies(user_id, top_n=5):
    try:
        user_id = int(user_id)  # ensure user_id is an int
    except ValueError:
        return ["Invalid User ID"]

    if user_id not in pivot_table.index:
        return ["User ID not found"]

    user_index = list(pivot_table.index).index(user_id)
    similar_users = list(enumerate(similarity[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]

    recommendations = {}
    for i, score in similar_users:
        other_user_id = pivot_table.index[i]
        for movie, rating in pivot_table.loc[other_user_id].items():
            if pivot_table.loc[user_id, movie] == 0 and rating >= 4.0:
                recommendations[movie] = recommendations.get(movie, 0) + score

        if len(recommendations) >= top_n:
            break

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_recs[:top_n]]
