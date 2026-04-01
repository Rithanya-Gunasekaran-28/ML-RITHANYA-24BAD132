#sc-1
# RITHANYA.G
# ROLL NO: 24BAD132
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
ratings = pd.read_csv(r'C:\Users\HP\Downloads\archive (20)\ml-latest-small\ratings.csv')
movies = pd.read_csv(r'C:\Users\HP\Downloads\archive (20)\ml-latest-small\movies.csv')
# -----------------------------
# STEP 2: DATA INSPECTION
# -----------------------------
print("RITHANYA.G 24BAD132")
print("Dataset Info:\n", ratings.info())
print("\nMissing Values:\n", ratings.isnull().sum())

# -----------------------------
# STEP 3: USER-ITEM MATRIX
# -----------------------------
user_item = ratings.pivot_table(index='userId',
                                columns='movieId',
                                values='rating').fillna(0)
# -----------------------------
# STEP 4: SPARSITY
# -----------------------------
sparsity = 1.0 - (np.count_nonzero(user_item) / user_item.size)
print("\nMatrix Sparsity:", round(sparsity, 4))
# -----------------------------
# STEP 5: SIMILARITY
# -----------------------------
user_similarity = cosine_similarity(user_item)
user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item.index,
                                 columns=user_item.index)
# -----------------------------
# STEP 6: SIMILAR USERS
# -----------------------------
def get_similar_users(user_id, n=5):
    return user_similarity_df[user_id].sort_values(ascending=False)[1:n+1]
# -----------------------------
# STEP 7: PREDICT RATING
# -----------------------------
def predict_rating(user_id, movie_id):
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    
    num, den = 0, 0
    for other_user, sim_score in sim_users.items():
        rating = user_item.loc[other_user, movie_id]
        if rating > 0:
            num += sim_score * rating
            den += sim_score
            
    return num / den if den != 0 else 0

# -----------------------------
# STEP 8: RECOMMENDATION
# -----------------------------
def recommend_movies(user_id, n=5):
    unseen_movies = user_item.loc[user_id][user_item.loc[user_id] == 0].index
    predictions = []
    for movie_id in unseen_movies:
        pred = predict_rating(user_id, movie_id)
        predictions.append((movie_id, pred))
    predictions.sort(key=lambda x: x[1], reverse=True)
    results = []
    for movie_id, score in predictions[:n]:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        results.append((title, round(score, 2)))
    return results
# -----------------------------
# STEP 9: EVALUATION
# -----------------------------
def evaluate(n_samples=1000):
    actual, predicted = [], []
    sample = ratings.sample(n_samples)
    for _, row in sample.iterrows():
        user = row['userId']
        movie = row['movieId']
        true_rating = row['rating']
        pred_rating = predict_rating(user, movie)
        if pred_rating > 0:
            actual.append(true_rating)
            predicted.append(pred_rating)
    rmse = sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print("\nRMSE:", round(rmse, 3))
    print("MAE:", round(mae, 3))
# -----------------------------
# STEP 10: VISUALIZATION
# -----------------------------
# 1. User-Item Matrix Heatmap
plt.figure()
plt.imshow(user_item.iloc[:50, :50])
plt.title("User-Item Matrix Heatmap")
plt.colorbar()
plt.show()
# 2. Similarity Matrix Heatmap
plt.figure()
plt.imshow(user_similarity_df.iloc[:50, :50])
plt.title("User Similarity Matrix")
plt.colorbar()
plt.show()
# 3. Top Recommendations Bar Chart
def plot_recommendations(user_id):
    recs = recommend_movies(user_id)
    movies_names = [i[0] for i in recs]
    scores = [i[1] for i in recs]
    plt.figure()
    plt.barh(movies_names, scores)
    plt.xlabel("Score")
    plt.title(f"Top Recommendations for User {user_id}")
    plt.gca().invert_yaxis()
    plt.show()
# -----------------------------
# STEP 11: RUN
# -----------------------------
user_id = 1
print("\nTop Similar Users:\n", get_similar_users(user_id))
print("\nRecommended Movies:\n")
for movie, score in recommend_movies(user_id):
    print(movie, "->", score)
evaluate()
plot_recommendations(user_id)