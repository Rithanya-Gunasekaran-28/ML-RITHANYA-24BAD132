#sc-2
# RITHANYA.G - 24BAD132
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# -----------------------------
# 1. LOAD DATASET
# -----------------------------
path = r"C:\Users\HP\Downloads\archive (22)\ratings.csv"
df = pd.read_csv(path)
df = df[['userId','movieId','rating']]
df.columns = ['user_id','item_id','rating']
# Reduce size for speed
df = df.head(10000)
# -----------------------------
# 2. CREATE ITEM-USER MATRIX
# -----------------------------
item_user = df.pivot_table(index='item_id', columns='user_id', values='rating').fillna(0)
# -----------------------------
# 3. ITEM SIMILARITY (COSINE)
# -----------------------------
item_sim = cosine_similarity(item_user)
item_sim_df = pd.DataFrame(item_sim, index=item_user.index, columns=item_user.index)
# -----------------------------
# 4. TOP SIMILAR ITEMS
# -----------------------------
def get_similar_items(item_id, top_n=5):
    return item_sim_df[item_id].sort_values(ascending=False).iloc[1:top_n+1]
print("RITHANYA.G 24BAD132")
print("\nTop similar items for item 1:\n", get_similar_items(1))
# -----------------------------
# 5. RECOMMEND ITEMS
# -----------------------------
def recommend_items(user_id, top_n=5):
    user_ratings = item_user[user_id]
    scores = np.dot(item_sim, user_ratings)
    scores = pd.Series(scores, index=item_user.index)
    watched = user_ratings[user_ratings > 0].index
    scores = scores.drop(watched)
    return scores.sort_values(ascending=False).head(top_n)
print("\nItem-based Recommendations for User 1:\n", recommend_items(1))
# -----------------------------
# 6. USER-BASED (COMPARISON)
# -----------------------------
user_item = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)
def user_based(user_id, top_n=5):
    sim_users = user_sim_df[user_id].sort_values(ascending=False)[1:6]
    weighted = np.dot(sim_users.values, user_item.loc[sim_users.index])
    recs = pd.Series(weighted, index=user_item.columns)
    watched = user_item.loc[user_id][user_item.loc[user_id] > 0].index
    recs = recs.drop(watched)
    return recs.sort_values(ascending=False).head(top_n)
print("\nUser-based Recommendations:\n", user_based(1))
# -----------------------------
# 7. RMSE
# -----------------------------
pred = np.dot(item_sim, item_user) / np.array([np.abs(item_sim).sum(axis=1)]).T
rmse = np.sqrt(mean_squared_error(item_user.values.flatten(), pred.flatten()))
print("\nRMSE:", rmse)
# -----------------------------
# 8. PRECISION@K
# -----------------------------
def precision_at_k(user_id, k=5):
    recs = recommend_items(user_id, k).index
    relevant = item_user[user_id][item_user[user_id] >= 4].index
    hits = len(set(recs) & set(relevant))
    return hits / k
print("Precision@5:", precision_at_k(1))
# -----------------------------
# 9. VISUALIZATIONS (MATPLOTLIB ONLY)
# -----------------------------
# Heatmap using matplotlib
plt.figure()
plt.imshow(item_sim_df.iloc[:20, :20], aspect='auto')
plt.colorbar()
plt.title("Item Similarity Heatmap")
plt.xlabel("Items")
plt.ylabel("Items")
plt.show()
# Top similar items bar chart
top_items = get_similar_items(1, 5)
plt.figure()
plt.bar(range(len(top_items)), top_items.values)
plt.xticks(range(len(top_items)), top_items.index)
plt.title("Top Similar Items for Item 1")
plt.xlabel("Item ID")
plt.ylabel("Similarity Score")
plt.show()
# Item-based recommendation scores
item_rec = recommend_items(1, 5)
plt.figure()
plt.bar(range(len(item_rec)), item_rec.values)
plt.xticks(range(len(item_rec)), item_rec.index)
plt.title("Item-Based Recommendation Scores")
plt.xlabel("Item ID")
plt.ylabel("Score")
plt.show()
# User-based recommendation scores
user_rec = user_based(1, 5)
plt.figure()
plt.bar(range(len(user_rec)), user_rec.values)
plt.xticks(range(len(user_rec)), user_rec.index)
plt.title("User-Based Recommendation Scores")
plt.xlabel("Item ID")
plt.ylabel("Score")
plt.show()