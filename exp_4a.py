# 1. Import Required Libraries
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 3. Text Cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# 4. Convert Text into Numerical Features
# Using built-in English stopwords
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_message'])

# 5. Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])  # ham=0, spam=1

# 6. Split Dataset
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# 7. Train Model
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 10. Misclassified Messages
misclassified = np.where(y_test != y_pred)[0]

print("\nMisclassified Messages:\n")
for i in misclassified[:5]:
    original_index = idx_test[i]
    print("Message:", df.loc[original_index, 'message'])
    print("Actual:", encoder.inverse_transform([y_test[i]])[0])
    print("Predicted:", encoder.inverse_transform([y_pred[i]])[0])
    print("-"*50)

# 11. Laplace Smoothing Change
model_smooth = MultinomialNB(alpha=0.1)
model_smooth.fit(X_train, y_train)
y_pred_smooth = model_smooth.predict(X_test)

print("\nAfter Changing Laplace Smoothing (alpha=0.1)")
print("Accuracy:", accuracy_score(y_test, y_pred_smooth))

# ======================
# VISUALIZATION
# ======================

# Confusion Matrix using matplotlib only
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Feature Importance (Top Spam Words)
feature_names = np.array(vectorizer.get_feature_names_out())
spam_index = 1

top_spam_indices = np.argsort(model.feature_log_prob_[spam_index])[-15:]

print("\nTop Words Influencing Spam Classification:")
for word in feature_names[top_spam_indices]:
    print(word)

# Word Frequency Comparison
spam_words = df[df['label'] == 'spam']['cleaned_message']
ham_words = df[df['label'] == 'ham']['cleaned_message']

spam_vector = vectorizer.transform(spam_words)
ham_vector = vectorizer.transform(ham_words)

spam_freq = np.asarray(spam_vector.sum(axis=0)).flatten()
ham_freq = np.asarray(ham_vector.sum(axis=0)).flatten()

top_indices = np.argsort(spam_freq)[-10:]

plt.figure()
plt.bar(feature_names[top_indices], spam_freq[top_indices], label='Spam')
plt.bar(feature_names[top_indices], ham_freq[top_indices], label='Ham')
plt.xticks(rotation=45)
plt.legend()
plt.title("Word Frequency Comparison")
plt.show()
