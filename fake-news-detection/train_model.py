import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Load and label the datasets
df_fake = pd.read_csv('dataset/Fake.csv')
df_true = pd.read_csv('dataset/True.csv')

df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0)
df = df[['text', 'label']]  # Keep only necessary columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
