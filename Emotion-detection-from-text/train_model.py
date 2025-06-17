import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and Labels
X = df["text"]
y = df["emotion"]

# Text Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
