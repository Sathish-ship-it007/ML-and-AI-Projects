import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')

# Handle missing values (simple example)
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Loan_Status'] = df['Loan_Status'].map({'Yes': 1, 'No': 0})

# Features and Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model (Random Forest used here)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("ROC AUC Score:", roc_auc_score(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

# Save model
joblib.dump(model, 'model/loan_model.pkl')
