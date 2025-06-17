from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/loan_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    income = float(request.form['income'])
    education = request.form['education']
    credit_score = float(request.form['credit_score'])

    # Encode education
    education_encoded = 1 if education.lower() == 'graduate' else 0

    data = np.array([[age, income, education_encoded, credit_score]])
    prediction = model.predict(data)[0]

    result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'
    return render_template('index.html', prediction_text=f'Loan Status: {result}')

if __name__ == "__main__":
    app.run(debug=True)
