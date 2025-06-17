# app.py

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)
        return render_template('index.html', prediction_text=f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)
