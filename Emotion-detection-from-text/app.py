from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    return render_template("index.html", message=message, emotion=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
