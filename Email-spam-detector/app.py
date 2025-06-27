from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load model from model folder
MODEL_PATH = os.path.join('model', 'spam_classifier_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction = model.predict([message])[0]
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
