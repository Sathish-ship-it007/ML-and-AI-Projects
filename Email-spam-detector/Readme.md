# 📧 Email Spam Detection Web App

This web app classifies email text as **Spam** or **Not Spam** using a machine learning classifier built with `Multinomial Naive Bayes` and `TF-IDF`.

## 🚀 Features
- Input: Raw email text
- Output: Spam / Not Spam
- Accuracy: 90%+
- Web UI built using Flask

## 🧠 ML Workflow
1. Load email text dataset (ham/spam)
2. Preprocess using:
   - Lowercasing
   - Stopword removal
   - TF-IDF Vectorization
3. Train/test split
4. Model: Multinomial Naive Bayes
5. Evaluate using accuracy, precision, recall

## 📦 Installation

```bash
git clone https://github.com/Sathish-ship-it007/Email-spam-detector
cd email-spam-detector
pip install -r requirements.txt
python train_model.py
python app.py
