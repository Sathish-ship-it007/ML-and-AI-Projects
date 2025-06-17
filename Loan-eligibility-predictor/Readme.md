# 📄 Loan Eligibility Predictor

A machine learning web application built with Flask to predict whether a loan applicant is eligible for a loan based on details such as age, income, education, and credit score.

---

## 📌 Problem Statement

Banks want to predict whether an applicant should be granted a loan using data like income, credit score, age, and education.

---

## 🎯 Objective

Build a classification model to predict loan approval status and deploy it using Flask.

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Random Forest (used for final prediction)

---

## 🗃️ Dataset Features

| Feature       | Description                         |
|---------------|-------------------------------------|
| Age           | Age of applicant                    |
| Income        | Monthly income of the applicant     |
| Education     | Graduate / Not Graduate             |
| Credit_Score  | Credit score value (300–850)        |
| Loan_Status   | Target variable (Yes / No)          |

📥 [Download Sample Dataset](https://gist.githubusercontent.com/smaranjitghose/1bcb7d3a8572305b8d90698dd20f4496/raw/5c548c8cf9302ea3677f5296892fa4a2f8fc65d7/dataset.csv)

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-eligibility-predictor.git
   cd loan-eligibility-predictor
