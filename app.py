from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("logistic_regression.joblib")

FIELDS = [
    "title","location","department","salary_range","company_profile",
    "description","requirements","benefits","employment_type",
    "required_experience","required_education","industry","function"
]

@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    text = " ".join(str(data.get(f, "") or "") for f in FIELDS)
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][pred])
    return jsonify({
        "label": "FAKE" if pred == 1 else "GENUINE",
        "confidence": round(probability * 100, 2),
        "model": "TF-IDF + Logistic Regression"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
