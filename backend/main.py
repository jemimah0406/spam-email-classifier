from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test route
@app.route("/")
def home():
    return "Spam classifier backend is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data["text"]

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    confidence = max(model.predict_proba(vector)[0]) * 100

    return jsonify({
        "prediction": str(prediction),
        "confidence": round(confidence, 2)
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)