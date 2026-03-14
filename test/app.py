import sys
import os

# === Fix Python path so src can be found ===
# BASE_DIR points to the parent folder where src and templates live
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from flask import Flask, render_template, request, jsonify
from sqlalchemy import text
from src.inference import KhmerNgramPredictor
from src.database import SessionLocal
from src.models import PredictionLog, User  # combined import

# === Templates and model paths ===
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "model")
predictor = KhmerNgramPredictor(MODEL_DIR)
app = Flask(__name__, template_folder=TEMPLATE_DIR)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.json["text"]

    predictions = predictor.predict(text)

    db = SessionLocal()

    for item in predictions:

        log = PredictionLog(
            input_text=text,
            predicted_word=item["word"],
            score=item["score"]
        )

        db.add(log)

    db.commit()
    db.close()

    return jsonify(predictions)

@app.route("/login", methods=["POST"])
def login():

    username = request.json["username"]
    password = request.json["password"]

    db = SessionLocal()

    user = db.query(User).filter(User.username == username).first()

    if not user:

        user = User(username=username, password=password)
        db.add(user)

    db.execute(
        text("UPDATE users SET created_at = NOW() WHERE username = :username"),
        {"username": username}
    )

    db.commit()
    db.close()

    return jsonify({"message": "login recorded"})

if __name__ == "__main__":
    app.run(debug=True)