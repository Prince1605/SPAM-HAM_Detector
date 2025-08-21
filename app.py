from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ---- load model + vectorizer ----
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    confidence_text = None

    if request.method == "POST":
        user_sms = request.form.get("message", "").strip()
        if user_sms:
            X = vectorizer.transform([user_sms])
            pred = model.predict(X)[0]
            # handle numeric or string labels robustly
            pred_str = str(pred).lower()
            label = "Spam" if pred_str in ("1", "spam") else "Ham"

            # confidence (max class probability if available)
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba)) * 100
                confidence_text = f"{confidence:.2f}%"
            except Exception:
                confidence_text = None

            prediction_text = f"{label}"
        else:
            prediction_text = "Please enter a message."

    return render_template(
        "index.html",
        prediction=prediction_text,
        confidence=confidence_text
    )

if __name__ == "__main__":
    app.run(debug=True)
