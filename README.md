This project is a machine learning + Flask web app that classifies messages (SMS or Email) as Spam or Ham (Not Spam).

🔹 Features

Web-based interface (HTML + CSS + Flask)

User can type or paste any message

Model predicts whether it is Spam or Ham

Shows prediction with confidence score

Modern responsive UI with animations

🔹 How It Works

A machine learning model (trained on SMS/Email dataset) processes the input text.

The model converts the text into numeric features (using techniques like TF-IDF/CountVectorizer).

The classifier (e.g., Naive Bayes, Logistic Regression, etc.) predicts Spam or Ham.

Flask shows the result on the webpage with colored highlight:

Red → Spam

Green → Ham
