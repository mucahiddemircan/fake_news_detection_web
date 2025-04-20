from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import spacy
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)  # React'ten gelen istekleri kabul etmesi için

model = load_model("rnn_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n|\t', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = " ".join([word for word in text.split() if len(word) > 2])
    text = " ".join([word for word in text.split() if word not in stop_words])
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")
    processed = clean_text(raw_text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=300)
    pred = model.predict(padded)[0][0]
    label = "REAL" if pred > 0.5 else "FAKE"
    return jsonify({"result": label, "confidence": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)
