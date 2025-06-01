from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import spacy
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS

# BERT için ekler
import torch
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)  # React'ten gelen istekleri kabul etmesi için

# --- Tokenizer Yükleme ---
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --- BERT Tokenizer ve Model Yükleme ---
bert_tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
bert_model.load_state_dict(torch.load("bert.pt", map_location=torch.device('cpu')))
bert_model.eval()

# --- Temizleme Fonksiyonu ---
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

# --- TensorFlow tabanlı modeller için API yaratıcı fonksiyon ---
def createAPI(modelName):
    loadModel = load_model(modelName + ".h5")
    
    def predict_model():
        data = request.get_json()
        raw_text = data.get("text", "")
        processed = clean_text(raw_text)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=300)
        pred = loadModel.predict(padded)[0][0]
        label = "REAL" if pred > 0.5 else "FAKE"
        return jsonify({"result": label, "confidence": float(pred)})
    # Her route'a farklı endpoint ismi veriyoruz
    app.add_url_rule(f"/{modelName}", endpoint=f"predict_{modelName}", view_func=predict_model, methods=["POST"])

# --- BERT için ayrı API ---
@app.route("/bert", methods=["POST"])
def predict_bert():
    data = request.get_json()
    raw_text = data.get("text", "")
    processed = clean_text(raw_text)
    
    inputs = bert_tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=256
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "REAL" if pred == 1 else "FAKE"
    return jsonify({"result": label, "confidence": confidence})

# --- TensorFlow Modelleri İçin Endpoints ---
createAPI("rnn")
createAPI("lstm")
createAPI("gru")

if __name__ == "__main__":
    app.run(debug=True)
