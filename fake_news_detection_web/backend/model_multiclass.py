# %% Kütüphaneler
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# %% Veri Seti Yükleme ve Etiketleme
news = pd.read_excel('liar_dataset.xlsx')
print(news['label'].value_counts())
label_map = {'pants-fire':0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5}
news['label'] = news['label'].map(label_map)

# %% Metin Ön İşlemesi
def preprocess(text):
    text = text.lower() #Küçük harfe dönüştürülmesi
    text = re.sub(r'[^A-Za-z\s]', '', text) #Özel karakterlerin, sayıların silinmesi
    text = re.sub(r'\n|\t', ' ', text) #Yeni satır ve sekme karakterlerinin silinmesi (\n \t)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #Linklerin silinmesi
    text = re.sub(r'<.*?>', '', text) #HTML Taglerinin silinmesi
    text = " ".join(text.split()) #Fazla boşlukların silinmesi
    text = " ".join([word for word in text.split() if len(word) > 2 and word not in stop_words]) #Kısa kelimelerin ve durak kelimelerinin temizlenmesi
    return text

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

print(news.head(10))
news['text'] = news['text'].apply(preprocess).apply(lemmatization)
print(news.head(10))

# %% RNN, LSTM ve GRU Model Hazırlığı
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
from keras.optimizers import Adam

# Tokenizer
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(news["text"])
sequences = tokenizer.texts_to_sequences(news["text"])
word_index = tokenizer.word_index

# Padding
X = pad_sequences(sequences, maxlen=300)
y = news['label']

# Word2Vec Embedding Matrisi
sentences = [text.split() for text in news["text"]]
word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1)
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ortak eğitim fonksiyonu
def train_and_evaluate_model(model, name):
    print(f"\n--- {name} Model Eğitimi Başladı ---")
    model.compile(optimizer= Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=12, batch_size=64, validation_data=(X_test, y_test))
    print(f"\n--- {name} Model Değerlendirmesi ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"{name} Test Loss: {loss:.4f}")
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# %% RNN Modeli
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True))
rnn_model.add(SimpleRNN(64, activation='tanh', dropout=0.1, return_sequences=True))
rnn_model.add(SimpleRNN(32, activation='tanh', dropout=0.1, return_sequences=True))
rnn_model.add(SimpleRNN(16, activation='tanh', dropout=0.1, return_sequences=False))
rnn_model.add(Dense(6, activation="softmax"))
train_and_evaluate_model(rnn_model, "RNN")

# %% LSTM Modeli
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True))
lstm_model.add(LSTM(64, activation='tanh', dropout=0.3, return_sequences=True))
lstm_model.add(LSTM(32, activation='tanh', dropout=0.3, return_sequences=True))
lstm_model.add(LSTM(16, activation='tanh', dropout=0.3, return_sequences=False))
lstm_model.add(Dense(6, activation="softmax"))
train_and_evaluate_model(lstm_model, "LSTM")

# %% GRU Modeli
gru_model = Sequential()
gru_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True))
gru_model.add(GRU(64, activation='tanh', dropout=0.3, return_sequences=True))
gru_model.add(GRU(32, activation='tanh', dropout=0.3, return_sequences=True))
gru_model.add(GRU(16, activation='tanh', dropout=0.3, return_sequences=False))
gru_model.add(Dense(6, activation="softmax"))
train_and_evaluate_model(gru_model, "GRU")

# %% BERT Modeli
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Hugging Face tokenizer ve model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Modelin anlayacağı şekilde Pytorch Dataset yapısını özelleştirme 
# Metinler, tokenizer ile input_ids ve attention_mask'e çevrilir. Tensor formatına dönüştürülür.
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Veri bölme
x = news['text']
y = news['label']
x_train_text, x_test_text, y_train_label, y_test_label = train_test_split(x, y, test_size=0.2, random_state=0)
train_dataset = NewsDataset(x_train_text.tolist(), y_train_label, bert_tokenizer)
test_dataset = NewsDataset(x_test_text.tolist(), y_test_label, bert_tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Eğitim Fonksiyonu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
optimizer = AdamW(bert_model.parameters(), lr=2e-5, weight_decay=0.01) # weight_decay overfitting önlemek için

def train_bert(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_bert(model, loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, predictions)
    print("Accuracy:", acc)
    print(classification_report(true_labels, predictions))
    return acc

# Model Eğitimi
best_accuracy = 0
for epoch in range(5):
    print(f"\nEpoch {epoch+1}")
    train_loss = train_bert(bert_model, train_loader)
    print(f"Training Loss: {train_loss:.4f}")
    acc = evaluate_bert(bert_model, test_loader)
    
    # En iyi modelin kaydedilmesi
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(bert_model.state_dict(), "bert.pt")
        print("Yeni en iyi model kaydedildi.")
'''
# %% Tokenizer'ı ve modelleri kaydetme
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
rnn_model.save("rnn_model.h5")
lstm_model.save("lstm_model.h5")
gru_model.save("gru_model.h5")

bert_tokenizer.save_pretrained("bert_tokenizer")
'''