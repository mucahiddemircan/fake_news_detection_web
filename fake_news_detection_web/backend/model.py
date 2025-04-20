# %% Kütüphaneler
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import spacy
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm") # lemmatize ingilizce

# %% Veri Seti Yüklemesi
news = pd.read_excel('news.xlsx')
news = news.head(500)

# %% Metin Ön İşlemesi
def word_operations(text):
    text = text.lower() #Küçük harfe dönüştürülmesi
    text = re.sub(r'[^\w\s]', ' ', text) #Noktalama işaretlerinin silinmesi
    text = re.sub(r'\d', '', text) #Sayıların silinmesi
    text = re.sub(r'\n', '', text) #Yeni satır karakterlerinin silinmesi (\n)
    text = re.sub(r'\t', '', text) #Sekme karakterlerinin silinmesi (\t)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text) #Özel karakterlerin silinmesi
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #Linklerin silinmesi
    text = re.sub(r'<.*?>', '', text) #HTML Taglerinin silinmesi
    text = " ".join(text.split()) #Fazla boşlukların silinmesi
    text = " ".join([word for word in text.split() if len(word) > 2]) #Kısa kelimelerin temizlenmesi
    #text = str(TextBlob(text).correct()) #Yazım hatalarının düzeltilmesi
    return text

# Durak kelimelerini kaldıran fonksiyon
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Lemmatization fonksiyonu
def lemmatize_text_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

print(news.head(10))
news['text'] = news['text'].apply(word_operations)
news['text'] = news['text'].apply(remove_stop_words)
news['text'] = news['text'].apply(lemmatize_text_spacy)
print(news.head(10))

# %% Eğitim ve Test Setlerine Ayırma
'''
x = news['text']
y = news['label']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
'''
# %% Öznitelik Çıkarımı
'''
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorization = CountVectorizer(max_features=10000, ngram_range=(1, 2)) #(1,2) bigram (kelime kelime)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Word2Vec
tokenized_sentences = [word_tokenize(text) for text in x_train] # metni kelimelere bölme
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)

# Kelimeleri vektörlere dönüştürme fonksiyonu
def document_vector(doc):
    words = word_tokenize(doc)
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(word_vectors) == 0:
        return np.zeros(100)
    return np.mean(word_vectors, axis=0)

# Eğitim ve test veri setleri için vektörlerin oluşturulması
xv_train_w2v = np.array([document_vector(doc) for doc in x_train])
xv_test_w2v = np.array([document_vector(doc) for doc in x_test])
'''
# %% BERT ile Metin Temsili
'''
from transformers import BertTokenizer, BertModel

# BERT Tokenizer ve Modeli Yükleme
bert_model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# BERT ile Metin Temsili (Embedding) Alma
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token'ı

xv_train = torch.tensor([get_bert_embedding(text) for text in x_train])
xv_test = torch.tensor([get_bert_embedding(text) for text in x_test])

# %% Makine öğrenmesi modelininin seçimi ve uygulanması
# Lojistik Regresyon Sınıflandırması
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=2000) # BERT vektörleri yüksek boyutlu olduğu için iterasyon artırıldı
LR.fit(xv_train, y_train)
prediction_LR = LR.predict(xv_test)

print("Lojistik Regresyon Doğruluğu:", LR.score(xv_test, y_test))
print(classification_report(y_test, prediction_LR))
print("Lojistik Regresyon Hata Matrisi:\n", confusion_matrix(y_test, prediction_LR))
sns.heatmap(confusion_matrix(y_test,prediction_LR),annot = True, cmap = 'coolwarm',fmt = 'd')
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.show()
'''
'''
LR_w2v = LogisticRegression()
LR_w2v.fit(xv_train_w2v, y_train)
prediction_LR_w2v = LR_w2v.predict(xv_test_w2v)

print("Word2Vec Lojistik Regresyon Doğruluğu:", LR_w2v.score(xv_test_w2v, y_test))
print(classification_report(y_test, prediction_LR_w2v))
sns.heatmap(confusion_matrix(y_test, prediction_LR_w2v), annot=True, cmap='coolwarm', fmt='.1f')
plt.show()
'''

# %% RNN, LSTM ve GRU Model Hazırlığı
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding

# Tokenizer
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(news["text"])
sequences = tokenizer.texts_to_sequences(news["text"])
word_index = tokenizer.word_index
print("Vocab size: ", len(word_index))

# Padding
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=300)
print("X shape: ", X.shape)

y = news['label']

# Word2Vec Embedding Matrisi
sentences = [text.split() for text in news["text"]]
word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=2)

embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
# Ortak fonksiyon - Model eğitimi ve değerlendirmesi
def train_and_evaluate_model(model, name):
    print(f"\n--- {name} Model Eğitimi Başladı ---")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=8, batch_size=64, validation_data=(X_test, y_test))
    print(f"\n--- {name} Model Değerlendirmesi ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"{name} Test Loss: {loss:.4f}")
    print(f"{name} Test Accuracy: {accuracy:.4f}")

# %% RNN Modeli
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False))
rnn_model.add(SimpleRNN(64, activation='tanh', return_sequences=True))
rnn_model.add(SimpleRNN(32, activation='tanh', return_sequences=True))
rnn_model.add(SimpleRNN(16, activation='tanh', return_sequences=False))
rnn_model.add(Dense(1, activation="sigmoid"))
train_and_evaluate_model(rnn_model, "RNN")

# %% LSTM Modeli
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False))
lstm_model.add(LSTM(64, activation='tanh', dropout=0.3, return_sequences=True))
lstm_model.add(LSTM(32, activation='tanh', dropout=0.3, return_sequences=True))
lstm_model.add(LSTM(16, activation='tanh', dropout=0.3, return_sequences=False))
lstm_model.add(Dense(1, activation="sigmoid"))
train_and_evaluate_model(lstm_model, "LSTM")

# %% GRU Modeli
gru_model = Sequential()
gru_model.add(Embedding(input_dim=len(word_index)+1, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False))
gru_model.add(GRU(64, activation='tanh', dropout=0.3, return_sequences=True))
gru_model.add(GRU(32, activation='tanh', dropout=0.3, return_sequences=True))
gru_model.add(GRU(16, activation='tanh', dropout=0.3, return_sequences=False))
gru_model.add(Dense(1, activation="sigmoid"))
train_and_evaluate_model(gru_model, "GRU")

# Tokenizer'ı ve modeli kaydet
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
rnn_model.save("rnn_model.h5") 