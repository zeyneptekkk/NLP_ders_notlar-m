#import libraries
import numpy as np
import pandas as pd
from gensim.models import Word2Vec # metin temsili
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#create dataset

data = {
    "text": [
        "Yemek çok güzeldi",
        "Yemek çok pişmişti",
        "Servis harikaydı",
        "Garson çok kaba davrandı",
        "Tatlılar muazzamdı",
        "Çorba tuzsuzdu",
        "Mekan çok şıktı",
        "Koltuklar rahatsızdı",
        "Yemekler tam zamanında geldi",
        "Çok bekledik, sinirlendik",
        "Tat olarak çok iyiydi",
        "Lezzetsizdi, yiyemedim",
        "Garsonlar güler yüzlüydü",
        "Hizmet çok yavaştı",
        "Sunum harikaydı",
        "Yemekler soğuktu",
        "Porsiyonlar doyurucuydu",
        "Çok küçük porsiyonlardı",
        "Fiyatlar makuldü",
        "Aşırı pahalıydı",
        "Mekan çok kalabalıktı",
        "Sessiz ve huzurluydu",
        "Garson siparişi karıştırdı",
        "Tam istediğimiz gibi geldi",
        "Yemekler taze ve sıcaktı",
        "Salata bayattı",
        "Tatlı enfesti",
        "İçecekler ılık geldi",
        "Masamız kirliydi",
        "Tertemiz bir ortam vardı",
        "Müşteriyle çok ilgililerdi",
        "Hiç kimse ilgilenmedi",
        "Et tam kıvamındaydı",
        "Et lastik gibiydi",
        "Tatlılar taze ve lezizdi",
        "Çatal bıçaklar temiz değildi",
        "Servis hızlıydı",
        "Yemek çok tuzluydu",
        "Çok lezzetliydi, tekrar geleceğim",
        "Kötüydü, asla bir daha gelmem",
        "Makarnanın sosu mükemmeldi",
        "Pilav kuru ve tatsızdı",
        "Menü çok çeşitliydi",
        "Hiçbir şey bulamadım yiyecek",
        "Yemekler ev yapımı gibiydi",
        "Hazır yemek gibiydi",
        "Çalışanlar çok anlayışlıydı",
        "İletişim kurmak zordu",
        "Her şey çok özenliydi",
        "İlgi alaka sıfırdı"
    ],
    "label": [
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "negatif", "pozitif", "negatif", "pozitif", "pozitif", "negatif", "pozitif", "negatif", "negatif", "pozitif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif",
        "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif", "pozitif", "negatif"
    ]
}

df = pd.DataFrame(data)

# %% Metin temizleme ve preprocessing: tokenization, padding, label encoding, train-test split

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])

# Padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen)
print(X.shape)

# %% Metin temsili: Word embedding: Word2Vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, min_count=1, window=5)
embedding_dim = 50
word_index = tokenizer.word_index  # This should be used instead of undefined `word_index`

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# %% Modelling: Build RNN model for training and testing

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model
model = tf.keras.Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
