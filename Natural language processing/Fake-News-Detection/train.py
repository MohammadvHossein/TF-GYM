import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# Load datasets
data_fake = pd.read_csv('datasets/data_fack.csv')
data_true = pd.read_csv('datasets/data_true.csv')

data_fake["label"] = 0
data_true['label'] = 1

data = pd.concat([data_fake, data_true], axis=0)
data = data.drop(['title', 'subject', 'date'], axis=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['text'] = data['text'].apply(preprocess)

x = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(x,  y, test_size=0.25)

vectorization = TfidfVectorizer()
X_train = vectorization.fit_transform(X_train)
X_test = vectorization.transform(X_test)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorization, f)

input_dim = X_train.shape[1]
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

machPoint = ModelCheckpoint('best_model_FackNews.keras' , monitor="val_loss", mode='min' , save_best_only=True)

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test) , callbacks=[machPoint])

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)