import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle
import os

def train_deep_learning_model():
    df = pd.read_csv('../Module_2/data/processed/processed_reddit_posts.csv')
    X = df['Processed_Text'].fillna('no_content')
    y = df['Sentiment']

    # Convert sentiment labels to numeric values
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = y.map(sentiment_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    maxlen = 100
    X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=maxlen),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(3, activation='softmax')  # Adjusted output layer for multi-class classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f"Accuracy: {accuracy}")

    os.makedirs('../Module_2/models', exist_ok=True)
    model.save('../Module_2/models/deep_learning_model.h5')
    with open('../Module_2/models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

def predict_deep(text):
    model = tf.keras.models.load_model('../Module_2/models/deep_learning_model.h5')
    with open('../Module_2/models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, padding='post', maxlen=100)
    prediction = model.predict(text_pad)
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_mapping[np.argmax(prediction)]

if __name__ == "__main__":
    train_deep_learning_model()

    print(predict_deep("I love UNC Charlotte Deep Learning!"))