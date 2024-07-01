import streamlit as st
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load classical model
with open(os.path.join(current_dir, 'models/classical_model.pkl'), 'rb') as f:
    classical_model = pickle.load(f)

# Load vectorizer
with open(os.path.join(current_dir, 'models/vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# Load deep learning model
deep_learning_model = load_model(os.path.join(current_dir, 'models/deep_learning_model.h5'))

# Load tokenizer
with open(os.path.join(current_dir, 'models/tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Define prediction functions
def predict_classical(text):
    text_vec = vectorizer.transform([text])
    prediction = classical_model.predict(text_vec)
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Handle both numeric and string predictions
    if prediction[0] in sentiment_mapping:
        return prediction[0]
    else:
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
        return reverse_mapping[prediction[0]]

def predict_deep(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=100)
    prediction = deep_learning_model.predict(text_pad)
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_mapping[np.argmax(prediction)]

# Streamlit app
st.title("Sentiment Analysis of Charlottes University city ")
text = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if text:
        classical_sentiment = predict_classical(text)
        deep_sentiment = predict_deep(text)
        
        st.write(f"Classical Model Prediction: {classical_sentiment}")
        st.write(f"Deep Learning Model Prediction: {deep_sentiment}")
    else:
        st.write("Please enter some text to analyze.")
