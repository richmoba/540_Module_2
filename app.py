import os
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check and load classical model
classical_model_path = os.path.join(current_dir, 'models/classical_model.pkl')
if not os.path.exists(classical_model_path):
    st.error(f"File not found: {classical_model_path}")
else:
    with open(classical_model_path, 'rb') as f:
        classical_model = pickle.load(f)

# Check and load vectorizer
vectorizer_path = os.path.join(current_dir, 'models/vectorizer.pkl')
if not os.path.exists(vectorizer_path):
    st.error(f"File not found: {vectorizer_path}")
else:
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

# Check and load deep learning model
deep_learning_model_path = os.path.join(current_dir, 'models/deep_learning_model.h5')
if not os.path.exists(deep_learning_model_path):
    st.error(f"File not found: {deep_learning_model_path}")
else:
    deep_learning_model = tf.keras.models.load_model(deep_learning_model_path)

# Check and load tokenizer
tokenizer_path = os.path.join(current_dir, 'models/tokenizer.pkl')
if not os.path.exists(tokenizer_path):
    st.error(f"File not found: {tokenizer_path}")
else:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

# Define prediction functions
def predict_classical(text):
    text_vec = vectorizer.transform([text])
    prediction = classical_model.predict(text_vec)
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Handle both numeric and string predictions
    if prediction[0] in sentiment_mapping:
        return sentiment_mapping[prediction[0]]
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
st.title("Sentiment Analysis of Charlotte's University City")
text = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if text:
        classical_sentiment = predict_classical(text)
        deep_sentiment = predict_deep(text)
        
        st.write(f"Classical Model Prediction: {classical_sentiment}")
        st.write(f"Deep Learning Model Prediction: {deep_sentiment}")
    else:
        st.write("Please enter some text to analyze.")

