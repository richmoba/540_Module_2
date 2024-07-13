import streamlit as st  # Streamlit
import os   # os module
import pickle   # pickle module
import numpy as np  # numpy module
from tensorflow.keras.models import load_model  # Load model function
from tensorflow.keras.preprocessing.sequence import pad_sequences   # Pad sequences function
import tensorflow as tf # tensorflow module

# Set current directory
current_dir = os.path.dirname(os.path.abspath(__file__))    # Current directory

# Load classical model
def load_classical_model():     # Function to load classical model 
    with open(os.path.join(current_dir, 'models/classical_model.pkl'), 'rb') as f:  # Open the model file
        return pickle.load(f)   # Load the model

# Load vectorizer
def load_vectorizer():  # Function to load vectorizer
    with open(os.path.join(current_dir, 'models/vectorizer.pkl'), 'rb') as f:   # Open the vectorizer file
        return pickle.load(f)   # Load the vectorizer

# Load deep learning model
def load_deep_learning_model():     # Function to load deep learning model
    model = load_model(os.path.join(current_dir, 'models/deep_learning_model.h5'))  # Load the deep learning model
    try:    # Try block
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])       # Compile the model
    except Exception as e:        # Except block
        st.error(f"Error compiling the deep learning model: {e}")   # Print error message
        raise e   # Raise exception
    return model    # Return the model

# Load tokenizer
def load_tokenizer():   # Function to load tokenizer
    with open(os.path.join(current_dir, 'models/tokenizer.pkl'), 'rb') as f:    # Open the tokenizer file
        return pickle.load(f)   # Load the tokenizer

# Define prediction functions
def predict_classical(text):    # Function to predict using classical model
    vectorizer = load_vectorizer()  # Load vectorizer
    classical_model = load_classical_model()    # Load classical model
    text_vec = vectorizer.transform([text]) # Transform text
    prediction = classical_model.predict(text_vec)  # Predict
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}    # Sentiment mapping
    
    if isinstance(prediction[0], int):  # If prediction is an integer
        return sentiment_mapping[prediction[0]]  # Return sentiment mapping
    else:   # Else
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}  # Reverse mapping
        return reverse_mapping[prediction[0]]   # Return reverse mapping

def predict_deep(text):    # Function to predict using deep learning model
    tokenizer = load_tokenizer()    # Load tokenizer
    deep_learning_model = load_deep_learning_model()    # Load deep learning model
    text_seq = tokenizer.texts_to_sequences([text])   # Convert text to sequences
    text_pad = pad_sequences(text_seq, maxlen=100)  # Pad sequences
    prediction = deep_learning_model.predict(text_pad)  # Predict
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}    # Sentiment mapping
    return sentiment_mapping[np.argmax(prediction)]  # Return sentiment mapping

# Streamlit app
st.title("Sentiment Analysis of Charlotte's University City")   # Title
text = st.text_area("Enter text to analyze")    # Text area

if st.button("Analyze"):    # If analyze button is clicked
    if text:    # If text is not empty
        try:    # Try block
            classical_sentiment = predict_classical(text)   # Predict using classical model
            deep_sentiment = predict_deep(text)  # Predict using deep learning model
            st.write(f"Classical Model Prediction: {classical_sentiment}")  # Print classical model prediction
            st.write(f"Deep Learning Model Prediction: {deep_sentiment}")   # Print deep learning model prediction
        except Exception as e:  # Except block
            st.error(f"An error occurred during analysis: {e}")  # Print error message
    else:   # Else
        st.write("Please enter some text to analyze.")  # Print message

