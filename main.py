# main.py
import streamlit as st
from scripts.preprocess_data import preprocess
from scripts.classical_model import predict_classical
from scripts.deep_learning_model import predict_deep

st.title("Sentiment Analysis of University Area of Charlotte")

user_input = st.text_area("Enter a social media post about the University area of Charlotte:")

if st.button("Analyze Sentiment"):
    if user_input:
        preprocessed_input = preprocess(user_input)
        classical_sentiment = predict_classical(preprocessed_input)
        deep_sentiment = predict_deep(preprocessed_input)
        st.write(f"Classical Model Sentiment: {classical_sentiment}")
        st.write(f"Deep Learning Model Sentiment: {deep_sentiment}")
    else:
        st.write("Please enter a social media post.")
