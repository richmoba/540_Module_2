import pandas as pd         # Importing the necessary libraries                      
from sklearn.model_selection import train_test_split       # Importing the necessary libraries        
from sklearn.feature_extraction.text import TfidfVectorizer # Importing the necessary libraries
from sklearn.linear_model import LogisticRegression  # Importing the necessary libraries
from sklearn.metrics import accuracy_score, classification_report # Importing the necessary libraries
import pickle   # Importing the necessary libraries
import os   # Importing the necessary libraries

def train_classical_model():    # Function to train classical model
    df = pd.read_csv('data/processed/processed_reddit_posts.csv')   # Read the processed data
    X = df['Processed_Text'].fillna('no_content')   # Fill missing values
    y = df['Sentiment'] # Get the sentiment column

    # Convert sentiment labels to numeric values
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}    # Sentiment mapping
    y = y.map(sentiment_mapping)    # Map sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # Split the data
    vectorizer = TfidfVectorizer()  # Initialize the vectorizer
    X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform the training data
    X_test_vec = vectorizer.transform(X_test)   # Transform the test data

    model = LogisticRegression()    # Initialize the model
    model.fit(X_train_vec, y_train) # Fit the model
    y_pred = model.predict(X_test_vec)  # Predict

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")    # Print accuracy
    print(classification_report(y_test, y_pred))    # Print classification report

    os.makedirs('../models', exist_ok=True)   # Create the models directory
    with open('../models/classical_model.pkl', 'wb') as f:  # Open the model file
        pickle.dump(model, f)   # Save the model
    with open('../models/vectorizer.pkl', 'wb') as f:   # Open the vectorizer file
        pickle.dump(vectorizer, f)  # Save the vectorizer

def predict_classical(text):    # Function to predict using classical model
    with open('../models/classical_model.pkl', 'rb') as f:  # Open the model file
        model = pickle.load(f)  # Load the model
    with open('../models/vectorizer.pkl', 'rb') as f:   # Open the vectorizer file
        vectorizer = pickle.load(f) # Load the vectorizer

    text_vec = vectorizer.transform([text]) # Transform text
    prediction = model.predict(text_vec)    # Predict
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}    # Sentiment mapping
    return sentiment_mapping[prediction[0]] # Return sentiment mapping

if __name__ == "__main__":  # Main block
    train_classical_model() # Train the classical model

    print(predict_classical("I love UNC Charlotte Classical Model!"))   # Predict using classical model

