import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_classical_model():
    df = pd.read_csv('../data/processed/processed_reddit_posts.csv')
    X = df['Processed_Text'].fillna('no_content')
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    os.makedirs('../models', exist_ok=True)
    with open('../models/classical_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

def predict_classical(text):
    with open('../models/classical_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

if __name__ == "__main__":
    train_classical_model()

    print(predict_classical("I love this classical model!"))
