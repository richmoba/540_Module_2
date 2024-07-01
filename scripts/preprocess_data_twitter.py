# preprocess_data.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(tokens)

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/tweets.csv')
    df['Processed_Text'] = df['Text'].apply(preprocess)
    df.to_csv('../data/processed/processed_tweets.csv', index=False)
