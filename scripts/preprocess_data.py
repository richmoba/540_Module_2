import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    if pd.isna(text):
        return ''
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def main():
    input_file = '../Module_2/data/raw/reddit_posts.csv'
    output_file = '../Module_2/data/processed/processed_reddit_posts.csv'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    df = pd.read_csv(input_file)

    # Combine Title and Text fields for processing
    df['Combined_Text'] = df['Title'].fillna('') + ' ' + df['Text'].fillna('')

    # Add random Sentiment column for demonstration
    sentiments = ['positive', 'neutral', 'negative']
    df['Sentiment'] = [random.choice(sentiments) for _ in range(len(df))]

    # Debug: Print the first few rows of the input data
    print("First few rows of the input data with Sentiment:")
    print(df.head())

    df['Processed_Text'] = df['Combined_Text'].apply(preprocess)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print("Data preprocessing completed successfully.")
    print("Columns in the processed data:", df.columns)

if __name__ == "__main__":
    main()

    
 
    print("Data preprocessing completed successfully whoot, whoot!.")