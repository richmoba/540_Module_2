import pandas as pd # Import the pandas library
import nltk # Import the nltk library
from nltk.corpus import stopwords   # Import the stopwords
from nltk.tokenize import word_tokenize  # Import the word_tokenize
from nltk.stem import WordNetLemmatizer # Import the WordNetLemmatizer
import string   # Import the string library
import os   # Import the os library
import random   # Import the random library

nltk.download('punkt')  # Download the punkt package
nltk.download('stopwords')  # Download the stopwords package
nltk.download('wordnet')    # Download the wordnet package

def preprocess(text):   # Function to preprocess text
    if pd.isna(text):   # If text is missing
        return ''   # Return empty string
    # Convert to lowercase  
    text = text.lower() # Convert to lowercase
    # Tokenize
    tokens = word_tokenize(text)    # Tokenize
    # Remove punctuation and stopwords  
    tokens = [word for word in tokens if word.isalpha()]    # Remove punctuation
    stop_words = set(stopwords.words('english'))    # Set of stopwords
    tokens = [word for word in tokens if word not in stop_words]    # Remove stopwords
    # Lemmatize
    lemmatizer = WordNetLemmatizer()    # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]    # Lemmatize
    return ' '.join(tokens) # Join tokens

def main(): # Main function
    # Define paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Script directory
    input_file = os.path.join(script_dir, '../data/raw/reddit_posts.csv')           # Input file
    output_file = os.path.join(script_dir, '../data/processed/processed_reddit_posts.csv')      # Output file

    if not os.path.exists(input_file):  # If input file does not exist
        raise FileNotFoundError(f"{input_file} not found.") # Raise an error

    df = pd.read_csv(input_file)    # Read the input file

    # Combine Title and Text fields for processing
    df['Combined_Text'] = df['Title'].fillna('') + ' ' + df['Text'].fillna('')  # Combine Title and Text fields

    # Add random Sentiment column for demonstration
    sentiments = ['positive', 'neutral', 'negative']    # Sentiments
    df['Sentiment'] = [random.choice(sentiments) for _ in range(len(df))]   # Add random Sentiment column

    # Debug: Print the first few rows of the input data
    print("First few rows of the input data with Sentiment:")   # Print message
    print(df.head())    # Print first few rows

    df['Processed_Text'] = df['Combined_Text'].apply(preprocess)    # Apply preprocessing

    os.makedirs(os.path.dirname(output_file), exist_ok=True)    # Create the output directory
    df.to_csv(output_file, index=False) # Save the processed data to a CSV file

    print("Data preprocessing completed successfully.")   # Print success message
    print("Columns in the processed data:", df.columns)  # Print columns in the processed data

if __name__ == "__main__":  # Main block
    main()  # Run the main function
