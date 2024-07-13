import pandas as pd     # Importing the necessary libraries
import os   # Importing the necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments   # Importing the necessary libraries
from datasets import Dataset, load_metric   # Importing the necessary libraries
import numpy as np  # Importing the necessary libraries
from sklearn.utils import resample  # Importing the necessary libraries
from sklearn.metrics import classification_report   # Importing the necessary libraries

def preprocess_data():      # Function to preprocess data
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_reddit_posts.csv')   # Input file path
    if not os.path.exists(input_file):  # Check if the input file exists
        raise FileNotFoundError(f"{input_file} not found.")  # Raise an error
    df = pd.read_csv(input_file)    # Read the input file
    return df   # Return the dataframe

def balance_data(df):   # Function to balance the data
    df_positive = df[df['Sentiment'] == 'positive']  # Get positive sentiment
    df_neutral = df[df['Sentiment'] == 'neutral']   # Get neutral sentiment
    df_negative = df[df['Sentiment'] == 'negative'] # Get negative sentiment
    
    n_samples = min(len(df_positive), len(df_neutral), len(df_negative))    # Get the minimum number of samples
    
    df_positive_downsampled = resample(df_positive, replace=False, n_samples=n_samples, random_state=42)    # Downsample positive sentiment
    df_neutral_downsampled = resample(df_neutral, replace=False, n_samples=n_samples, random_state=42)  # Downsample neutral sentiment
    df_negative_downsampled = resample(df_negative, replace=False, n_samples=n_samples, random_state=42)    # Downsample negative sentiment
    
    df_balanced = pd.concat([df_positive_downsampled, df_neutral_downsampled, df_negative_downsampled])   # Concatenate the dataframes
    return df_balanced  # Return the balanced dataframe

def tokenize_function(examples):    # Function to tokenize the data
    return tokenizer(examples['Processed_Text'], truncation=True, padding=True)  # Tokenize the data

def compute_metrics(pred):  # Function to compute metrics
    metric = load_metric("accuracy", trust_remote_code=True)    # Load the accuracy metric
    logits, labels = pred   # Get the logits and labels
    predictions = np.argmax(logits, axis=-1)    # Get the predictions
    accuracy = metric.compute(predictions=predictions, references=labels)   # Compute the accuracy
    return accuracy   # Return the accuracy

def train_model():  # Function to train the model
    df = preprocess_data()  # Preprocess the data
    df = balance_data(df)   # Balance the data

    global tokenizer    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Initialize the tokenizer
    
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}    # Sentiment mapping
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)    # Map the sentiment

    dataset = Dataset.from_pandas(df[['Processed_Text', 'Sentiment']])  # Create a dataset
    train_test_split = dataset.train_test_split(test_size=0.2)  # Split the dataset
    train_dataset = train_test_split['train']   # Get the training dataset
    val_dataset = train_test_split['test']  # Get the validation dataset        

    train_dataset = train_dataset.map(tokenize_function, batched=True)  # Tokenize the training dataset                                                      
    val_dataset = val_dataset.map(tokenize_function, batched=True)  # Tokenize the validation dataset                         

    train_dataset = train_dataset.rename_column("Sentiment", "labels")  # Rename the column                        
    val_dataset = val_dataset.rename_column("Sentiment", "labels")  # Rename the column             
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])   # Set the format       
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])  # Set the format

    training_args = TrainingArguments(  # Training arguments      
        output_dir='./results',                                              
        num_train_epochs=3,                   
        per_device_train_batch_size=16,          
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,        
    )
    
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)    # Initialize the model

    trainer = Trainer(  # Initialize the trainer              
        model=model,                              
        args=training_args,                  
        train_dataset=train_dataset,            
        eval_dataset=val_dataset,           
        compute_metrics=compute_metrics,                            
    )       

    trainer.train() # Train the model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'deep_learning_model')    # Model save path
    trainer.save_model(model_save_path) # Save the model
    tokenizer.save_pretrained(os.path.join(os.path.dirname(__file__), '..', 'models'))  # Save the tokenizer

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate(eval_dataset=val_dataset)   # Evaluate the model
    print(f"Validation Accuracy: {eval_results['eval_accuracy']}")  # Print the validation accuracy
    
    # Generate classification report
    predictions = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)  # Get the predictions
    labels = val_dataset['labels']  # Get the labels      
    report = classification_report(labels, predictions, target_names=['negative', 'neutral', 'positive'])   # Generate the classification report
    print("Classification Report:\n", report)   # Print the classification report

if __name__ == "__main__":  # Main function
    train_model()   # Train the model
