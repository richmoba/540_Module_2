 # Richmond Bakers Sentiment Analysis of Social Media Posts about the University Area of Charlotte

## Project Description
This project analyzes the sentiment of social media posts related to the University area of Charlotte. It uses both classical machine learning and deep learning approaches to classify posts as positive, negative, or neutral.

## Setup and Run
1. Install dependencies: `pip install -r requirements.txt`
2. Collect data: `python scripts/collect_data.py`
   setup a twitter developer account
      PROJECT APP
      1807466513722404864baker_richm
      
3. Preprocess data: `python scripts/preprocess_data.py`
4. Train models:
   - Classical ML: `python scripts/classical_model.py`
   - Deep Learning: `python scripts/deep_learning_model.py`
5. Run the Streamlit app: `streamlit run main.py`

## Project Structure
- `scripts/`: Contains scripts for data collection, preprocessing, and modeling.
- `models/`: Directory for trained models.
- `data/`: Directory for project data.
- `notebooks/`: Directory for exploratory notebooks.

Explanation
Data Collection: The collect_data.py script uses Reddits API's to collect data about the University area of Charlotte.
Data Preprocessing: The preprocess_data.py script preprocesses the text data, including tokenization, stop word removal, and lemmatization.
Classical Machine Learning Model: The classical_model.py script trains a Logistic Regression model and saves it for future predictions.
Deep Learning Model: The deep_learning_model.py script trains an LSTM model and saves it for future predictions.
User Interface: The main.py script uses Streamlit to create a web interface where users can input social media posts and get sentiment predictions from both models.