# Description: Collects data from Reddit using the PRAW library and saves it to a CSV file.
import praw # Import the praw library
import pandas as pd # Import the pandas library
import os   # Import the os library

# Replace with your own credentials
reddit = praw.Reddit(   # Reddit credentials
    client_id='W1BXSyuHUbzWSobsUX2Byg',   # Client ID
    client_secret='yV_ADY8lbdoVUaBo6XZJjL40lyEDQw', # Client Secret
    user_agent='UniversitySentimentAnalysis/0.1 by richmoba'    # User Agent
)

subreddits = ['UNCCharlotte', 'Charlotte']  # Subreddits to search  
query = 'University of Charlotte OR UNC Charlotte OR UNCC OR University City'   # Query to search

def collect_reddit_data():  # Function to collect Reddit data
    try:    # Try block
        posts = []  # List of posts
        for subreddit in subreddits:    # Loop through the subreddits
            subreddit_posts = reddit.subreddit(subreddit).search(query, limit=1000)  # Search the subreddit
            for post in subreddit_posts:    # Loop through the posts
                posts.append([post.title, post.selftext, post.created_utc, post.author.name])   # Append the post
        
        df = pd.DataFrame(posts, columns=['Title', 'Text', 'Timestamp', 'User'])    # Create a DataFrame
        output_dir = os.path.join(os.path.dirname(__file__), '../data/raw')  # Output directory
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory
        df.to_csv(os.path.join(output_dir, 'reddit_posts.csv'), index=False)    # Save the DataFrame to a CSV file
        print("Data collection completed successfully.")    # Print success message
    except Exception as e:  # Except block
        print(f"An error occurred: {e}")    # Print error message

if __name__ == "__main__":  # Main block
    collect_reddit_data()   # Collect Reddit data
