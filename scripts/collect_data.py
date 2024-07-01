import praw
import pandas as pd
import os

# Replace with your own credentials
reddit = praw.Reddit(
    client_id='W1BXSyuHUbzWSobsUX2Byg',
    client_secret='yV_ADY8lbdoVUaBo6XZJjL40lyEDQw',
    user_agent='UniversitySentimentAnalysis/0.1 by richmoba'
)

subreddits = ['UNCCharlotte', 'Charlotte']
query = 'University of Charlotte OR UNC Charlotte OR UNCC OR University City'

def collect_reddit_data():
    posts = []
    for subreddit in subreddits:
        subreddit_posts = reddit.subreddit(subreddit).search(query, limit=100)
        for post in subreddit_posts:
            posts.append([post.title, post.selftext, post.created_utc, post.author.name])
    
    df = pd.DataFrame(posts, columns=['Title', 'Text', 'Timestamp', 'User'])
    os.makedirs('../Module_2/data/raw', exist_ok=True)
    df.to_csv('../Module_2/data/raw/reddit_posts.csv', index=False)
    print("Data collection completed successfully.")

if __name__ == "__main__":
    collect_reddit_data()
# Compare this snippet from collect_data_twitter.py:
