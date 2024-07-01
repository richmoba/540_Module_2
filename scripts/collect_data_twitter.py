import tweepy
import pandas as pd

# Replace with your own credentials
consumer_key = 'W3MkFQqjKdVCsmZjQN2fb8GUT'
consumer_secret = 'i7TKsTbqx2gTWddkSbCszqvov9xW9kzw11ngnA0OJ0KvNhcXjE'
access_token = '1470482673961422850-NPQJtwqkTWGb74tnFlBwN2EH4hNS0g'
access_token_secret = 'Bkz5szjf6f2IC2bv6r7G0YcaZ7ohs7pEOuz7RHhNesWBb'

# Authenticate to Twitter
try:
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    api.verify_credentials()
    print("Authentication OK")
except tweepy.TweepyException as e:
    print(f"Error during authentication: {e}")
    exit(1)

# Collect tweets
try:
    tweets = tweepy.Cursor(api.search_tweets,
                           q="University of Charlotte OR UNC Charlotte OR UNCC OR University City",
                           lang="en").items(100)

    tweet_data = [[tweet.text, tweet.created_at, tweet.user.screen_name] for tweet in tweets]

    df = pd.DataFrame(tweet_data, columns=['Text', 'Timestamp', 'User'])
    df.to_csv('../data/raw/tweets.csv', index=False)
    print("Data collection completed successfully.")
except tweepy.TweepyException as e:
    print(f"Error during data collection: {e}")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)
