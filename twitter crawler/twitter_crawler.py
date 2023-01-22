import twint
import pandas as pd

import os
import sys
import shutil

import logging


# Write a bit that scrapes twits from a given user and return a pandas dataframe


def extract_mentions_and_hashtags_from_twit(twit):
    # Extract mentions and hashtags from twit
    mentions = []
    hashtags = []
    for mention in twit.mentions:
        mentions.append(mention)
    for hashtag in twit.hashtags:
        hashtags.append(hashtag)
    return mentions, hashtags


def extract(screen_name):
    section_contact_tweeter = True
    if section_contact_tweeter:
        c = twint.Config()

        # Get all tweets from a given user from 2018 to 2021
        # Include retweets

        c.Username = screen_name
        c.Store_object = True
        c.Hide_output = True
        c.Limit = 10000
        c.Retweets = True
        c.Since = "2018-01-01"

        c.Pandas = True
        c.json = True
        twint.run.Search(c)
        tweets = twint.output.tweets_list
        df = twint.storage.panda.Tweets_df

    section_parse_tweeter = True
    if section_parse_tweeter:
        # Convert df to list of dicts
        tweets_dict = df.to_dict('records')
        source_name = tweets[0].name
        source_screen_name = tweets[0].username
        source_id = tweets[0].user_id

        mentions_df = pd.DataFrame(columns=[
            'tweet_id', 'text', 'time',
            'source screen_name', 'source name', 'source id',
            'target screen_name', 'target name', 'target id',
            'hashtags', 'likes', 'retweets', 'replies', 'link', 'retweet',
        ],
        )
        i = 0
        for idx, t in enumerate(tweets):
            mentions, hashtags = extract_mentions_and_hashtags_from_twit(t)
            for mention in mentions:
                mentions_df.loc[i, ['target screen_name', 'target name', 'target id']] = [mention['screen_name'],
                                                                                          mention['name'],
                                                                                          mention['id']]
                mentions_df.loc[i, ['source screen_name', 'source name', 'source id']] = [source_screen_name,
                                                                                          source_name,
                                                                                          source_id]
                mentions_df.loc[i, ['tweet_id', 'text']] = [t.id, t.tweet]
                mentions_df.loc[i, ['time']] = t.datetime
                mentions_df.loc[i, ['hashtags']] = str(hashtags)
                mentions_df.loc[i, ['likes', 'retweets', 'replies']] = [t.likes_count, t.retweets_count,
                                                                        t.replies_count]
                mentions_df.loc[i, ['link']] = t.link
                mentions_df.loc[i, ['retweet']] = t.retweet

                i += 1
            tweets_dict[idx]['mentions'] = mentions
            tweets_dict[idx]['hashtags'] = hashtags

    return df, mentions_df


if __name__ == '__main__':
    src_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(src_path, 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    persona = 'netanyahu'
    tweets_df, mentions_df = extract(persona)
    mentions_df_path = os.path.join(data_path, f'{persona}_mentions.csv')
    tweets_path = os.path.join(data_path, f'{persona}_tweets.csv')
    # Save mentions_df to csv with unicode encoding
    mentions_df.to_csv(mentions_df_path, encoding='utf-8-sig', index=False)
    # Save tweets_df to csv with unicode encoding
    tweets_df.to_csv(tweets_path, encoding='utf-8-sig', index=False)
    print(f'CSV of {persona} [items: {len(tweets_df)}]saved to {tweets_path}')
    print(f'CSV of {persona} mentions [items: {len(mentions_df)}]saved to {mentions_df_path}')
