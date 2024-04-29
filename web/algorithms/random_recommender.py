# Dominik Adam
# MUNI FI, Brno

import random

def random_rating_recommender(beh_df, news_df):
    beh_df['NewsOptions'] = beh_df['Impression'].str.split().apply(lambda impressions: ' '.join([impression[:-2] for impression in impressions]))
    beh_df['Scores'] = beh_df['NewsOptions'].str.split().apply(lambda impressions: [round(random.random(), 5) for _ in impressions])
    return beh_df[['ID', 'Scores']]
