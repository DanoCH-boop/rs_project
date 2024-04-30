# Daniel Chudý
# MUNI FI, Brno
import pandas as pd
import numpy as np
import random
from web.algorithms.tf_idf import tf_idf_cosine_rec

# Switch mode to "concat" to concatenate all titles into one
def tf_idf_cosine_recommender(beh_df, news_df, mode="split"):  
    beh_df['Articles_to_evaluate'] = beh_df['Impression'].str.split().apply(lambda impressions: ' '.join([impression[:-2] for impression in impressions]))

    df_evaluate = beh_df[['ID', 'History', 'Articles_to_evaluate']]

    all_ratings = []
    for _, line in df_evaluate.iterrows():
        articles = line['Articles_to_evaluate'].split()
        history = [] if isinstance(line['History'], float) and np.isnan(line['History']) else line['History'].split()
        ratings = []

        if len(history) == 0:
            all_ratings.append([round(random.random(), 5) for _ in articles])
            continue

        for article in articles:
            title = news_df[news_df['ID'] == article]['Title'].values[0]
            concat_df = news_df[news_df['ID'].isin(history)]
            if not mode == "split":
                concat_df = pd.DataFrame({'Title': [' '.join(concat_df['Title'])]})
            rating = tf_idf_cosine_rec(concat_df[['Title']], category='eval', title_string=title, mode="eval")
            ratings.append(rating)

        all_ratings.append(ratings)

    beh_df['Scores'] = [[sorted(sublist, reverse=True).index(value) + 1 for value in sublist] for sublist in all_ratings]
    return beh_df[['ID', 'Scores']]