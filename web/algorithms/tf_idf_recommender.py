# Daniel Chudý
# MUNI FI, Brno

import pandas as pd
import numpy as np
import random
from web.algorithms.tf_idf import tf_idf_cosine_rec
from typing import Literal

# Switch mode to "concat" to concatenate all titles into one
def tf_idf_cosine_recommender(
    beh_df,
    news_df,
    mode: Literal["split", "concat"] = "split",
    sim_col: Literal["Title", "Abstract"] = "Title"
):  
    beh_df['Articles_to_evaluate'] = beh_df['Impression'].str.split().apply(lambda impressions: ' '.join([impression.split("-")[0] for impression in impressions]))
    print(mode.upper()) 
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
            sim_col_str = news_df[news_df['ID'] == article][sim_col].values[0]
            concat_df = news_df[news_df['ID'].isin(history)]
            if not mode == "split":
                concat_df = pd.DataFrame({sim_col: [' '.join(concat_df[sim_col])]})
            rating = tf_idf_cosine_rec(concat_df[[sim_col]], category='eval', sim_col_string=sim_col_str, mode="eval", sim_col=sim_col)
            ratings.append(rating)

        all_ratings.append(ratings)

    beh_df['Scores'] = all_ratings
    return beh_df[['ID', 'Scores']]