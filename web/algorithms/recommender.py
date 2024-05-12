import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from web.algorithms.most_popular_recommender import most_popular_recommender
from web.algorithms.random_recommender import random_rating_recommender
from web.algorithms.similiar_category_combinations import sim_cat_rec
from web.algorithms.tf_idf_recommender import tf_idf_cosine_recommender

class Recommender:
    def __init__(self, row_id, dataset_dir="datasets/MINDsmall_dev", truncate_behaviors=100):
        self._behaviors_path = f"{dataset_dir}/behaviors.tsv"
        self._news_path = f"{dataset_dir}/news.tsv"
        self._entity_embedding_path = f"{dataset_dir}/entity_embedding.vec"
        self._relation_embedding_path = f"{dataset_dir}/relation_embedding.vec"

        news_columns = ['ID', 'Category', 'Subcategory', 'Title', 'Abstract']
        self._news_df = pd.read_csv(self._news_path, sep='\t', names=news_columns, usecols=[0, 1, 2, 3, 4])

        beh_columns = ['ID', 'UserID', 'Time', 'History', 'Impression']
        self._beh_df = pd.read_csv(self._behaviors_path, sep='\t', names=beh_columns, usecols=[0, 1, 2, 3, 4])
        
        row = []

        try:
            row = self._beh_df.loc[int(row_id)]
        except:
            row = self._beh_df.loc[46]
        
        date = row['Time']
        self._userid = row['UserID']

        self._row_unique = self._beh_df.iloc[[row_id]].copy()
        self._row_unique.reset_index(drop=True, inplace=True)

        self._row_notunique = self._beh_df.iloc[[row_id]].copy()
        self._row_notunique.reset_index(drop=True, inplace=True)

        self._ids = self.__get_ids_old_24(date)
        self._unique_ids = list(set(self._ids))

        self._row_unique.at[0, "Impression"] = " ".join(self._unique_ids)
        self._row_notunique.at[0, "Impression"] = " ".join(self._ids)

    def __get_ids_old_24(self, date):
        
        format = '%m/%d/%Y %I:%M:%S %p'
        end_date = datetime.strptime(date, format)
        start_date = end_date - timedelta(hours=24)
        filter_df = self._beh_df[(pd.to_datetime(self._beh_df['Time'], format=format) >= start_date) & (
                    pd.to_datetime(self._beh_df['Time'], format=format) <= end_date)]

        split_ids = filter_df["Impression"].str.split()

        all_article_ids = []

        for ids in split_ids:
            for id in ids:
                if id.endswith("1"):
                    all_article_ids.append(id)

        return all_article_ids

    def __best_score_ids(self, ids, scores_df):
        scores_array = scores_df["Scores"].values[0]

        df = pd.DataFrame({"scores": scores_array, "id": range(len(scores_array))})

        df["ImpressionID"] = [(ids[i]).split("-")[0] for i in df["id"]]
        df = df.drop_duplicates(subset="ImpressionID")
        top_5_impression_ids = (
            df.sort_values(by="scores", ascending=False)
            .head(6)["ImpressionID"]
            .values
        )

        return top_5_impression_ids

    def get_reccomendations(self):
        # random
        random_popular_scores_df = random_rating_recommender(self._row_unique, self._news_df)
        random_popular_best = self.__best_score_ids(self._ids, random_popular_scores_df)

        # Most popular
        most_popular_scores_df = most_popular_recommender(
            self._row_notunique, news_df=[], recommender=True
        )
        most_popular_best = self.__best_score_ids(self._ids, most_popular_scores_df)

        # tf-idf
        tf_idf_scores_df = tf_idf_cosine_recommender(self._row_unique, self._news_df)
        td_idf_best = self.__best_score_ids(self._ids, tf_idf_scores_df)

        # similar-category
        sim_cat_rec_scores_df = sim_cat_rec(self._row_notunique, self._news_df, self._userid, recommender=True)
        sim_cat_best = self.__best_score_ids(self._ids, sim_cat_rec_scores_df)

        result = {
            'random': random_popular_best,
            'most_popular': most_popular_best, 
            'td_idf': td_idf_best,
            'sim_cat': sim_cat_best
        }

        return result, self._userid
