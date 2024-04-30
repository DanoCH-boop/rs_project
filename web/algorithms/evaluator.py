#!/usr/bin/env python

# Dominik Adam
# MUNI FI, Brno
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class Evaluator:
    def __init__(self):
        dataset_dir = "datasets/MINDsmall_dev"
        self._behaviors_path = f"{dataset_dir}/behaviors.tsv"
        self._news_path = f"{dataset_dir}/news.tsv"
        self._entity_embedding_path = f"{dataset_dir}/entity_embedding.vec"
        self._relation_embedding_path = f"{dataset_dir}/relation_embedding.vec"

        news_columns = ['ID', 'Category', 'Subcategory', 'Title', 'Abstract']
        # TODO Remove head for evaluation from the line under, potentially optimize load time of evaluation page
        self._news = pd.read_csv(self._news_path, sep='\t', names=news_columns, usecols=[0, 1, 2, 3, 4])

        beh_columns = ['ID', 'UserID', 'Time', 'History', 'Impression']
        # TODO Remove head for evaluation from the line under, potentially optimize load time of evaluation page
        self._behaviors = pd.read_csv(self._behaviors_path, sep='\t', names=beh_columns, usecols=[0, 1, 2, 3, 4]).head(100)

    def __dcg_score(self, y_true, y_score, k=10):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __ndcg_score(self, y_true, y_score, k=10):
        best = self.__dcg_score(y_true, y_true, k)
        actual = self.__dcg_score(y_true, y_score, k)
        return actual / best

    def __mrr_score(self, y_true, y_score):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true)
    
    def __get_labels(self):
        self._behaviors['Labels'] = self._behaviors['Impression'].str.split().apply(lambda impressions: [impression[-1] for impression in impressions])
        return self._behaviors[['ID', 'Labels']].copy()
    
    def __get_eval_scores(self, preds):
        preds = preds.copy()
        preds['Ranks'] = preds['Scores'].apply(lambda scores: (np.argsort(np.argsort(scores)[::-1]) + 1).tolist())
        preds['EvalScores'] = preds['Ranks'].apply(lambda ranks: [round(1./rank, 5) for rank in ranks])
        return preds[['ID', 'EvalScores']]

    def evaluate(self, recommender):
        aucs = []
        mrrs = []
        ndcg5s = []
        ndcg10s = []

        labels_df = self.__get_labels()

        predictions = recommender(self._behaviors, self._news)
        eval_scores_df = self.__get_eval_scores(predictions)

        line_index = 1
        for (impression_id, labels), (prediction_id, eval_scores) in zip(labels_df.itertuples(index=False), eval_scores_df.itertuples(index=False)):

            if labels == []:
                continue

            if impression_id != prediction_id:
                raise ValueError("line-{}: Inconsistent Impression Id {} and {}".format(
                    line_index,
                    impression_id,
                    prediction_id
                ))

            y_true = np.array(labels, dtype='float32')
            y_score = np.array(eval_scores, dtype='float32')

            if len(y_true) != len(y_score):
                raise ValueError("line-{}: Inconsistent Impressions lengths {} and {}".format(
                    impression_id,
                    len(y_true),
                    len(y_score)
                ))

            auc = roc_auc_score(y_true, y_score)
            mrr = self.__mrr_score(y_true, y_score)
            ndcg5 = self.__ndcg_score(y_true, y_score, 5)
            ndcg10 = self.__ndcg_score(y_true, y_score, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            line_index += 1

        return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
