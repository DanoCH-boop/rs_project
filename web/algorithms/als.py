import numpy as np
import implicit
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from web.algorithms.evaluator import Evaluator
def load_data(truncate_behaviors=10):
    dataset_dir = "../../datasets/MINDsmall_dev"

    behavior_path = f"{dataset_dir}/behaviors.tsv"
    beh_columns = ['ID', 'UserID', 'Time', 'History', 'Impression']
    behaviors = pd.read_csv(behavior_path, sep='\t', names=beh_columns, usecols=[0, 1, 2, 3, 4]).head(
        truncate_behaviors)

    news_path = f"{dataset_dir}/news.tsv"
    news_columns = ['ID', 'Category', 'Subcategory', 'Title', 'Abstract']
    news = pd.read_csv(news_path, sep='\t', names=news_columns, usecols=[0, 1, 2, 3, 4])

    return behaviors, news

def split_impression(row):
    impressions = row['Impression'].split()
    return [impression.split('-')[0].lower() for impression in impressions]

def _row_als(row, interaction_matrix, user_index_dict, news_index_dict, model, proportion_of_seen):
    row_impressions = split_impression(row)
    user_id = user_index_dict[row['UserID']]

    news_in = [news_index_dict[impression] for impression in row_impressions if impression in news_index_dict]

    proportion_of_seen.append(len(news_in) / len(row_impressions))

    if len(news_in) == 0:
        return [0] * len(row_impressions)

    itemids, scores = model.recommend(user_id, interaction_matrix[user_id], items=news_in, N=len(news_in))

    result = []
    for index,impression in enumerate(row_impressions):
        if impression in news_index_dict:
            result.append(scores[np.where(itemids == news_index_dict[impression])[0]][0])
        else:
            result.append(0)

    assert len(result) == len(row_impressions)
    return result
def alternating_least_square(beh_df, news_df):
    # Change NaN values to empty strings
    beh_df['History'] = beh_df['History'].fillna('')

    # Convert History to a user-item interaction matrix
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    interaction_matrix = vectorizer.fit_transform(beh_df['History'])

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    alpha_val = 40
    data_conf = (interaction_matrix * alpha_val).astype('double')
    model.fit(data_conf)

    feature_names = vectorizer.get_feature_names_out()
    news_index_dict = {news_id: index for index, news_id in enumerate(feature_names)}

    user_ids = beh_df['UserID'].unique()
    user_index_dict = {user_id: index for index, user_id in enumerate(user_ids)}

    proportion_of_seen = []
    beh_df['Scores'] = beh_df.apply(_row_als, args=(interaction_matrix, user_index_dict, news_index_dict, model, proportion_of_seen), axis=1)

    print(f"Proportion of seen articles: {np.mean(proportion_of_seen)}")
    return beh_df[['ID', 'Scores']]


if __name__ == '__main__':
    evaluator = Evaluator("../../datasets/MINDsmall_dev", truncate_behaviors=10000)
    recommenders = {
        "alternating least square": alternating_least_square,
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]

    print(evals)

    #
