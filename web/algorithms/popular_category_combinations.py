from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from web.algorithms.evaluator import Evaluator
from web.algorithms.most_popular_recommender import most_popular_recommender


def _create_cooccurence_matrix(beh_df, news_df):
    # remove duplicate history entries
    beh_df = beh_df.drop_duplicates(subset=['UserID', 'History'])

    # show user ids that are more than once in the dataset
    assert beh_df['UserID'].value_counts()[beh_df['UserID'].value_counts() > 1].empty

    history_expanded = beh_df.set_index(['ID', 'UserID', 'Time'])['History'].str.split(
        ' ').explode().reset_index()

    # Step 2: Merge history with news_df
    # Ensure that IDs are matched as strings
    history_expanded = history_expanded.merge(news_df, left_on='History', right_on='ID', how='left')

    # Create the pivot table
    user_category_matrix = history_expanded.pivot_table(index='UserID', columns='Subcategory', aggfunc='size',
                                                        fill_value=0)

    # Calculate co-occurrence matrix
    category_cooccurrence = np.dot(user_category_matrix.T, user_category_matrix)

    # We don't consider the diagonal because it represents self-co-occurrence
    np.fill_diagonal(category_cooccurrence, 0)

    category_cooccurrence_df = pd.DataFrame(category_cooccurrence, index=user_category_matrix.columns,
                                            columns=user_category_matrix.columns)

    # Assuming 'category_cooccurrence_df' is your DataFrame
    # Sum across rows to get the total counts for each category
    total_likes_per_category = category_cooccurrence_df.sum(axis=0)

    # Divide each element in a row by the total likes for the row category
    conditional_prob_df = category_cooccurrence_df.div(total_likes_per_category, axis=1)

    return category_cooccurrence_df, user_category_matrix, conditional_prob_df


def _recommend_category(user_likes, category_cooccurrence_df, user_category_matrix):
    # Check if user_likes is a list of categories, and handle it accordingly
    if not isinstance(user_likes, list):
        user_likes = [user_likes]

    # Initialize a series to hold the sum of co-occurrences
    recommendation_scores = pd.Series(data=0, index=user_category_matrix.columns)

    # Sum co-occurrence scores for each liked category
    for category in user_likes:
        recommendation_scores += category_cooccurrence_df[category]

    # Exclude already liked categories from recommendations
    recommendation_scores[user_likes] = 0

    # Return the category with the highest score
    if recommendation_scores.max() > 0:  # Check if there is any recommendation score greater than zero
        return recommendation_scores.idxmax()
    else:
        return "No recommendation available based on current preferences"


def _calculate_popularity_of_news(beh_df, row):
    def find_index(series, name):
        try:
            index = series.index.get_loc(name) + 1  # Adding 1 to start index from 1
            return index
        except KeyError:
            return 0

    date = row['Time']
    format = '%m/%d/%Y %I:%M:%S %p'
    end_date = datetime.strptime(date, format)
    start_date = end_date - timedelta(hours=24)

    filter_df = beh_df[(pd.to_datetime(beh_df['Time'], format=format) >= start_date) & (
            pd.to_datetime(beh_df['Time'], format=format) < end_date)]
    split_ids = filter_df['Impression'].str.split()

    all_article_ids = []

    for ids in split_ids:
        all_article_ids.extend([id.split("-")[0] for id in ids if id.endswith('-1')])

    all_article_ids_series = pd.Series(all_article_ids)

    popular_clicked_articles = all_article_ids_series.value_counts()

    sorted_popular = popular_clicked_articles.sort_values(ascending=False)

    num_rows = sorted_popular.shape[0]

    impression_ids = row['Impression'].split()

    split_Impression_list = [word for sublist in impression_ids for word in sublist.split()]

    impression_coef_array = {}

    for id in split_Impression_list:
        sliced_id = id.split("-")[0]

        row_index = find_index(sorted_popular, sliced_id)

        if row_index == 0:
            impression_coef_array[sliced_id] = 0
        else:
            impression_coef_array[sliced_id] = (1 - (int(row_index) - 1) / (num_rows ))

    return impression_coef_array


# Function to calculate scores for impressions
def _calculate_impression_scores(row, user_category_matrix, category_cooccurrence_df, news_df, beh_df):
    # Create a mapping of News ID to Subcategory from news_df
    news_id_to_category = news_df.set_index('ID')['Subcategory'].to_dict()

    user_id = row['UserID']
    impressions = row['Impression'].split()

    # Extract the user's preferred categories from the user_category_matrix
    try:
        user_preferences = user_category_matrix.loc[user_id]
    except KeyError:
        return [0] * len(impressions)

    # Initialize scores for each impression
    scores = []
    popularity_of_news_until_now = _calculate_popularity_of_news(beh_df, row)

    for impression in impressions:
        news_id, clicked = impression.split('-')

        category = news_id_to_category.get(news_id, None)

        if category and category in category_cooccurrence_df.columns:
            normalized_cooccurrence = category_cooccurrence_df[category] / category_cooccurrence_df[category].max()
            normalized_preferences = user_preferences / user_preferences.max()
            # Calculate category score based on co-occurrence with user's preferred categories
            score = sum(normalized_preferences * normalized_cooccurrence)

            # Calculate score based on co-occurrence with user's preferred categories
            score = sum(user_preferences * category_cooccurrence_df[category])

            # Calculate popularity score based on the number of times the news was clicked
            popularity_score = popularity_of_news_until_now.get(news_id, 0)

            # Combine the two scores
            score = score + popularity_score
        else:
            score = 0  # Default score if category is unknown

        scores.append(score)

    return scores


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


def category_combinations_recommender(beh_df, news_df):
    beh_full_df, _ = load_data(100)  # because beh_df might contain just one line of data
    category_coccurrence_df, user_category_matrix, conditional_prob_df = _create_cooccurence_matrix(beh_full_df,
                                                                                                    news_df)

    # Apply the function to each row in beh_df to get scores for each impression
    beh_df['Scores'] = beh_df.apply(_calculate_impression_scores,
                                    args=(user_category_matrix, category_coccurrence_df, news_df, beh_df),
                                    axis=1)

    return beh_df[['ID', 'Scores']]


if __name__ == '__main__':
    evaluator = Evaluator("../../datasets/MINDsmall_dev", truncate_behaviors=100)
    recommenders = {
        "Popular category combinations": category_combinations_recommender,
        "popular": most_popular_recommender,
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]

    print(evals)

    #
    print("Recommend best 5 articles")
    beh_df, news_df = load_data()

    beh_df = beh_df.head(1)
    scores = category_combinations_recommender(beh_df, news_df)

    print(scores)
