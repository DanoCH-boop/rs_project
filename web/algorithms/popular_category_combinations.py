import random

import numpy as np
import pandas as pd
from web.algorithms.evaluator import Evaluator


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


# Function to calculate scores for impressions
def _calculate_impression_scores(row, user_category_matrix, category_cooccurrence_df, news_df):
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
    categories_with_scores = []

    for impression in impressions:
        news_id, clicked = impression.split('-')
        category = news_id_to_category.get(news_id, None)

        if category and category in category_cooccurrence_df.columns:
            # Calculate score based on co-occurrence with user's preferred categories
            score = sum(user_preferences * category_cooccurrence_df[category])
        else:
            score = 0  # Default score if category is unknown

        scores.append(score)
        categories_with_scores.append((category, score))

    return scores


def category_combinations_recommender(beh_df, news_df):
    category_cooccurrence_df, user_category_matrix, conditional_prob_df = _create_cooccurence_matrix(beh_df, news_df)

    # Apply the function to each row in beh_df to get scores for each impression
    beh_df['Scores'] = beh_df.apply(_calculate_impression_scores,
                                               args=(user_category_matrix, category_cooccurrence_df, news_df), axis=1)

    # RANDOM RECOMMENDER TO SHOW SCORE
    beh_df['NewsOptions'] = beh_df['Impression'].str.split().apply(
        lambda impressions: ' '.join([impression[:-2] for impression in impressions]))
    beh_df['Scores_original'] = beh_df['NewsOptions'].str.split().apply(
        lambda impressions: [round(random.random(), 5) for _ in impressions])
    return beh_df[['ID', 'Scores']]


if __name__ == '__main__':
    evaluator = Evaluator("../../datasets/MINDsmall_dev", truncate_behaviors=10000)
    recommenders = {
        "Popular category combinations": category_combinations_recommender,
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]

    print(evals)
