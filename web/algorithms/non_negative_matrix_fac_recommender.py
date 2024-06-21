import pandas as pd
import numpy as np
from sklearn.decomposition import NMF


# Create user-subcategory interaction matrix
def create_interaction_matrix(beh_df, news_df, news_id_to_subcategory):
    user_ids = beh_df["UserID"].unique()
    subcategories = news_df["Category"].unique()

    history_ids = beh_df["History"].str.split(" ").explode()
    unique_history_ids = history_ids.unique()

    interaction_matrix = pd.DataFrame(0, index=user_ids, columns=subcategories)
    news_id_user_id_matrix = pd.DataFrame(
        0, index=user_ids, columns=unique_history_ids
    )
    for index, row in beh_df.iterrows():
        user_id = row["UserID"]
        if isinstance(row["History"], float) != True:
            for news_id in (row["History"]).split(" "):
                if (
                    news_id in news_id_to_subcategory
                    and news_id_user_id_matrix.at[user_id, news_id] == 0
                ):
                    subcategory = news_id_to_subcategory[news_id]
                    news_id_user_id_matrix.at[user_id, news_id] += 1
                    interaction_matrix.at[user_id, subcategory] += 1

    return interaction_matrix


def matrix_factorization(interaction_matrix, n_components=10):
    model = NMF(n_components=n_components, init="random", random_state=42)
    W = model.fit_transform(interaction_matrix)
    H = model.components_
    return W, H, interaction_matrix.index, interaction_matrix.columns


def get_scores_for_impressions(
    behaviors_data, W, H, user_ids, subcategories, news_id_to_subcategory
):
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    subcategory_index = {
        subcategory: index for index, subcategory in enumerate(subcategories)
    }

    scores = []
    for index, row in behaviors_data.iterrows():
        user_id = row["UserID"]
        impressions = row["Impressions"]
        user_scores = []
        for impression in impressions:
            news_id, _ = impression.split("-")
            if user_id in user_index and news_id in news_id_to_subcategory:
                subcategory = news_id_to_subcategory[news_id]
                if subcategory in subcategory_index:
                    user_idx = user_index[user_id]
                    subcategory_idx = subcategory_index[subcategory]
                    score = np.dot(W[user_idx], H[:, subcategory_idx])
                    user_scores.append(score)
                else:
                    user_scores.append(0)
            else:
                user_scores.append(0)
        scores.append(user_scores)

    behaviors_data["Scores"] = scores
    return behaviors_data[["ImpressionID", "Scores"]]


def non_negative_matrix(beh_df, news_df):

    news_id_to_subcategory = dict(zip(news_df["NewsID"], news_df["Category"]))

    # Create interaction matrix
    interaction_matrix = create_interaction_matrix(
        beh_df, news_id_to_subcategory
    )

    # Perform matrix factorization
    W, H, user_ids, subcategories = matrix_factorization(interaction_matrix)

    # Get scores for each impression
    result_df = get_scores_for_impressions(
        beh_df, W, H, user_ids, subcategories, news_id_to_subcategory
    )
    return result_df
