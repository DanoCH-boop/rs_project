import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# Perform matrix factorization using the optimal number of latent factors
def matrix_factorization(interaction_matrix, num_features):
    U, sigma, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    sigma = np.diag(sigma)
    U_reduced = U[:, :num_features]
    sigma_reduced = sigma[:num_features, :num_features]
    Vt_reduced = Vt[:num_features, :]

    reduced_matrix = np.dot(np.dot(U_reduced, sigma_reduced), Vt_reduced)

    return reduced_matrix


# Get scores for each impression
def get_scores_for_impressions(
    behaviors_data,
    user_ids,
    subcategories,
    news_id_to_subcategory,
    interaction_matrix_reduced,
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

                    score = interaction_matrix_reduced[
                        user_idx, subcategory_idx
                    ]

                    user_scores.append(score)
                else:
                    user_scores.append(
                        0
                    )  # Assign 0 score if subcategory not in the matrix
            else:
                user_scores.append(
                    0
                )  # Assign 0 score if user or impression not in the matrix
        scores.append(user_scores)

    behaviors_data["Scores"] = scores
    return behaviors_data[["ImpressionID", "Scores"]]


def find_latent_factors(interaction_matrix):

    def calculate_explained_variance(interaction_matrix):
        U, sigma, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)
        explained_variance = np.square(sigma) / np.sum(np.square(sigma))
        cumulative_variance_explained = np.cumsum(explained_variance)

        return cumulative_variance_explained

    cumulative_variance_explained = calculate_explained_variance(
        interaction_matrix.values
    )

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance_explained, marker="o")
    plt.xlabel("Number of Latent Factors")
    plt.ylabel("Cumulative Variance Explained")
    plt.title("Variance Explained by Latent Factors")
    plt.grid(True)
    plt.show()

    optimal_num_features = np.argmax(cumulative_variance_explained >= 0.90) + 1

    print(f"Optimal number of latent features: {optimal_num_features}")


def svd_recommender(beh_df, news_df):

    news_id_to_subcategory = dict(zip(news_df["NewsID"], news_df["Category"]))

    # Create interaction matrix
    interaction_matrix = create_interaction_matrix(
        beh_df, news_id_to_subcategory
    )

    # find_latent_factors(beh_df, news_id_to_subcategory, interaction_matrix)
    optimal_num_features = 11

    interaction_matrix_reduced = matrix_factorization(
        interaction_matrix.values, optimal_num_features
    )

    user_ids = beh_df["UserID"].unique()
    subcategories = news_df["Category"].unique()

    result_df = get_scores_for_impressions(
        beh_df,
        user_ids,
        subcategories,
        news_id_to_subcategory,
        interaction_matrix_reduced,
    )

    return result_df
