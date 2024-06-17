from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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


def preprocess_data(behaviors, news):
    # Explode the 'History' column to have one article per row
    behaviors['History'] = behaviors['History'].apply(lambda x: x.split(' ') if pd.notna(x) else [])
    exploded_behaviors = behaviors.explode('History')

    # Merge behaviors with news to get subcategories
    merged = exploded_behaviors.merge(news, left_on='History', right_on='ID', how='left')

    # Create the user-subcategory interaction matrix
    interaction_matrix = merged.pivot_table(index='UserID', columns='Category', aggfunc='size', fill_value=0)

    return interaction_matrix

def _calculate_popularity_of_news(row, beh_df, cluster_for_users):
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
    cluster = cluster_for_users[row['UserID']] if row['UserID'] in cluster_for_users else None

    # add cluster info to the beh_df
    beh_df['Cluster'] = beh_df['UserID'].apply(lambda x: cluster_for_users[x] if x in cluster_for_users else None)

    # filter rows so that only the users in the same cluster are considered (if row for which we calculate popularity score is in no cluster (does not have a history, keep the row)
    filter_df = beh_df[
        beh_df['Cluster'] == cluster if cluster is not None else beh_df['Cluster'] == beh_df['Cluster']]

    # filter so that only the last 24 hours are considered
    filter_df = filter_df[(pd.to_datetime(filter_df['Time'], format=format) >= start_date) & (
            pd.to_datetime(filter_df['Time'], format=format) < end_date)]

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

    impression_coef_array = []

    for id in split_Impression_list:
        sliced_id = id.split("-")[0]

        row_index = find_index(sorted_popular, sliced_id)

        if row_index == 0:
            impression_coef_array.append(0)
        else:
            impression_coef_array.append((1 - (int(row_index) - 1) / (num_rows)))

    return impression_coef_array


# With PCA
def cluster_users(interaction_matrix, n_clusters=3, pca_components=2):
    # Apply PCA to reduce the dimensionality of the data
    if pca_components > 0:
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(interaction_matrix)
    else:
        reduced_data = interaction_matrix

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_data)
    labels = kmeans.labels_

    # Create a DataFrame with UserID and their cluster labels
    user_clusters = pd.DataFrame({
        'UserID': interaction_matrix.index,
        'Cluster': labels
    })

    return reduced_data, labels, user_clusters


def cluster_users_dbscan(interaction_matrix, eps=0.5, min_samples=5):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)

    return scaled_data, labels


def visualize_clusters(reduced_data, labels):
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('User Clustering Based on Category Interaction')
    plt.show()


def evaluate_clusters(reduced_data, labels):
    silhouette_avg = silhouette_score(reduced_data, labels)
    davies_bouldin_avg = davies_bouldin_score(reduced_data, labels)
    calinski_harabasz_avg = calinski_harabasz_score(reduced_data, labels)

    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Davies-Bouldin Index: {davies_bouldin_avg}')
    print(f'Calinski-Harabasz Index: {calinski_harabasz_avg}')


def find_optimal_clusters(interaction_matrix, max_clusters=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)

    inertia = []
    for n in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()


def cluster_users_kmeans_no_pca(interaction_matrix, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_

    # Create a DataFrame with UserID and their cluster labels
    user_clusters = pd.DataFrame({
        'UserID': interaction_matrix.index,
        'Cluster': labels
    })

    return scaled_data, labels, user_clusters


def compare_cluster_methods(beh_df, news_df):
    behaviors, news = load_data(truncate_behaviors=1000)
    interaction_matrix = preprocess_data(behaviors, news)

    # THE BEST
    reduced_data, labels, _ = cluster_users(interaction_matrix, n_clusters=2)
    visualize_clusters(reduced_data, labels)
    print("BEST: K-2 clusters with PCA 2 components")
    evaluate_clusters(reduced_data, labels)

    ### K-Means Clustering
    reduced_data, labels, _ = cluster_users(interaction_matrix, n_clusters=20)
    visualize_clusters(reduced_data, labels)
    print("K-20 clusters with PCA 2 components")
    evaluate_clusters(reduced_data, labels)

    for i in range(2, 30):
        reduced_data, labels, _ = cluster_users(interaction_matrix, n_clusters=i)
        print(f"\n\nK-{i} clusters with PCA 2 components")
        evaluate_clusters(reduced_data, labels)

    ### K-Means Clustering
    reduced_data, labels, _ = cluster_users(interaction_matrix, n_clusters=20, pca_components=3)
    print("K-20 clusters with PCA 3 components")
    evaluate_clusters(reduced_data, labels)

    ### K-Means Clustering without PCA
    scaled_data, labels_kmeans, _ = cluster_users_kmeans_no_pca(interaction_matrix, n_clusters=20)
    print("K-20 clusters WITHOUT PCA")
    evaluate_clusters(scaled_data, labels_kmeans)

    ### DBSCAN Clustering
    scaled_data, labels_dbscan = cluster_users_dbscan(interaction_matrix, eps=0.5, min_samples=5)
    print("DBSCAN clusters")
    evaluate_clusters(scaled_data, labels_dbscan)

    ### K-Means Clustering
    # Determine the optimal number of clusters using the Elbow Method
    # find_optimal_clusters(interaction_matrix, max_clusters=20)

    scaled_data, labels_kmeans, _ = cluster_users_kmeans_no_pca(interaction_matrix, n_clusters=7)
    print("K-7 clusters")
    evaluate_clusters(scaled_data, labels_kmeans)


def kn_recommender(beh_df, news_df):
    beh_df_full, _ = load_data(truncate_behaviors=10000)
    interaction_matrix = preprocess_data(beh_df, news_df)

   # compare_cluster_methods(beh_df, news_df)

    reduced_data, labels, user_clusters = cluster_users(interaction_matrix, n_clusters=2, pca_components=2)
    #visualize_clusters(reduced_data, labels)
    evaluate_clusters(reduced_data, labels)

    # print number of users in each cluster
    print(user_clusters['Cluster'].value_counts())

    # remove clusters with less than 50 users
    cluster_counts = user_clusters['Cluster'].value_counts()
    #cluster_counts = cluster_counts[cluster_counts >= 50]
    user_clusters = user_clusters[user_clusters['Cluster'].isin(cluster_counts.index)]
    cluster_for_users = user_clusters.set_index('UserID')['Cluster'].to_dict()

    print("Clusters with more than 50 users:")
    print(cluster_counts)

    beh_df['Scores'] = beh_df.apply(_calculate_popularity_of_news,
                                    args=(beh_df_full, cluster_for_users),
                                    axis=1)
    return beh_df[['ID', 'Scores']]


if __name__ == '__main__':
    evaluator = Evaluator("../../datasets/MINDsmall_dev", truncate_behaviors=1000)
    recommenders = {
        "kn recommender": kn_recommender,
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]

    print(evals)

    #
