import networkx as nx
import matplotlib.pyplot as plt
import ast
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from web.algorithms.tf_idf import preprocess
from web.algorithms.naive_rec import get_top_categories
from web.algorithms.most_popular_recommender import most_popular_recommender


# Find shared subcategories
def find_shared_subcats(news_df):
    subcategory_groups = news_df.groupby("Subcategory")

    shared_subcategories = []

    for subcategory, group in subcategory_groups:
        unique_categories = group["Category"].nunique()

        if unique_categories > 1:
            shared_subcategories.append(subcategory)

    return shared_subcategories


def load_cat_dict(category):
    # File path from which to read the dictionary
    file_path = f"category_sims/{category}.txt"

    try:
        with open(file_path, "r") as file:
            # Read the entire file content as a string
            data_str = file.read().strip()

            # Evaluate the string to reconstruct the dictionary
            data_dict = ast.literal_eval(data_str)

            print("Dictionary loaded successfully from file: ", file_path)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except ValueError:
        print(f"Error: File '{file_path}' contains invalid data format.")

    return data_dict


def draw_graph_sim_cat(news_df):
    # Create an empty graph
    G = nx.Graph()

    # Remove low occurence categories like ("kids", "northamerica", "middleeast")
    categories = news_df["Category"].unique()[:-3]

    category_counts = news_df["Category"].value_counts()
    category_size_map = category_counts.to_dict()

    # Calculate node sizes based on occurence_dict
    max_occurrence = max(category_size_map.values())
    min_occurrence = min(category_size_map.values())

    node_sizes = [
        1000
        + (10000 - 1000)
        * (
            (category_size_map[cat] - min_occurrence)
            / (max_occurrence - min_occurrence)
        )
        for cat in categories
    ]

    # Colors for nodes
    colors = [
        "#1f77b4",
        "#8c564b",
        "#98df8a",
        "#c7c7c7",
        "#aec7e8",
        "#bcbd22",
        "#c49c94",
        "#d62728",
        "#dbdb8d",
        "#e377c2",
        "#ff7f0e",
        "#ff9896",
        "#17becf",
        "#9467bd",
    ]

    data_dict = load_cat_dict("categories")

    # Add nodes for and edges between categories
    for i in range(len(categories)):
        cat1 = categories[i]
        G.add_node(cat1, node_type="category")
        for j in range(i + 1, len(categories)):
            cat2 = categories[j]
            print(cat1, cat2)
            sim_score = data_dict[(categories[i], categories[j])]
            # sim_score = sim_score if sim_score < 20 else 20
            G.add_edge(cat1, cat2, weight=sim_score)

    # Visualize the graph
    pos = nx.spring_layout(G)  # Layout algorithm for graph visualization
    plt.figure(figsize=(16, 9))

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)

    # Add labels and legend
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color="black")

    plt.title(
        "Category Similarity Graph using TF-IDF and Cosine Similiarity",
        fontsize=24,
    )
    plt.tight_layout(pad=0.5)
    plt.axis("off")
    plt.savefig("graphs/fig_cats.pdf")


def draw_graph_sim_subcat(news_df):
    # Create an empty graph
    G = nx.Graph()

    categories = news_df["Category"].unique()

    # Colors for nodes
    colors = [
        "#1f77b4",
        "#8c564b",
        "#98df8a",
        "#c7c7c7",
        "#aec7e8",
        "#bcbd22",
        "#c49c94",
        "#d62728",
        "#dbdb8d",
        "#e377c2",
        "#ff7f0e",
        "#ff9896",
        "#17becf",
        "#9467bd",
        "#f7b6d2",
        "#ffbb78",
        "#2ca02c",
        "#7f7f7f",
        "#9edae5",
        "#c5b0d5",
    ]

    # Assign colors to categories dynamically
    category_color_map = {
        category: colors[i % len(colors)]
        for i, category in enumerate(categories)
    }

    print(category_color_map)
    # for category in categories:
    #     G.add_node(category, node_type='category', color=category_color_map[category])

    # Add nodes for subcategories and edges between subcategories of the same category
    for category in categories:
        # Filter categories with low count
        if category in ("kids", "northamerica", "middleeast"):
            continue

        data_dict = load_cat_dict(category)

        subcategories = news_df[news_df["Category"] == category][
            "Subcategory"
        ].unique()
        shared_subcategories = find_shared_subcats(news_df)
        print(category)
        for i in range(len(subcategories)):
            subcat1 = (
                subcategories[i]
                if not subcategories[i] in shared_subcategories
                else f"{category}_{subcategories[i]}"
            )
            G.add_node(
                subcat1,
                node_type="subcategory",
                color=category_color_map[category],
            )
            for j in range(i + 1, len(subcategories)):
                subcat2 = (
                    subcategories[j]
                    if not subcategories[j] in shared_subcategories
                    else f"{category}_{subcategories[j]}"
                )
                print(subcat1, subcat2)
                sim_score = data_dict[(subcategories[i], subcategories[j])]
                # sim_score = sim_score if sim_score < 20 else 20
                G.add_edge(subcat1, subcat2, weight=sim_score)

    # Visualize the graph
    number_of_nodes = G.number_of_nodes()
    pos = nx.spring_layout(
        G, k=1 / math.sqrt(math.sqrt(number_of_nodes))
    )  # Layout algorithm for graph visualization
    plt.figure(figsize=(32, 18))

    # Draw nodes and edges
    for node in G.nodes():
        node_type = G.nodes[node]["node_type"]
        node_color = G.nodes[node]["color"]
        if node_type == "category":
            # nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=1000, node_color=node_color, label=node)
            pass
        else:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node], node_size=100, node_color=node_color
            )
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.1)

    # Add labels and legend
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color="black")
    plt.legend(
        title="Category",
        labels=category_color_map.keys(),
        fontsize=20,
        title_fontsize=25,
        loc="upper left",
        markerscale=2,
    )

    ax = plt.gca()
    leg = ax.get_legend()
    for handle, color in zip(leg.legendHandles, colors):
        handle.set_color(color)

    plt.title(
        "Subcategory Similarity Graph within Main Categories", fontsize=30
    )
    plt.tight_layout(pad=0.5)
    plt.axis("off")
    plt.savefig("graphs/fig.pdf")


def compute_sims_cat(news_df):
    categories = news_df["Category"].unique()

    # Dictionary to store preprocessed titles by category
    category_titles = {}

    for category in categories:
        # Filter news for category
        category_news = news_df[news_df["Category"] == category]

        titles = category_news["Title"].tolist()
        preprocessed_titles = [preprocess(title) for title in titles]
        category_titles[category] = preprocessed_titles

    # Calculate TF-IDF vectors and cosine similarity for unique pairs of categories
    results = {}

    for i in range(len(categories)):
        cat1 = categories[i]
        titles1 = category_titles[cat1]
        for j in range(i + 1, len(categories)):
            cat2 = categories[j]
            # Combine titles for each pair of categories
            titles2 = category_titles[cat2]
            all_titles = titles1 + titles2

            # Compute TF-IDF vectors
            tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_titles)

            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Extract similarities between titles from cat1 and cat2
            similarity_scores = similarity_matrix[
                : len(titles1), len(titles1) :
            ]

            # Compute average similarity
            average_similarity = similarity_scores.sum() / len(titles2)

            # Store results
            results[(cat1, cat2)] = round(average_similarity, 5)

        sorted_results = {}

    for cat1 in categories:
        # Filter results by cat1
        filtered_results = {
            pair: similarity
            for pair, similarity in results.items()
            if pair[0] == cat1
        }

        # Sort filtered results by similarity value in descending order
        sorted_filtered_results = dict(
            sorted(
                filtered_results.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # Append sorted results to main dictionary
        sorted_results.update(sorted_filtered_results)

    # Save results
    with open("categories.txt", "w") as file:
        file.write(str(sorted_results))


def compute_sims_subcat(news_df):
    categories = news_df["Category"].unique()

    for category in categories:
        # Filter news for category
        category_news = news_df[news_df["Category"] == category]

        # Group titles by subcategory
        subcategories = category_news["Subcategory"].unique()

        # Dictionary to store preprocessed titles by subcategory
        subcategory_titles = {}

        # Preprocess titles for each subcategory
        for subcategory in subcategories:
            titles = category_news[
                category_news["Subcategory"] == subcategory
            ]["Title"].tolist()
            preprocessed_titles = [preprocess(title) for title in titles]
            subcategory_titles[subcategory] = preprocessed_titles

        # Calculate TF-IDF vectors and cosine similarity for unique pairs of subcategories
        results = {}

        for i in range(len(subcategories)):
            subcat1 = subcategories[i]
            titles1 = subcategory_titles[subcat1]
            for j in range(i + 1, len(subcategories)):
                subcat2 = subcategories[j]
                # Combine titles for each pair of subcategories
                titles2 = subcategory_titles[subcat2]
                all_titles = titles1 + titles2

                # Compute TF-IDF vectors
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(all_titles)

                # Calculate pairwise cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Extract similarities between titles from subcat1 and subcat2
                similarity_scores = similarity_matrix[
                    : len(titles1), len(titles1) :
                ]

                # Compute average similarity
                average_similarity = similarity_scores.sum() / len(titles2)

                # Store results
                results[(subcat1, subcat2)] = round(average_similarity, 5)

        sorted_results = {}

        for subcat1 in subcategories:
            # Filter results by subcat1
            filtered_results = {
                pair: similarity
                for pair, similarity in results.items()
                if pair[0] == subcat1
            }

            # Sort filtered results by similarity value in descending order
            sorted_filtered_results = dict(
                sorted(
                    filtered_results.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            # Append sorted results to main dictionary
            sorted_results.update(sorted_filtered_results)

        # Save results
        with open(f"{category}.txt", "w") as file:
            file.write(str(sorted_results))


def find_sim_subcat(dictionary, category_str):
    max_score = -1  # Initialize maximum score
    best_sim = [category_str]  # Initialize tuple with the highest score

    for (cat1, cat2), score in dictionary.items():
        if cat1 == category_str or cat2 == category_str:
            if score > max_score:
                max_score = score
                best_sim = [cat1, cat2]

    best_sim.remove(category_str)

    return best_sim[0]


def sim_cat_rec(beh_df, news_df, user_id="U81540", recommender=False):
    # Loading behaviors again so user can be foind
    beh_columns = ["ID", "UserID", "Time", "History", "Impression"]
    beh_df_user = pd.read_csv(
        "datasets/MINDsmall_dev/behaviors.tsv",
        sep="\t",
        names=beh_columns,
        usecols=[0, 1, 2, 3, 4],
    )
    # Get favorite categories and subcategories of user
    top_cats, top_subcats = get_top_categories(
        beh_df_user, news_df, user_id, count=3
    )

    # Get similiar categories to favorite catgories
    fav_cats = []

    for subcat in top_subcats:
        subcat_cat = news_df[news_df["Subcategory"] == subcat][
            "Category"
        ].unique()[0]

        subcat_sim_dict = load_cat_dict(subcat_cat)
        sim_subcat = find_sim_subcat(subcat_sim_dict, subcat)
        fav_cats.append(subcat)
        fav_cats.append(sim_subcat)

    # Get most popular articles of each category
    result_df = most_popular_recommender(
        beh_df, news_df, fav_cats, recommender
    )

    return result_df


if __name__ == "__main__":

    # Load dataset
    dataset_dir = "../../datasets/MINDsmall_dev"
    news_path = f"{dataset_dir}/news.tsv"
    beh_path = f"{dataset_dir}/behaviors.tsv"

    news_columns = ["ID", "Category", "Subcategory", "Title"]
    news_df = pd.read_csv(
        news_path, sep="\t", names=news_columns, usecols=[0, 1, 2, 3]
    )
    # news_df.drop_duplicates(subset=['Title'], inplace=True)
    # news_df.dropna(inplace=True)

    beh_columns = ["ID", "UserID", "Time", "History", "Impression"]
    beh_df = pd.read_csv(
        beh_path, sep="\t", names=beh_columns, usecols=[0, 1, 2, 3, 4]
    )

    # Compuate similiarities between categories and write them into a file
    # compute_sims_cat(news_df)

    # Draw graph of similiarties between categories
    # draw_graph_sim_cat(news_df)

    # Compuate similiarities between subcategories and write them into a file
    # compute_sims_subcat(news_df)

    # Draw graph of similiarties between subcategories
    # draw_graph_sim_subcat(news_df)

    # Get recommendations for a certain user
    result_df = sim_cat_rec(beh_df.head(100), news_df, user_id="U81540")
    print(result_df["Scores"].values)
