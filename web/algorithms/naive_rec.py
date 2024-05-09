# Martin Kozák
# FI MUNI, Brno

import pandas as pd
from datetime import datetime, timedelta


def load_data():
    news_columns = ['ID', 'Category', 'Subcategory', 'Title', 'Abstract']
    news_df = pd.read_csv('datasets/MINDsmall_dev/news.tsv', sep='\t', names=news_columns, usecols=[0, 1, 2, 3, 4])

    beh_columns = ['ID', 'UserID', 'Time', 'History', 'Impression']
    beh_df = pd.read_csv('datasets/MINDsmall_dev/behaviors.tsv', sep='\t', names=beh_columns, usecols=[0, 1, 2, 3, 4])

    return news_df, beh_df

# Most Popular

def get_most_popular(date_str, beh_df):
    date = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
    start_date = date - timedelta(hours=24)
    
    filtered_df = beh_df[(beh_df['Time'] >= start_date) & (beh_df['Time'] < date)]
    split_ids = filtered_df['Impression'].str.split()
    start_date = start_date.strftime('%m/%d/%Y %I:%M:%S %p')

    all_article_ids = []
    
    for ids in split_ids:
        all_article_ids.extend([id.strip('-1') for id in ids if id.endswith('-1')])
    
    all_article_ids_series = pd.Series(all_article_ids)
    
    popular_clicked_articles = all_article_ids_series.value_counts()
    
    top_5_clicked_articles = popular_clicked_articles.head(5)

    return top_5_clicked_articles

def get_coeficcients_most_popular(beh_df): 
    def find_index(series, name):
        try:
            index = series.index.get_loc(name) + 1  # Adding 1 to start index from 1
            return index
        except KeyError:
            return 0

    result_df = pd.DataFrame(columns=['id', 'array'])
    
    for index, row in beh_df.iterrows():

        result_df = pd.DataFrame(columns=['id', 'array'])

        date = row['Time']
        date_ = datetime.strptime(date, '%m/%d/%Y %I:%M:%S %p')
        start_date = date_ - timedelta(hours=24)
        start_date = start_date.strftime('%m/%d/%Y %I:%M:%S %p')

        filtered_df = beh_df[(beh_df['Time'] >= start_date) & (beh_df['Time'] < date)]
        split_ids = filtered_df['Impression'].str.split()

        all_article_ids = []

        for ids in split_ids:
            all_article_ids.extend([id.strip('-1') for id in ids if id.endswith('-1')])

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
                impression_coef_array.append(1 - (int(row_index) - 1) / (num_rows - 1))

        result_df.loc[len(result_df.index)] = [index + 1, impression_coef_array] 
    return result_df

# Most Popular by categories

def get_top_categories(beh_df, news_df, user_id, count=5):
    # Filter behavior data for the given user
    user_data = beh_df[beh_df['UserID'] == user_id]
    
    # Combine history of the user
    oldest_entry = user_data.sort_values(by='Time').iloc[0]
    history = oldest_entry['History']

    category_counts = news_df[news_df['ID'].isin(history.split())].groupby('Category').size()
    subcategory_counts = news_df[news_df['ID'].isin(history.split())].groupby('Subcategory').size()
    
    top_categories = category_counts.nlargest(count).index.tolist()
    top_subcategories = subcategory_counts.nlargest(count).index.tolist()
    
    return top_categories, top_subcategories

def select_random_articles(news_df, categories, subcategories):
    selected_category_articles = {}
    selected_subcategory_articles = {}
    
    # Select 5 random articles for each category and subcategory
    for category in categories:
        category_articles = news_df[news_df['Category'] == category]
        random_category_articles = category_articles.sample(n=5)
        selected_category_articles[category] = random_category_articles['ID'].tolist()
        
    for subcategory in subcategories:
        subcategory_articles = news_df[news_df['Subcategory'] == subcategory]
        random_subcategory_articles = subcategory_articles.sample(n=5)
        selected_subcategory_articles[subcategory] = random_subcategory_articles['ID'].tolist()
        
        
    return selected_category_articles, selected_subcategory_articles

def example():
    # Example usage:
    news_df, beh_df = load_data()

    user_id = 'U81540'  # Example user ID

    top_categories, top_subcategories = get_top_categories(beh_df, news_df, user_id)
    selected_category_articles, selected_subcategory_articles  = select_random_articles(news_df, top_categories, top_subcategories)

    date_str = '11/14/2019 12:26:47 PM'  # Example timestamp
    top_articles = get_most_popular(date_str, beh_df)
    print("Top 5 Most Popular Clicked Articles within 24 hours before", date_str, ":")
    print(top_articles)

    print(120*"-")

    print("Top 5 most popular categories for user", user_id, ":", top_categories)
    print("Top 5 most popular subcategories for user", user_id, ":", top_subcategories)
    print("Selected category articles for user", user_id, ":", selected_category_articles)
    print("Selected subcategory articles for user", user_id, ":", selected_subcategory_articles)

    top_articles = news_df[news_df['ID'].isin(top_articles.keys())]

    return top_articles
