from datetime import datetime, timedelta
import pandas as pd

def most_popular_recommender(beh_df, news_df, category_filter=[], recommender=False):
    def find_index(series, name):
        try:
            index = series.index.get_loc(name) + 1  # Adding 1 to start index from 1
            return index
        except KeyError:
            return 0

    result_df = pd.DataFrame(columns=['ID', 'Scores'])
    
    for index, row in beh_df.iterrows():
        date = row['Time']
        format = '%m/%d/%Y %I:%M:%S %p'
        end_date = datetime.strptime(date, format)
        start_date = end_date - timedelta(hours=24)

        if recommender:
            filter_df = beh_df[(pd.to_datetime(beh_df['Time'], format=format) >= start_date) & (
                    pd.to_datetime(beh_df['Time'], format=format) <= end_date)]
        else:
            filter_df = beh_df[(pd.to_datetime(beh_df['Time'], format=format) >= start_date) & (
                pd.to_datetime(beh_df['Time'], format=format) < end_date)]

        split_ids = filter_df['Impression'].str.split()
        all_article_ids = []

        for ids in split_ids:
            all_article_ids.extend([id.split("-")[0] for id in ids if id.endswith('-1')])

        # Filter out articles not in wanted subcategories
        if len(category_filter) != 0:
            for article_id in all_article_ids:
                art_subcat = news_df[news_df["ID"] == article_id]['Subcategory'].values[0]
                if not art_subcat in category_filter:
                    all_article_ids.remove(article_id)
                
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
                impression_coef_array.append(1 - (int(row_index)) / (num_rows))

        result_df.loc[len(result_df.index)] = [index + 1, impression_coef_array] 
    return result_df
