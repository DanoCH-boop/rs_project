from datetime import datetime, timedelta
import pandas as pd

def most_popular_recommender(beh_df, news_df):
    def find_index(series, name):
        try:
            index = series.index.get_loc(name) + 1  # Adding 1 to start index from 1
            return index
        except KeyError:
            return 0

    result_df = pd.DataFrame(columns=['ID', 'Scores'])
    
    for index, row in beh_df.iterrows():

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
