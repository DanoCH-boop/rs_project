import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def sentence_embedding_recommender(beh_df, news_df):

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def aggregate_similarity_scores(impression_id, history_ids):
        history_embeddings = [model.encode(news_df.loc[news_df['ID'] == news_id, 'Title'].values[0]) for news_id in history_ids]
        impression_embedding = model.encode(news_df.loc[news_df['ID'] == impression_id, 'Title'].values[0])
        return sum(cosine_similarity(history_embedding, impression_embedding) for history_embedding in history_embeddings) / max(len(history_embeddings), 1)
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    beh_df['NewsOptions'] = beh_df['Impression'].str.split().apply(lambda impressions: [impression[:-2] for impression in impressions])
    beh_df['History'] = beh_df['History'].apply(lambda history: [] if isinstance(history, float) and np.isnan(history) else history.split())

    result_data = []

    for index, row in beh_df.iterrows():
        print(index)
        scores = [aggregate_similarity_scores(impression, row['History']) for impression in row['NewsOptions']]
        result_data.append({'ID': row['ID'], 'Scores': scores})

    result_df = pd.DataFrame(result_data)
    return result_df[['ID', 'Scores']]
