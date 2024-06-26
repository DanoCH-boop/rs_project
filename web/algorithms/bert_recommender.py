import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def bert_recommender(beh_df, news_df):

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    def get_bert_embeddings(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def get_history_embedding(history):
        if pd.isnull(history) or history.strip() == "":
            return []
        titles = [
            news_id_to_title.get(news_id, "") for news_id in history.split()
        ]
        titles_combined = " ".join(titles)
        return get_bert_embeddings(titles_combined)

    # map NewsID to Title
    news_id_to_title = dict(zip(news_df["NewsID"], news_df["Title"]))

    # get embeddings for impressions by looking up their titles in the news data
    def get_impression_embeddings(impressions):
        embeddings = []
        for impression in impressions:
            title = news_id_to_title.get(impression[:-2], "")
            if title:
                embedding = get_bert_embeddings(title)
                embeddings.append(embedding)
        return embeddings

    beh_df["History_Embeddings"] = beh_df["History"].apply(
        get_history_embedding
    )
    beh_df["Impression_Embeddings"] = beh_df["Impressions"].apply(
        get_impression_embeddings
    )

    def compute_cosine_similarities(history_embedding, impression_embeddings):
        if len(history_embedding) == 0:
            # generate random similarity scores if history_embedding is empty
            similarities = np.random.rand(len(impression_embeddings)).tolist()
        else:
            # ensure history_embedding is 1-dimensional
            history_embedding = np.array(history_embedding).flatten()
            similarities = []
            for impression_embedding in impression_embeddings:
                # ensure each impression_embedding is 1-dimensional
                impression_embedding = np.array(impression_embedding).flatten()
                similarity = cosine_similarity(
                    [history_embedding], [impression_embedding]
                )[0][0]
                similarities.append(similarity)
        return similarities

    result_data = []
    for index, row in beh_df.iterrows():
        history_embedding = row["History_Embeddings"]
        impression_embeddings = row["Impression_Embeddings"]
        scores = compute_cosine_similarities(
            history_embedding, impression_embeddings
        )
        result_data.append({"ID": row["ImpressionID"], "Scores": scores})
        result_df = pd.DataFrame(result_data)
        return result_df[["ID", "Scores"]]
