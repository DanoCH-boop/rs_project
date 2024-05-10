# Marián Ligocký
# MUNI FI, Brno

from django.shortcuts import render
from web.algorithms.evaluator import Evaluator
from web.algorithms.recommender import Recommender
from web.algorithms.naive_rec import load_data, get_most_popular
from web.algorithms.popular_category_combinations import category_combinations_recommender
from web.algorithms.random_recommender import random_rating_recommender
from web.algorithms.most_popular_recommender import most_popular_recommender
from web.algorithms.tf_idf_recommender import tf_idf_cosine_recommender


def index(request, category=None, page=1):
    articles_df, _beh_df = load_data()

    categories = articles_df['Category'].unique()

    if category:
        articles_df = articles_df[articles_df['Category'] == category]
    articles = articles_df.to_dict('records')[:100]

    return render(request, 'index.html', {'articles': articles, 'categories': categories, 'active_category': category})


def article_detail(request, article_id):
    # find in articles by article_id
    articles_df, beh_df = load_data()

    date_str = '11/14/2019 12:26:47 PM'
    recommended_articles_df = get_most_popular(date_str, beh_df)
    recommended_articles_ids = recommended_articles_df.index.tolist()

    recommended_articles = articles_df[articles_df['ID'].isin(recommended_articles_ids)].to_dict('records')

    article_df = articles_df[articles_df['ID'] == article_id]
    article = article_df.to_dict('records')[0]
    return render(request, 'detail.html', {'article': article, 'recommended_articles': recommended_articles})

def evaluation(request):
    evaluator = Evaluator()
    recommenders = {
        "Random recommender": random_rating_recommender,
        # "Most popular recommender": most_popular_recommender,
        "Tf-idf recommender": tf_idf_cosine_recommender,
        "Category combinations recommender": category_combinations_recommender
        # "Sentence embedding recommender": sentence_embedding_recommender,
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]
    evals.extend([
        {
            "name": "MIND competition 1. place",
            "evaluation": [0.7304, 0.3770, 0.4180, 0.4718]
        },
        {
            "name": "MIND competition 2. place",
            "evaluation": [0.7275, 0.3724, 0.4102, 0.4661]
        },
        {
            "name": "MIND competition 3. place",
            "evaluation": [0.7268, 0.3745, 0.4151, 0.4684]
        },
    ])
    return render(request, 'evaluation.html', {'recommenders': evals})
 

def recommend(request, row_id=46):
    
    recommender = Recommender(row_id=row_id)
    
    result = recommender.get_reccomendations()
    print(result)
    return render(request, 'recommendation.html', {'result': result})
