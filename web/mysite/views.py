# Marián Ligocký
# MUNI FI, Brno

from django.shortcuts import render
from web.algorithms.evaluator import Evaluator
from web.algorithms.naive_rec import load_data, get_most_popular
from web.algorithms.random_recommender import random_rating_recommender


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
        "Random recommender": random_rating_recommender
    }
    evals = [{
        "name": name,
        "evaluation": [round(result, 4) for result in evaluator.evaluate(recommender)]
    } for name, recommender in recommenders.items()]
    return render(request, 'evaluation.html', {'recommenders': evals})
