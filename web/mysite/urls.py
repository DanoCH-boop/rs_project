from django.contrib import admin
from django.urls import include, path

from .views import index, article_detail

urlpatterns = [
    path('admin/', admin.site.urls),
    path('article/<slug:article_id>/', article_detail, name='detail'),
    path('', index, name='index'),
    path('category/<slug:category>/', index, name='index_by_category'),
]