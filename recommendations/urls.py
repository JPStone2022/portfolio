# recommendations/urls.py

from django.urls import path
from . import views

app_name = 'recommendations' # Namespace

urlpatterns = [
    path('', views.recommendation_list_view, name='recommendation_list'),
    # Add detail view URL later if needed
]
