# demos/urls.py

from django.urls import path
from . import views

app_name = 'demos' # Namespace

urlpatterns = [
    path('image-classifier/', views.image_classification_view, name='image_classifier'),
    # Add paths for other demos here later
    # Add path for sentiment analysis demo
    path('sentiment-analyzer/', views.sentiment_analysis_view, name='sentiment_analyzer'),
]
