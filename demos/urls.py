# demos/urls.py

from django.urls import path
from . import views

app_name = 'demos' # Namespace

urlpatterns = [
    path('image-classifier/', views.image_classification_view, name='image_classifier'),
    # Add paths for other demos here later
    # Add path for sentiment analysis demo
    path('sentiment-analyzer/', views.sentiment_analysis_view, name='sentiment_analyzer'),
    # Add path for data analysis demo
    path('csv-analyzer/', views.data_analysis_view, name='csv_analyzer'),
    # Add path for data wrangling demo
    path('data-wrangler/', views.data_wrangling_view, name='data_wrangler'),
    # Add path for explainable AI demo
    path('explainable-ai/', views.explainable_ai_view, name='explainable_ai'),
    # Add path for Flask API demo explanation page
    path('flask-ml-api/', views.flask_api_demo_view, name='flask_api_demo'),
    # Add path for Django concepts demo explanation page
    path('django-concepts/', views.django_concepts_demo_view, name='django_concepts_demo'),
    # Add path for causal inference demo
    path('causal-inference/', views.causal_inference_demo_view, name='causal_inference'),
    # Add path for optimization demo
    path('scipy-optimize/', views.optimization_demo_view, name='optimization_demo'),
    # Add path for Django security demo explanation page
    path('django-security/', views.django_security_demo_view, name='django_security_demo'),
    # Add path for Django testing demo explanation page
    path('django-testing/', views.django_testing_demo_view, name='django_testing_demo'),
    # Add path for AI tools explanation page
    path('ai-tools-dev/', views.ai_tools_demo_view, name='ai_tools_demo'),
    # Add path for Python concepts demo explanation page
    path('python-concepts/', views.python_concepts_demo_view, name='python_concepts_demo'),
    # Add path for R concepts demo explanation page
    path('r-concepts/', views.r_concepts_demo_view, name='r_concepts_demo'),
    # Add path for Go concepts demo explanation page
    path('go-concepts/', views.go_concepts_demo_view, name='go_concepts_demo'),
]
