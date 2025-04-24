# portfolio/urls.py (App-level URLs)

from django.urls import path
from . import views # Import views from the current directory

app_name = 'portfolio' # Namespace for URLs

urlpatterns = [
    # Home page
    path('', views.index, name='index'),
    # Project detail page
    path('project/<slug:slug>/', views.project_detail, name='project_detail'),
    # Add path for certificates page
    path('certificates/', views.certificates_view, name='certificates'),
    # Add path for the all projects page
    path('projects/', views.all_projects_view, name='all_projects'),
    # Add path for the contact page view
    path('contact/', views.contact_view, name='contact'),
    # Add path for the detailed about me page
    path('about-me/', views.about_me_view, name='about_me'),

]
