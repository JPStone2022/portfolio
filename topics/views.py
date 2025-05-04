# topics/views.py

from django.shortcuts import render, get_object_or_404
from .models import ProjectTopic
# Import Project model safely for prefetching
try:
    from portfolio.models import Project
    PORTFOLIO_APP_EXISTS = True
except ImportError:
    Project = None
    PORTFOLIO_APP_EXISTS = False

def topic_list(request):
    """ Displays a list of all project topics. """
    all_topics = ProjectTopic.objects.all() # Fetches all, ordered by Meta
    context = {
        'topics': all_topics,
        'page_title': 'Project Topics',
        'meta_description': "Browse projects categorized by topic areas like Computer Vision, NLP, Web Development, and more.",
        'meta_keywords': "project topics, categories, portfolio, AI, machine learning, data science",
    }
    return render(request, 'topics/topic_list.html', context)


def topic_detail(request, topic_slug):
    """ Displays details for a single topic and lists associated projects. """
    topic = get_object_or_404(ProjectTopic, slug=topic_slug)
    related_projects = None
    if PORTFOLIO_APP_EXISTS and Project and hasattr(topic, 'projects'):
        # Get related projects using the related_name 'projects'
        # Prefetch skills/topics for efficiency in the template
        related_projects = topic.projects.prefetch_related('skills', 'topics').all()

    context = {
        'topic': topic,
        'projects': related_projects, # Pass projects related to this topic
        'page_title': f"Projects - {topic.name}",
        'meta_description': f"Projects related to {topic.name}. {topic.description[:120]}..." if topic.description else f"Projects related to {topic.name}.",
        'meta_keywords': f"{topic.name}, project, portfolio, deep learning, AI",
    }
    return render(request, 'topics/topic_detail.html', context=context)

