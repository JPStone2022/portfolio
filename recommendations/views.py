# recommendations/views.py

from django.shortcuts import render
from .models import RecommendedProduct

def recommendation_list_view(request):
    """ Displays a list of all recommended products. """
    recommendations = RecommendedProduct.objects.all() # Fetch all, ordered by Meta
    context = {
        'recommendations': recommendations,
        'page_title': 'Recommendations',
    }
    return render(request, 'recommendations/recommendation_list.html', context)

