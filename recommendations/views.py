# recommendations/views.py

from django.shortcuts import render, get_object_or_404 # Import get_object_or_404
from .models import RecommendedProduct

def recommendation_list_view(request):
    """ Displays a list of all recommended products. """
    recommendations = RecommendedProduct.objects.all() # Fetch all, ordered by Meta
    context = {
        'recommendations': recommendations,
        'page_title': 'Recommendations',
    }
    return render(request, 'recommendations/recommendation_list.html', context)

# Add the detail view function
def recommendation_detail_view(request, slug):
    """ Displays details for a single recommended product. """
    product = get_object_or_404(RecommendedProduct, slug=slug)
    context = {
        'product': product,
        'page_title': product.name, # Use product name as title
        'meta_description': f"Recommendation for {product.name}: {product.description[:150]}...",
        'meta_keywords': f"{product.name}, {product.category}, recommendation, {', '.join(product.name.split())}",
    }
    return render(request, 'recommendations/recommendation_detail.html', context=context)

