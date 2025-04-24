# skills/views.py

from django.shortcuts import render, get_object_or_404
from .models import Skill, SkillCategory

def skill_list(request):
    """ Displays skills grouped by category. """
    # Fetch categories with their related skills prefetched for efficiency
    categories = SkillCategory.objects.prefetch_related('skills').order_by('order', 'name')
    # Fetch skills without a category
    uncategorized_skills = Skill.objects.filter(category__isnull=True).order_by('order', 'name')

    context = {
        'categories': categories,
        'uncategorized_skills': uncategorized_skills,
        'page_title': 'Technical Skills',
    }
    return render(request, 'skills/skill_list.html', context)

def skill_detail(request, slug):
    """ Displays details for a single skill. """
    skill = get_object_or_404(Skill, slug=slug)
    context = {
        'skill': skill,
        'page_title': f"{skill.name} - Skill Details",
    }
    # Optional: Add related projects or blog posts here later
    return render(request, 'skills/skill_detail.html', context)