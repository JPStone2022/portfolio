# portfolio/views.py

from django.shortcuts import render, get_object_or_404
from .models import Project, Certificate # Import Certificate model

def index(request):
    """
    View function for the home page of the site.
    Fetches featured Project objects and featured Certificate objects.
    """
    # Fetch featured projects (e.g., top 6 based on ordering in model Meta)
    featured_projects = Project.objects.all()[:6] # Get the first 6 projects
    # Fetch featured certificates
    featured_certificates = Certificate.objects.all()[:3] # Get the first 3

    context = {
        'page_title': 'My Deep Learning Portfolio',
        # Pass only the featured projects to the index template
        'featured_projects': featured_projects,
        'featured_certificates': featured_certificates,
    }
    return render(request, 'portfolio/index.html', context=context)


def project_detail(request, slug):
    """
    View function for displaying a single project's details.
    Fetches the Project object based on its unique slug.
    """
    project = get_object_or_404(Project, slug=slug)
    context = {
        'project': project,
        'page_title': f"{project.title} - Project Details"
    }
    return render(request, 'portfolio/project_detail.html', context=context)


def certificates_view(request):
    """
    View function for the certificates page.
    Fetches Certificate objects and renders the certificates.html template.
    """
    certificates = Certificate.objects.all()
    context = {
        'page_title': 'My Certificates',
        'certificates': certificates,
    }
    return render(request, 'portfolio/certificates.html', context=context)


def all_projects_view(request):
    """
    View function for displaying all projects.
    """
    projects = Project.objects.all() # Get all projects ordered by Meta
    context = {
        'page_title': 'All Projects',
        'projects': projects, # Pass all projects here
    }
    return render(request, 'portfolio/all_projects.html', context=context)

