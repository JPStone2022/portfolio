# portfolio/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.contrib import messages
from .models import Project, Certificate
# Import models from other apps
try:
    from blog.models import BlogPost
except ImportError:
    BlogPost = None
try:
    from skills.models import Skill, SkillCategory
except ImportError:
    Skill = None
    SkillCategory = None

from django.utils import timezone
from .forms import ContactForm
from django.core.mail import send_mail
from django.conf import settings
from django.db.models import Q # Import Q object for complex lookups
# Import utility for truncating words
from django.utils.text import Truncator

def index(request):
    """ View function for the home page. """
    featured_projects = Project.objects.all()[:6]
    featured_certificates = Certificate.objects.all()[:3]
    latest_blog_post = None
    if BlogPost:
        try:
            latest_blog_post = BlogPost.objects.filter(
                status='published',
                published_date__lte=timezone.now()
            ).order_by('-published_date').first()
        except Exception as e:
            print(f"Could not query BlogPost: {e}")
            pass

    context = {
        'page_title': 'My Deep Learning Portfolio',
        'featured_projects': featured_projects,
        'featured_certificates': featured_certificates,
        'latest_blog_post': latest_blog_post,
    }
    return render(request, 'portfolio/index.html', context=context)

# --- Project Views ---
def all_projects_view(request):
    """ View function for displaying all projects, with filtering and sorting. """
    projects_list = Project.objects.all() # Start with all projects
    skills_list = Skill.objects.all().order_by('name') if Skill else None # Get all skills for filtering

    # Get filter/sort parameters from GET request
    selected_skill_slug = request.GET.get('skill', '')
    sort_by = request.GET.get('sort', '-date_created') # Default sort: newest first

    # Filter by skill if selected
    if selected_skill_slug and skills_list:
        try:
            selected_skill = Skill.objects.get(slug=selected_skill_slug)
            projects_list = projects_list.filter(skills=selected_skill)
        except Skill.DoesNotExist:
            selected_skill_slug = '' # Reset if invalid slug provided
            messages.warning(request, f"Skill filter '{selected_skill_slug}' not found.") # Optional message

    # Apply sorting
    valid_sort_options = {
        '-date_created': '-date_created', # Newest First (Default)
        'date_created': 'date_created',   # Oldest First
        'title': 'title',                 # Title A-Z
        '-title': '-title',               # Title Z-A
        'order': 'order',                 # Custom Order Asc
        '-order': '-order',               # Custom Order Desc
    }
    # Use default if provided sort option is invalid
    sort_key = valid_sort_options.get(sort_by, '-date_created')
    projects_list = projects_list.order_by(sort_key)

    context = {
        'page_title': 'All Projects',
        'projects': projects_list,
        'skills_list': skills_list, # Pass skills for the filter dropdown
        'selected_skill_slug': selected_skill_slug, # Pass current filter back to template
        'current_sort': sort_by, # Pass current sort back to template
        'valid_sort_options': valid_sort_options.keys(), # Pass valid keys for dropdown/links
    }
    return render(request, 'portfolio/all_projects.html', context=context)

def project_detail(request, slug):
    """ View function for displaying a single project's details. """
    project = get_object_or_404(Project, slug=slug)
    meta_description = Truncator(project.description).words(25, truncate=' ...')
    keywords = [project.title, 'deep learning', 'AI', 'project']
    if hasattr(project, 'skills'):
        keywords.extend([skill.name for skill in project.skills.all()])
    meta_keywords = ", ".join(list(set(keywords)))
    context = {
        'project': project,
        'page_title': f"{project.title} - Project Details",
        'meta_description': meta_description,
        'meta_keywords': meta_keywords,
    }
    return render(request, 'portfolio/project_detail.html', context=context)

# --- Certificate View ---
def certificates_view(request):
    """ View function for the certificates page. """
    certificates = Certificate.objects.all()
    context = { 'page_title': 'My Certificates', 'certificates': certificates, }
    return render(request, 'portfolio/certificates.html', context=context)

# --- Contact View ---
def contact_view(request):
    """ Handles the contact form display and submission. """
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['honeypot']:
                 messages.error(request, "Spam detected.")
                 return redirect(reverse('portfolio:contact'))
            name = form.cleaned_data['name']; email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']; message_body = form.cleaned_data['message']
            try:
                recipient_email = settings.EMAIL_HOST_USER
                if not recipient_email: raise ValueError("Recipient email not configured.")
                send_mail(f"Contact Form: {subject}", f"Name: {name}\nEmail: {email}\n\nMsg:\n{message_body}",
                          settings.DEFAULT_FROM_EMAIL, [recipient_email], fail_silently=False)
                messages.success(request, 'Message sent successfully! Thank you.')
            except Exception as e:
                print(f"Email error: {e}"); messages.error(request, 'Sorry, error sending message.')
            return redirect(reverse('portfolio:contact'))
        else: messages.error(request, 'Please correct the errors below.')
    else: form = ContactForm()
    context = {'form': form, 'page_title': 'Contact Me'}
    return render(request, 'portfolio/contact_page.html', context=context)

# --- About Me View ---
def about_me_view(request):
    """ Renders the detailed About Me page. """
    context = {
        'page_title': 'About Me',
        'meta_description': "Learn more about [Your Name], a deep learning engineer specializing in [mention key areas like CV/NLP]. Discover my background, experience, and technical skills.",
        'meta_keywords': "about me, deep learning engineer, AI developer, machine learning, background, experience, skills, [Your Name]",
    }
    return render(request, 'portfolio/about_me_page.html', context=context)

# --- CV View ---
def cv_view(request):
    """ Renders the CV/Resume download page. """
    context = {
        'page_title': 'CV / Resume',
        'meta_description': "View or download the CV / Resume for [Your Name], detailing experience in deep learning, AI, and software development.",
        'meta_keywords': "cv, resume, curriculum vitae, deep learning, AI, machine learning, portfolio, download, [Your Name]",
    }
    return render(request, 'portfolio/cv_page.html', context=context)

# --- Search View ---
def search_results_view(request):
    """ Handles searching across projects and skills. """
    query = request.GET.get('q', '')
    project_results = Project.objects.none()
    skill_results = Skill.objects.none() if Skill else Skill.objects.none()
    if query:
        project_results = Project.objects.filter(
            Q(title__icontains=query) | Q(description__icontains=query) | Q(skills__name__icontains=query)
        ).distinct()
        if Skill:
            skill_results = Skill.objects.filter(
                Q(name__icontains=query) | Q(description__icontains=query)
            ).distinct()
    context = {
        'query': query, 'projects': project_results, 'skills': skill_results,
        'page_title': f'Search Results for "{query}"' if query else 'Search',
        'meta_description': f"Search results for '{query}' on [Your Name]'s deep learning portfolio.",
        'meta_keywords': f"search results, {query}, deep learning, AI, portfolio",
    }
    return render(request, 'portfolio/search_results.html', context=context)

