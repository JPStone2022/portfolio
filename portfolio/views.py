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
from django.db.models import Q
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
        # Add default meta tags for index if needed, or rely on base.html defaults
    }
    return render(request, 'portfolio/index.html', context=context)

# --- Project Views ---
def all_projects_view(request):
    """ View function for displaying all projects. """
    projects = Project.objects.all()
    context = { 'page_title': 'All Projects', 'projects': projects, }
    return render(request, 'portfolio/all_projects.html', context=context)

def project_detail(request, slug):
    """ View function for displaying a single project's details. """
    project = get_object_or_404(Project, slug=slug)

    # --- Generate Meta Tags ---
    # Description: Use the start of the project description
    meta_description = Truncator(project.description).words(25, truncate=' ...') # Truncate to ~25 words

    # Keywords: Combine project title, skills, and some defaults
    keywords = [project.title, 'deep learning', 'AI', 'project']
    if hasattr(project, 'skills'): # Check if skills relationship exists
        keywords.extend([skill.name for skill in project.skills.all()])
    meta_keywords = ", ".join(list(set(keywords))) # Join unique keywords

    context = {
        'project': project,
        'page_title': f"{project.title} - Project Details",
        'meta_description': meta_description, # Pass to template
        'meta_keywords': meta_keywords,     # Pass to template
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

            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message_body = form.cleaned_data['message']

            try:
                recipient_email = settings.EMAIL_HOST_USER
                if not recipient_email:
                     raise ValueError("Recipient email (EMAIL_HOST_USER) not configured.")

                send_mail(
                    f"Contact Form Submission: {subject}",
                    f"Name: {name}\nEmail: {email}\n\nMessage:\n{message_body}",
                    settings.DEFAULT_FROM_EMAIL,
                    [recipient_email],
                    fail_silently=False,
                )
                
                messages.success(request, 'Your message has been sent successfully! Thank you.')
            except Exception as e:
                print(f"Email sending error: {e}")
                messages.error(request, f'Sorry, there was an error sending your message. Please try again later.')

            return redirect(reverse('portfolio:contact'))
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ContactForm()

    context = {
        'form': form,
        'page_title': 'Contact Me',
    }
    return render(request, 'portfolio/contact_page.html', context=context)

# --- About Me View ---
def about_me_view(request):
    """ Renders the detailed About Me page. """
    context = {
        'page_title': 'About Me',
        # Add specific meta description/keywords for the about page if desired
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
            Q(title__icontains=query) |
            Q(description__icontains=query) |
            Q(skills__name__icontains=query) # Search linked skill names
        ).distinct()

        if Skill:
            skill_results = Skill.objects.filter(
                Q(name__icontains=query) |
                Q(description__icontains=query)
            ).distinct()

    context = {
        'query': query,
        'projects': project_results,
        'skills': skill_results,
        'page_title': f'Search Results for "{query}"' if query else 'Search',
        # Add meta tags for search results page
        'meta_description': f"Search results for '{query}' on [Your Name]'s deep learning portfolio.",
        'meta_keywords': f"search results, {query}, deep learning, AI, portfolio",
    }
    return render(request, 'portfolio/search_results.html', context=context)

