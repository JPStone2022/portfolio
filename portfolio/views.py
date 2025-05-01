# portfolio/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.contrib import messages
from .models import Project, Certificate, ProjectTopic
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
try: from recommendations.models import RecommendedProduct # Import recommendations model
except ImportError: RecommendedProduct = None
#try: from demos.models import Demo # Import Demo model
#except ImportError: Demo = None

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
    featured_certificates = Certificate.objects.all()[:6]
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

    # Fetch featured recommendations
    featured_recommendations = None
    if RecommendedProduct:
        try:
            featured_recommendations = RecommendedProduct.objects.all()[:3] # Get top 3 based on order
        except Exception as e: print(f"Could not query RecommendedProduct: {e}"); pass

    # Fetch featured topics (e.g., top 3 based on ordering)
    featured_topics = ProjectTopic.objects.all()[:10]

    # Fetch featured skills (e.g., top 6 based on ordering)
    featured_skills = None
    if Skill:
        try:
            featured_skills = Skill.objects.all()[:10] # Get top 6 based on default ordering
        except Exception as e:
            print(f"Could not query Skill: {e}"); pass

    # Fetch featured demos
    featured_demos = None
    # if Demo:
    #     try:
    #         featured_demos = Demo.objects.filter(is_featured=True).order_by('order')[:6] # Get top 3 featured
    #     except Exception as e:
    #         print(f"Could not query Demo: {e}"); pass

    context = {
        'page_title': 'My Deep Learning Portfolio',
        'featured_projects': featured_projects,
        'featured_certificates': featured_certificates,
        'latest_blog_post': latest_blog_post,
        'featured_recommendations': featured_recommendations, # Add to context
        'featured_topics': featured_topics, # Add featured topics to context
        'featured_skills': featured_skills, # Add featured skills to context
        #'featured_demos': featured_demos, # Add featured demos to context
    }
    return render(request, 'portfolio/index.html', context=context)

# --- Project Views ---
def all_projects_view(request):
    """ View function for displaying all projects, with filtering and sorting. """
    projects_list = Project.objects.prefetch_related('skills', 'topics').all() # Prefetch for efficiency
    skills_list = Skill.objects.all().order_by('name') if Skill else None
    topics_list = ProjectTopic.objects.all().order_by('name') # Get topics for filtering

    # Get filter/sort parameters
    selected_skill_slug = request.GET.get('skill', '')
    selected_topic_slug = request.GET.get('topic', '') # New topic filter
    sort_by = request.GET.get('sort', '-date_created')

    # Filter by skill
    if selected_skill_slug and skills_list:
        try:
            selected_skill = Skill.objects.get(slug=selected_skill_slug)
            projects_list = projects_list.filter(skills=selected_skill)
        except Skill.DoesNotExist: selected_skill_slug = ''; messages.warning(request, f"Skill filter '{selected_skill_slug}' not found.")

    # Filter by topic
    if selected_topic_slug:
        try:
            selected_topic = ProjectTopic.objects.get(slug=selected_topic_slug)
            projects_list = projects_list.filter(topics=selected_topic)
        except ProjectTopic.DoesNotExist: selected_topic_slug = ''; messages.warning(request, f"Topic filter '{selected_topic_slug}' not found.")

    # Apply sorting
    valid_sort_options = {'-date_created': '-date_created', 'date_created': 'date_created', 'title': 'title', '-title': '-title', 'order': 'order', '-order': '-order'}
    sort_key = valid_sort_options.get(sort_by, '-date_created')
    projects_list = projects_list.order_by(sort_key)

    context = {
        'page_title': 'All Projects',
        'projects': projects_list,
        'skills_list': skills_list,
        'topics_list': topics_list, # Pass topics
        'selected_skill_slug': selected_skill_slug,
        'selected_topic_slug': selected_topic_slug, # Pass topic
        'current_sort': sort_by,
        'valid_sort_options': valid_sort_options.keys(),
    }
    return render(request, 'portfolio/all_projects.html', context=context)

def project_detail(request, slug):
    """ View function for displaying a single project's details. """
    # Prefetch related topics and skills for efficiency
    project = get_object_or_404(Project.objects.prefetch_related('skills', 'topics'), slug=slug)

    meta_description = Truncator(project.description).words(25, truncate=' ...')
    keywords = [project.title, 'deep learning', 'AI', 'project']
    if hasattr(project, 'skills'): keywords.extend([skill.name for skill in project.skills.all()])
    if hasattr(project, 'topics'): keywords.extend([topic.name for topic in project.topics.all()]) # Add topics
    meta_keywords = ", ".join(list(set(keywords)))

    context = {
        'project': project,
        'page_title': f"{project.title} - Project Details",
        'meta_description': meta_description,
        'meta_keywords': meta_keywords,
        # Topics are accessible via project.topics.all in template
    }
    return render(request, 'portfolio/project_detail.html', context=context)

# --- Topic Detail View (NEW) ---
def topic_detail(request, topic_slug):
    """ Displays details for a single topic and lists associated projects. """
    topic = get_object_or_404(ProjectTopic, slug=topic_slug)
    # Get related projects using the related_name 'projects' from the Project model
    related_projects = topic.projects.prefetch_related('skills', 'topics').all()

    context = {
        'topic': topic,
        'projects': related_projects, # Pass projects related to this topic
        'page_title': f"Projects - {topic.name}",
        'meta_description': f"Projects related to {topic.name}. {topic.description[:120]}..." if topic.description else f"Projects related to {topic.name}.",
        'meta_keywords': f"{topic.name}, project, portfolio, deep learning, AI",
    }
    return render(request, 'portfolio/topic_detail.html', context=context)
# --- End Topic Detail View ---


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

# --- Search View (UPDATED) ---
def search_results_view(request):
    """ Handles searching across projects, skills, and topics. """
    query = request.GET.get('q', '')
    project_results = Project.objects.none()
    skill_results = Skill.objects.none() if Skill else Skill.objects.none()
    topic_results = ProjectTopic.objects.none() # Initialize empty queryset for topics

    if query:
        # Search Projects: title, description, skills__name, topics__name
        project_query = (
            Q(title__icontains=query) |
            Q(description__icontains=query) |
            Q(skills__name__icontains=query) | # Search related skill names
            Q(topics__name__icontains=query)    # Search related topic names
        )
        project_results = Project.objects.filter(project_query).distinct().prefetch_related('topics', 'skills')

        # Search Skills: name, description
        if Skill:
            skill_query = ( Q(name__icontains=query) | Q(description__icontains=query) )
            skill_results = Skill.objects.filter(skill_query).distinct()

        # Search Topics: name, description (NEW)
        topic_query = ( Q(name__icontains=query) | Q(description__icontains=query) )
        topic_results = ProjectTopic.objects.filter(topic_query).distinct()

    context = {
        'query': query,
        'projects': project_results,
        'skills': skill_results,
        'topics': topic_results, # Add topics to context
        'page_title': f'Search Results for "{query}"' if query else 'Search',
        'meta_description': f"Search results for '{query}' on [Your Name]'s deep learning portfolio.",
        'meta_keywords': f"search results, {query}, deep learning, AI, portfolio, topic, skill",
    }
    return render(request, 'portfolio/search_results.html', context=context)

# --- Hire Me View (NEW) ---
def hire_me_view(request):
    """ Renders the Hire Me page. """
    context = {
        'page_title': 'Hire Me',
        # Add specific meta tags if desired
        'meta_description': "Looking to hire a Deep Learning Engineer? Learn about my skills, availability, and the types of opportunities I'm seeking.",
        'meta_keywords': "hire me, deep learning engineer, machine learning engineer, AI developer, freelance AI, available for hire, [Your Name]",
    }
    # You could potentially add context here about specific services if you model them
    return render(request, 'portfolio/hire_me_page.html', context=context)