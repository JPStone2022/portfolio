# portfolio/views.py

from django.shortcuts import render, get_object_or_404, redirect
# Import BlogPost model
from .models import Project, Certificate
from blog.models import BlogPost
from django.utils import timezone # Needed for filtering published posts
from django.contrib import messages # Import messages framework
from .forms import ContactForm # Import the new form
from django.core.mail import send_mail # Uncomment if implementing email sending
from django.conf import settings # Uncomment if implementing email sending
from django.urls import reverse

def index(request):
    """
    View function for the home page of the site.
    Fetches featured projects, certificates, and the latest blog post.
    """
    featured_projects = Project.objects.all()[:6]
    featured_certificates = Certificate.objects.all()[:3]

    latest_blog_post = BlogPost.objects.filter(
        status='published',
        published_date__lte=timezone.now()
    ).order_by('-published_date').first()

    context = {
        'page_title': 'My Deep Learning Portfolio',
        'featured_projects': featured_projects,
        'featured_certificates': featured_certificates,
        'latest_blog_post': latest_blog_post,
    }
    return render(request, 'portfolio/index.html', context=context)

# ... (other views: project_detail, certificates_view, all_projects_view) ...
def project_detail(request, slug):
    project = get_object_or_404(Project, slug=slug)
    context = { 'project': project, 'page_title': f"{project.title} - Project Details" }
    return render(request, 'portfolio/project_detail.html', context=context)

def certificates_view(request):
    certificates = Certificate.objects.all()
    context = { 'page_title': 'My Certificates', 'certificates': certificates, }
    return render(request, 'portfolio/certificates.html', context=context)

def all_projects_view(request):
    projects = Project.objects.all()
    context = { 'page_title': 'All Projects', 'projects': projects, }
    return render(request, 'portfolio/all_projects.html', context=context)


# Add the contact view
def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Basic spam check (honeypot)
            if form.cleaned_data['honeypot']:
                 messages.error(request, "Spam detected.")
                 return redirect(reverse('portfolio:contact')) # Redirect back

            # Form is valid and not spam, process the data
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            subject = form.cleaned_data['subject']
            message_body = form.cleaned_data['message']

            # --- Placeholder for sending email or saving to DB ---
            # Example using send_mail (requires email backend setup in settings.py):
            try:
                send_mail(
                    f"Contact Form Submission: {subject}",
                    f"Name: {name}\nEmail: {email}\n\nMessage:\n{message_body}",
                    settings.DEFAULT_FROM_EMAIL, # Or your 'from' address
                    ['braingymstoke@gmail.com'], # List of recipients
                    fail_silently=False,
                )
                messages.success(request, 'Your message has been sent successfully! Thank you.')
            except Exception as e:
                messages.error(request, f'Sorry, there was an error sending your message: {e}')

            # For now, just show a success message
            messages.success(request, f'Thank you, {name}! Your message regarding "{subject}" has been received (simulation).')
            # --- End Placeholder ---

            # Redirect to the same page after successful POST to prevent resubmission
            return redirect(reverse('portfolio:contact'))
        else:
            # Form is invalid, show errors
            messages.error(request, 'Please correct the errors below.')
    else:
        # GET request, show an empty form
        form = ContactForm()

    context = {
        'form': form,
        'page_title': 'Contact Me',
    }
    return render(request, 'portfolio/contact_page.html', context)

# --- About Me View (NEW) ---
def about_me_view(request):
    """ Renders the detailed About Me page. """
    context = {
        'page_title': 'About Me',
    }
    # Add any specific context needed for this page here
    # e.g., fetch specific achievements or data points
    return render(request, 'portfolio/about_me_page.html', context=context)

