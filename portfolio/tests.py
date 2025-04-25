# portfolio/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils.text import slugify
from django.utils import timezone
from django.core import mail # To test email sending output
from django.conf import settings

from .models import Project, Certificate
from .forms import ContactForm

# Import models from other apps safely
try:
    from skills.models import Skill
    SKILL_APP_EXISTS = True
except ImportError:
    Skill = None
    SKILL_APP_EXISTS = False

try:
    from blog.models import BlogPost
    BLOG_APP_EXISTS = True
except ImportError:
    BlogPost = None
    BLOG_APP_EXISTS = False


# --- Model Tests ---

class CertificateModelTests(TestCase):

    def test_certificate_creation(self):
        """ Test basic certificate creation and defaults. """
        cert = Certificate.objects.create(title="Test Cert", issuer="Test Issuer")
        self.assertEqual(str(cert), "Test Cert - Test Issuer")
        self.assertEqual(cert.order, 0)
        self.assertTrue(isinstance(cert, Certificate))

class ProjectModelTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods in this class
        if SKILL_APP_EXISTS:
            cls.skill_py = Skill.objects.create(name="Python Test")
            cls.skill_dj = Skill.objects.create(name="Django Test")

        cls.project = Project.objects.create(
            title="Test Project One",
            description="Description for project one."
        )
        if SKILL_APP_EXISTS:
            cls.project.skills.add(cls.skill_py, cls.skill_dj)

    def test_project_creation_and_defaults(self):
        """ Test basic project creation and field defaults. """
        self.assertEqual(self.project.title, "Test Project One")
        self.assertTrue(isinstance(self.project, Project))
        self.assertEqual(self.project.order, 0)
        self.assertEqual(self.project.results_metrics, "") # Check new fields default
        self.assertEqual(self.project.challenges, "")
        self.assertEqual(self.project.lessons_learned, "")

    def test_str_representation(self):
        """ Test the __str__ method. """
        self.assertEqual(str(self.project), "Test Project One")

    def test_slug_generation_on_save(self):
        """ Test if slug is auto-generated correctly. """
        expected_slug = slugify("Test Project One")
        self.assertEqual(self.project.slug, expected_slug)
        # Test uniqueness handling (requires saving another with same title first)
        project2 = Project.objects.create(title="Test Project One", description="Second")
        self.assertNotEqual(self.project.slug, project2.slug)
        self.assertTrue(project2.slug.startswith(expected_slug + '-'))

    def test_get_absolute_url(self):
        """ Test the get_absolute_url method. """
        expected_url = reverse('portfolio:project_detail', kwargs={'slug': self.project.slug})
        self.assertEqual(self.project.get_absolute_url(), expected_url)

    def test_skills_relationship(self):
        """ Test the ManyToMany relationship with Skills. """
        if SKILL_APP_EXISTS:
            self.assertEqual(self.project.skills.count(), 2)
            self.assertIn(self.skill_py, self.project.skills.all())
        else:
             self.skipTest("Skills app not found or Skill model import failed.")

    def test_get_technologies_list_from_skills(self):
        """ Test get_technologies_list prioritizes skills field. """
        if SKILL_APP_EXISTS:
             expected_skills = sorted(["Python Test", "Django Test"])
             self.assertListEqual(sorted(self.project.get_technologies_list()), expected_skills)
        else:
             self.skipTest("Skills app not found or Skill model import failed.")

# --- Form Tests ---

class ContactFormTests(TestCase):

    def test_valid_contact_form(self):
        """ Test the contact form with valid data. """
        form_data = {
            'name': 'Test User', 'email': 'test@example.com',
            'subject': 'Valid Subject', 'message': 'Valid message.'
        }
        form = ContactForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_invalid_contact_form_missing_fields(self):
        """ Test the contact form with missing required fields. """
        form_data = {'name': 'Test User'} # Missing email, subject, message
        form = ContactForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('email', form.errors)
        self.assertIn('subject', form.errors)
        self.assertIn('message', form.errors)

    def test_invalid_contact_form_bad_email(self):
        """ Test the contact form with an invalid email format. """
        form_data = {
            'name': 'Test User', 'email': 'not-an-email',
            'subject': 'Bad Email Test', 'message': 'Message content.'
        }
        form = ContactForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('email', form.errors)

    def test_contact_form_honeypot_filled(self):
        """ Test the contact form with the honeypot field filled. """
        form_data = {
            'name': 'Spam Bot', 'email': 'spam@example.com',
            'subject': 'Spam Subject', 'message': 'Spam message.',
            'honeypot': 'I am a bot' # Honeypot field filled
        }
        form = ContactForm(data=form_data)
        # The form itself is valid, but the view should catch the honeypot
        self.assertTrue(form.is_valid())
        self.assertTrue(form.cleaned_data['honeypot'])


# --- View Tests ---

class PortfolioViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        # Create data needed for view tests
        cls.project1 = Project.objects.create(title="View Test Project 1", description="Desc 1")
        cls.cert1 = Certificate.objects.create(title="View Test Cert", issuer="Issuer")
        if BLOG_APP_EXISTS:
            cls.blogpost1 = BlogPost.objects.create(title="View Test Post", content="Content", status='published')
        if SKILL_APP_EXISTS:
             cls.skill_view_test = Skill.objects.create(name="View Test Skill")
             cls.project1.skills.add(cls.skill_view_test)


    def setUp(self):
        # Setup the client for each test
        self.client = Client()
        # Ensure required settings have fallback values for tests
        # These might be overridden by actual settings if they exist
        settings.DEFAULT_FROM_EMAIL = getattr(settings, 'DEFAULT_FROM_EMAIL', 'webmaster@localhost')
        settings.EMAIL_HOST_USER = getattr(settings, 'EMAIL_HOST_USER', 'testuser@localhost')


    def test_index_view(self):
        """ Test index view status, template, and basic context. """
        url = reverse('portfolio:index')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/index.html')
        self.assertTemplateUsed(response, 'portfolio/base.html') # Check base template too
        self.assertIn('featured_projects', response.context)
        self.assertIn('featured_certificates', response.context)
        self.assertIn('latest_blog_post', response.context)
        # Check if specific objects appear (adjust based on slicing in view)
        self.assertContains(response, self.project1.title[:20]) # Check partial title

    def test_all_projects_view(self):
        """ Test all projects view status and template. """
        url = reverse('portfolio:all_projects')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/all_projects.html')
        self.assertIn('projects', response.context)
        self.assertContains(response, self.project1.title)

    def test_project_detail_view(self):
        """ Test project detail view status, template, and context. """
        url = reverse('portfolio:project_detail', kwargs={'slug': self.project1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/project_detail.html')
        self.assertEqual(response.context['project'], self.project1)
        self.assertContains(response, self.project1.title)
        self.assertContains(response, self.project1.description)

    def test_project_detail_view_404(self):
        """ Test project detail view returns 404 for invalid slug. """
        url = reverse('portfolio:project_detail', kwargs={'slug': 'non-existent-slug'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_certificates_view(self):
        """ Test certificates view status and template. """
        url = reverse('portfolio:certificates')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/certificates.html')
        self.assertIn('certificates', response.context)
        self.assertContains(response, self.cert1.title)

    def test_about_me_view(self):
        """ Test about me view status and template. """
        url = reverse('portfolio:about_me')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/about_me_page.html')

    def test_cv_view(self):
        """ Test CV view status and template. """
        url = reverse('portfolio:cv')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/cv_page.html')

    def test_contact_view_get(self):
        """ Test contact view GET request. """
        url = reverse('portfolio:contact')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertIsInstance(response.context['form'], ContactForm)

    def test_contact_view_post_success(self):
        """ Test contact view POST with valid data (mocking email). """
        url = reverse('portfolio:contact')
        form_data = {
            'name': 'Test User', 'email': 'test@example.com',
            'subject': 'Valid Subject', 'message': 'Valid message.'
        }

        # *** Use locmem backend for reliable outbox testing ***
        with self.settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend'):
            response = self.client.post(url, form_data, follow=True) # follow=True to check redirected page

        self.assertEqual(response.status_code, 200) # Should redirect back to contact page
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertContains(response, 'message has been sent successfully') # Check success message

        # *** This assertion should now pass ***
        self.assertEqual(len(mail.outbox), 1) # Check that one email was stored in the outbox

        # These assertions will only run if the previous one passes
        self.assertIn(form_data['subject'], mail.outbox[0].subject)
        self.assertIn(form_data['message'], mail.outbox[0].body)
        # Check recipient (ensure EMAIL_HOST_USER was set correctly in setUp or settings)
        self.assertEqual(mail.outbox[0].to, [settings.EMAIL_HOST_USER])
        # Check sender (ensure DEFAULT_FROM_EMAIL was set correctly in setUp or settings)
        self.assertEqual(mail.outbox[0].from_email, settings.DEFAULT_FROM_EMAIL)


    def test_contact_view_post_invalid(self):
        """ Test contact view POST with invalid data. """
        url = reverse('portfolio:contact')
        form_data = {'name': 'Test User'} # Missing fields
        response = self.client.post(url, form_data) # POST request

        self.assertEqual(response.status_code, 200) # Stays on the same page (doesn't redirect)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html') # Re-renders the contact page

        # Check for errors on the 'form' object within the response context
        self.assertFormError(response.context['form'], 'email', 'This field is required.')
        self.assertFormError(response.context['form'], 'subject', 'This field is required.')
        self.assertFormError(response.context['form'], 'message', 'This field is required.')
        # Check that the message framework message is present
        self.assertContains(response, 'Please correct the errors below')

    def test_contact_view_post_honeypot(self):
        """ Test contact view POST with honeypot filled. """
        url = reverse('portfolio:contact')
        form_data = {
            'name': 'Spammer', 'email': 'spam@example.com',
            'subject': 'Spam', 'message': 'Spam message.',
            'honeypot': 'gotcha' # Honeypot filled
        }
        response = self.client.post(url, form_data, follow=True) # Follow the redirect

        self.assertEqual(response.status_code, 200) # Should redirect back to contact page
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertContains(response, 'Spam detected') # Check spam message from messages framework
        self.assertEqual(len(mail.outbox), 0) # Ensure no email was sent

    def test_search_results_view_get(self):
        """ Test search results view GET request (no query). """
        url = reverse('portfolio:search_results')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/search_results.html')
        self.assertIn('query', response.context)
        self.assertEqual(response.context['query'], '')

    def test_search_results_view_with_query(self):
        """ Test search results view with a query term. """
        url = reverse('portfolio:search_results') + '?q=Test' # Add query parameter
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/search_results.html')
        self.assertEqual(response.context['query'], 'Test')
        self.assertIn('projects', response.context)
        self.assertIn('skills', response.context)
        # Check if project1 is in results (assuming 'Test' matches title/desc)
        self.assertIn(self.project1, response.context['projects'])
        # Check if skill is in results
        if SKILL_APP_EXISTS:
             self.assertIn(self.skill_view_test, response.context['skills'])


