# portfolio/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils.text import slugify
from django.utils import timezone
from django.core import mail
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile # For image field testing
from django.db.models import Q

from .models import Project, Certificate, ProjectTopic # Import new model
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

try:
    from recommendations.models import RecommendedProduct
    RECOMMENDATIONS_APP_EXISTS = True
except ImportError:
    RecommendedProduct = None
    RECOMMENDATIONS_APP_EXISTS = False

try:
    from demos.models import Demo
    DEMOS_APP_EXISTS = True
except ImportError:
    Demo = None
    DEMOS_APP_EXISTS = False


# --- Model Tests ---

class ProjectTopicModelTests(TestCase):

    def test_topic_creation(self):
        """ Test basic ProjectTopic creation and defaults. """
        topic = ProjectTopic.objects.create(name="Computer Vision Test")
        self.assertEqual(str(topic), "Computer Vision Test")
        self.assertEqual(topic.order, 0)
        self.assertTrue(isinstance(topic, ProjectTopic))

    def test_topic_slug_generation(self):
        """ Test slug generation for topics. """
        topic = ProjectTopic.objects.create(name="Natural Language Processing")
        self.assertEqual(topic.slug, "natural-language-processing")

    def test_topic_get_absolute_url(self):
        """ Test get_absolute_url method for topics. """
        topic = ProjectTopic.objects.create(name="Web App Test")
        expected_url = reverse('portfolio:topic_detail', kwargs={'topic_slug': topic.slug})
        self.assertEqual(topic.get_absolute_url(), expected_url)

class CertificateModelTests(TestCase):

    def test_certificate_creation(self):
        """ Test basic certificate creation and defaults. """
        cert = Certificate.objects.create(title="Test Cert", issuer="Test Issuer")
        self.assertEqual(str(cert), "Test Cert - Test Issuer")
        self.assertEqual(cert.order, 0)
        self.assertTrue(isinstance(cert, Certificate))
        self.assertIsNone(cert.certificate_file.name) # Check file field default
        self.assertIsNone(cert.logo_image.name) # Check image field default

    def test_certificate_logo_image_field(self):
        """ Test assigning a value to the logo_image field (does not test upload). """
        # Create a dummy image file in memory
        dummy_image = SimpleUploadedFile("test_logo.png", b"file_content", content_type="image/png")
        cert = Certificate.objects.create(
            title="Cert with Logo",
            issuer="Issuer Inc.",
            logo_image=dummy_image
        )
        self.assertTrue(cert.logo_image.name.startswith('certificate_logos/test_logo'))
        # Remember to configure MEDIA_ROOT for tests if needed, or mock storage

class ProjectModelTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        if SKILL_APP_EXISTS:
            cls.skill_py = Skill.objects.create(name="Python Test")
            cls.skill_dj = Skill.objects.create(name="Django Test")
        cls.topic_cv = ProjectTopic.objects.create(name="CV Test Topic")
        cls.topic_nlp = ProjectTopic.objects.create(name="NLP Test Topic")

        cls.project = Project.objects.create(
            title="Test Project One",
            description="Description for project one.",
            results_metrics="Achieved 90% accuracy.",
            challenges="Data scarcity was a challenge.",
            lessons_learned="Preprocessing is key.",
            code_snippet="print('hello world')",
            code_language="python"
        )
        if SKILL_APP_EXISTS:
            cls.project.skills.add(cls.skill_py, cls.skill_dj)
        cls.project.topics.add(cls.topic_cv, cls.topic_nlp) # Add topics

    # --- Previous tests (keep them) ---
    def test_project_creation_and_defaults(self):
        self.assertEqual(self.project.title, "Test Project One")
        self.assertEqual(self.project.order, 0)
        # Test new fields were saved
        self.assertEqual(self.project.results_metrics, "Achieved 90% accuracy.")
        self.assertEqual(self.project.challenges, "Data scarcity was a challenge.")
        self.assertEqual(self.project.lessons_learned, "Preprocessing is key.")
        self.assertEqual(self.project.code_snippet, "print('hello world')")
        self.assertEqual(self.project.code_language, "python")

    def test_str_representation(self): self.assertEqual(str(self.project), "Test Project One")
    def test_slug_generation_on_save(self): self.assertEqual(self.project.slug, "test-project-one")
    def test_get_absolute_url(self): self.assertEqual(self.project.get_absolute_url(), f'/project/{self.project.slug}/')
    def test_skills_relationship(self):
        if SKILL_APP_EXISTS: self.assertEqual(self.project.skills.count(), 2)
        else: self.skipTest("Skills app not found.")
    def test_get_technologies_list_from_skills(self):
        if SKILL_APP_EXISTS: self.assertListEqual(sorted(self.project.get_technologies_list()), sorted(["Python Test", "Django Test"]))
        else: self.skipTest("Skills app not found.")
    # --- End Previous tests ---

    def test_topics_relationship(self):
        """ Test the ManyToMany relationship with ProjectTopics. """
        self.assertEqual(self.project.topics.count(), 2)
        self.assertIn(self.topic_cv, self.project.topics.all())
        self.assertIn(self.topic_nlp, self.project.topics.all())
        # Test reverse relationship
        self.assertIn(self.project, self.topic_cv.projects.all())


# --- Form Tests ---
class ContactFormTests(TestCase):
    # Keep previous ContactForm tests as they are...
    def test_valid_contact_form(self):
        form_data = { 'name': 'Test User', 'email': 'test@example.com', 'subject': 'Valid Subject', 'message': 'Valid message.'}
        form = ContactForm(data=form_data)
        self.assertTrue(form.is_valid())
    def test_invalid_contact_form_missing_fields(self):
        form_data = {'name': 'Test User'}
        form = ContactForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('email', form.errors)
    def test_invalid_contact_form_bad_email(self):
        form_data = { 'name': 'Test User', 'email': 'not-an-email', 'subject': 'Bad Email Test', 'message': 'Message content.'}
        form = ContactForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn('email', form.errors)
    def test_contact_form_honeypot_filled(self):
        form_data = { 'name': 'Spam Bot', 'email': 'spam@example.com', 'subject': 'Spam Subject', 'message': 'Spam message.', 'honeypot': 'I am a bot'}
        form = ContactForm(data=form_data)
        self.assertTrue(form.is_valid())
        self.assertTrue(form.cleaned_data['honeypot'])


# --- View Tests ---
class PortfolioViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.topic1 = ProjectTopic.objects.create(name="Test Topic 1", slug="test-topic-1")
        cls.topic2 = ProjectTopic.objects.create(name="Test Topic 2", slug="test-topic-2")
        if SKILL_APP_EXISTS:
            cls.skill1 = Skill.objects.create(name="Test Skill 1", slug="test-skill-1")
            cls.skill2 = Skill.objects.create(name="Test Skill 2", slug="test-skill-2")
        cls.project1 = Project.objects.create(title="View Test Project 1", description="Desc 1", results_metrics="Result 1", challenges="Challenge 1", lessons_learned="Lesson 1", code_snippet="code 1")
        cls.project2 = Project.objects.create(title="View Test Project 2", description="Desc 2", order=1)
        # Assign topics/skills
        cls.project1.topics.add(cls.topic1) # Project 1 only has Topic 1
        cls.project2.topics.add(cls.topic2) # Project 2 only has Topic 2
        if SKILL_APP_EXISTS:
            cls.project1.skills.add(cls.skill1)
            cls.project2.skills.add(cls.skill2)
        # Other models
        cls.cert1 = Certificate.objects.create(title="View Test Cert", issuer="Issuer")
        if BLOG_APP_EXISTS:
            cls.blogpost1 = BlogPost.objects.create(title="View Test Post", content="Content", status='published')
        if RECOMMENDATIONS_APP_EXISTS:
            cls.rec1 = RecommendedProduct.objects.create(name="Test Rec", product_url="http://example.com")
        if DEMOS_APP_EXISTS:
            cls.demo1 = Demo.objects.create(title="Test Demo", demo_url_name="fake:url")


    def setUp(self):
        self.client = Client()
        settings.DEFAULT_FROM_EMAIL = getattr(settings, 'DEFAULT_FROM_EMAIL', 'webmaster@localhost')
        settings.EMAIL_HOST_USER = getattr(settings, 'EMAIL_HOST_USER', 'testuser@localhost')

    def test_index_view(self):
        url = reverse('portfolio:index')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/index.html')
        self.assertIn('featured_projects', response.context)
        self.assertIn('featured_certificates', response.context)
        self.assertIn('latest_blog_post', response.context)
        self.assertIn('featured_recommendations', response.context)
        self.assertIn('featured_topics', response.context)
        self.assertIn('featured_skills', response.context)
        self.assertIn('featured_demos', response.context)
        self.assertContains(response, self.project1.title[:20])

    def test_all_projects_view_no_filters(self):
        url = reverse('portfolio:all_projects')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/all_projects.html')
        self.assertIn('projects', response.context)
        self.assertEqual(len(response.context['projects']), 2)
        self.assertContains(response, self.project1.title)
        self.assertContains(response, self.project2.title)
        self.assertIn('topics_list', response.context)
        self.assertIn('skills_list', response.context)

    def test_all_projects_view_filter_by_topic(self):
        url = reverse('portfolio:all_projects') + f'?topic={self.topic1.slug}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['projects']), 1)
        self.assertEqual(response.context['projects'][0], self.project1)
        self.assertEqual(response.context['selected_topic_slug'], self.topic1.slug)

    def test_all_projects_view_filter_by_skill(self):
        if not SKILL_APP_EXISTS: self.skipTest("Skills app not installed")
        url = reverse('portfolio:all_projects') + f'?skill={self.skill1.slug}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['projects']), 1)
        self.assertEqual(response.context['projects'][0], self.project1)
        self.assertEqual(response.context['selected_skill_slug'], self.skill1.slug)

    def test_all_projects_view_filter_combined(self):
        if not SKILL_APP_EXISTS: self.skipTest("Skills app not installed")
        url = reverse('portfolio:all_projects') + f'?topic={self.topic1.slug}&skill={self.skill1.slug}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['projects']), 1)
        self.assertEqual(response.context['projects'][0], self.project1)

    def test_all_projects_view_sorting(self):
        url = reverse('portfolio:all_projects') + '?sort=title' # Sort A-Z
        response = self.client.get(url)
        self.assertEqual(list(response.context['projects']), [self.project1, self.project2])
        url = reverse('portfolio:all_projects') + '?sort=-title' # Sort Z-A
        response = self.client.get(url)
        self.assertEqual(list(response.context['projects']), [self.project2, self.project1])

    # --- Test Project Detail View (Corrected Assertion) ---
    def test_project_detail_view(self):
        url = reverse('portfolio:project_detail', kwargs={'slug': self.project1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/project_detail.html')
        self.assertEqual(response.context['project'], self.project1)
        self.assertContains(response, self.project1.title)
        self.assertContains(response, self.project1.description)
        self.assertContains(response, self.project1.results_metrics)
        self.assertContains(response, self.project1.challenges)
        self.assertContains(response, self.project1.lessons_learned)
        self.assertContains(response, self.project1.code_snippet)
        # Check ONLY topics associated with project1
        self.assertContains(response, self.topic1.name)
        self.assertNotContains(response, self.topic2.name) # REMOVED check for topic2
        # Check skills (if app exists)
        if SKILL_APP_EXISTS:
            self.assertContains(response, self.skill1.name)
            self.assertNotContains(response, self.skill2.name) # Also check skill2 isn't present

    def test_project_detail_view_404(self):
        url = reverse('portfolio:project_detail', kwargs={'slug': 'non-existent-slug'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_topic_detail_view(self):
        url = reverse('portfolio:topic_detail', kwargs={'topic_slug': self.topic1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/topic_detail.html')
        self.assertEqual(response.context['topic'], self.topic1)
        self.assertEqual(len(response.context['projects']), 1)
        self.assertIn(self.project1, response.context['projects'])
        self.assertContains(response, self.topic1.name)

    def test_topic_detail_view_404(self):
        url = reverse('portfolio:topic_detail', kwargs={'topic_slug': 'invalid-topic'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_hire_me_view(self):
        url = reverse('portfolio:hire_me')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/hire_me_page.html')
        self.assertContains(response, "Work With Me")

    def test_search_results_view_with_topic_query(self):
        url = reverse('portfolio:search_results') + '?q=Topic'
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('projects', response.context)
        self.assertIn('skills', response.context)
        self.assertIn('topics', response.context)
        self.assertIn(self.project1, response.context['projects'])
        self.assertIn(self.project2, response.context['projects'])
        self.assertIn(self.topic1, response.context['topics'])
        self.assertIn(self.topic2, response.context['topics'])

    # --- Keep other view tests ---
    def test_certificates_view(self):
        url = reverse('portfolio:certificates')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/certificates.html')
        self.assertContains(response, self.cert1.title)
    def test_about_me_view(self):
        url = reverse('portfolio:about_me')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/about_me_page.html')
    def test_cv_view(self):
        url = reverse('portfolio:cv')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/cv_page.html')
    def test_contact_view_get(self):
        url = reverse('portfolio:contact')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertIsInstance(response.context['form'], ContactForm)
    def test_contact_view_post_success(self):
        url = reverse('portfolio:contact')
        form_data = {'name': 'Test User', 'email': 'test@example.com', 'subject': 'Valid Subject', 'message': 'Valid message.'}
        with self.settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend'): response = self.client.post(url, form_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertContains(response, 'Message sent successfully! Thank you.')
        self.assertEqual(len(mail.outbox), 1)
    def test_contact_view_post_invalid(self):
        url = reverse('portfolio:contact')
        form_data = {'name': 'Test User'}
        response = self.client.post(url, form_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertFormError(response.context['form'], 'email', 'This field is required.')
    def test_contact_view_post_honeypot(self):
        url = reverse('portfolio:contact')
        form_data = {'name': 'Spammer', 'email': 'spam@example.com', 'subject': 'Spam', 'message': 'Spam message.', 'honeypot': 'gotcha'}
        response = self.client.post(url, form_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'portfolio/contact_page.html')
        self.assertContains(response, 'Spam detected')
        self.assertEqual(len(mail.outbox), 0)
    def test_search_results_view_get(self):
        url = reverse('portfolio:search_results')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['query'], '')

