# topics/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils.text import slugify

# Import models from THIS app
from .models import ProjectTopic
# Import models from OTHER apps safely for relationship testing
try:
    from portfolio.models import Project
    PORTFOLIO_APP_EXISTS = True
except ImportError:
    Project = None
    PORTFOLIO_APP_EXISTS = False

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
        # Test uniqueness
        topic2 = ProjectTopic.objects.create(name="Web Development & APIs")
        self.assertNotEqual(topic.slug, topic2.slug)
        self.assertTrue(topic2.slug.startswith("web-development-apis"))

    def test_topic_get_absolute_url(self):
        """ Test get_absolute_url method for topics. """
        topic = ProjectTopic.objects.create(name="Web App Test")
        # Use the 'topics' namespace defined in topics/urls.py
        expected_url = reverse('topics:topic_detail', kwargs={'topic_slug': topic.slug})
        self.assertEqual(topic.get_absolute_url(), expected_url)

# --- View Tests ---

class TopicViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.topic1 = ProjectTopic.objects.create(name="Topic Alpha", slug="topic-alpha", description="Desc Alpha")
        cls.topic2 = ProjectTopic.objects.create(name="Topic Beta", slug="topic-beta", order=1)
        if PORTFOLIO_APP_EXISTS:
            cls.project1 = Project.objects.create(title="Project A")
            cls.project2 = Project.objects.create(title="Project B")
            cls.project1.topics.add(cls.topic1) # Link project A to topic 1
            cls.project2.topics.add(cls.topic1, cls.topic2) # Link project B to both

    def setUp(self):
        self.client = Client()

    def test_topic_list_view(self):
        """ Test the topic list view status, template, and context. """
        url = reverse('topics:topic_list') # Use topics namespace
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'topics/topic_list.html')
        self.assertIn('topics', response.context)
        self.assertEqual(len(response.context['topics']), 2)
        # Check order based on model Meta (order, then name)
        self.assertEqual(response.context['topics'][0], self.topic1)
        self.assertEqual(response.context['topics'][1], self.topic2)
        self.assertContains(response, "Topic Alpha")
        self.assertContains(response, "Topic Beta")

    def test_topic_detail_view(self):
        """ Test the topic detail view status, template, and context. """
        url = reverse('topics:topic_detail', kwargs={'topic_slug': self.topic1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'topics/topic_detail.html')
        self.assertEqual(response.context['topic'], self.topic1)
        self.assertContains(response, "Topic Alpha Projects") # Check heading
        self.assertContains(response, "Desc Alpha") # Check description

        # Check related projects are passed correctly
        self.assertIn('projects', response.context)
        if PORTFOLIO_APP_EXISTS:
            self.assertEqual(len(response.context['projects']), 2) # Both projects linked to topic1
            self.assertIn(self.project1, response.context['projects'])
            self.assertIn(self.project2, response.context['projects'])
            self.assertContains(response, "Project A")
            self.assertContains(response, "Project B")
        else:
             self.assertIsNone(response.context['projects']) # Or check for empty list depending on view logic

    def test_topic_detail_view_404(self):
        """ Test topic detail view returns 404 for invalid slug. """
        url = reverse('topics:topic_detail', kwargs={'topic_slug': 'invalid-topic'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

