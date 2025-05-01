# recommendations/tests.py

from django.test import TestCase, Client, RequestFactory
from django.urls import reverse
from django.utils.text import slugify

from .models import RecommendedProduct
from .context_processors import recommendation_context # Import context processor

class RecommendedProductModelTests(TestCase):

    def test_recommendation_creation(self):
        """ Test basic RecommendedProduct creation and defaults. """
        rec = RecommendedProduct.objects.create(
            name="Test Book",
            description="A great book.",
            product_url="http://example.com/book"
        )
        self.assertEqual(str(rec), "Test Book")
        self.assertEqual(rec.order, 0)
        self.assertEqual(rec.category, "")
        self.assertTrue(isinstance(rec, RecommendedProduct))

    def test_recommendation_slug_generation(self):
        """ Test slug generation. """
        rec = RecommendedProduct.objects.create(name="Another Tool", product_url="http://example.com/tool")
        self.assertEqual(rec.slug, "another-tool")

    def test_recommendation_get_absolute_url(self):
        """ Test get_absolute_url method. """
        rec = RecommendedProduct.objects.create(name="Course Name", product_url="http://example.com/course")
        expected_url = reverse('recommendations:recommendation_detail', kwargs={'slug': rec.slug})
        self.assertEqual(rec.get_absolute_url(), expected_url)

class RecommendationViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.rec1 = RecommendedProduct.objects.create(name="Rec 1", slug="rec-1", product_url="http://example.com/1", description="Desc 1")
        cls.rec2 = RecommendedProduct.objects.create(name="Rec 2", slug="rec-2", product_url="http://example.com/2", description="Desc 2", order=1)

    def setUp(self):
        self.client = Client()

    def test_recommendation_list_view(self):
        """ Test the recommendation list view. """
        url = reverse('recommendations:recommendation_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'recommendations/recommendation_list.html')
        self.assertIn('recommendations', response.context)
        self.assertEqual(len(response.context['recommendations']), 2)
        self.assertContains(response, self.rec1.name)
        self.assertContains(response, self.rec2.name)

    def test_recommendation_detail_view(self):
        """ Test the recommendation detail view. """
        url = reverse('recommendations:recommendation_detail', kwargs={'slug': self.rec1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'recommendations/recommendation_detail.html')
        self.assertEqual(response.context['product'], self.rec1)
        self.assertContains(response, self.rec1.name)
        self.assertContains(response, self.rec1.description)

    def test_recommendation_detail_view_404(self):
        """ Test detail view returns 404 for invalid slug. """
        url = reverse('recommendations:recommendation_detail', kwargs={'slug': 'invalid-rec'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

class RecommendationContextProcessorTests(TestCase):

    def test_recommendation_count_context_processor(self):
        """ Test that the recommendation count is added to context. """
        # Test with no recommendations
        factory = RequestFactory()
        request = factory.get('/') # Dummy request needed for context processor
        context = recommendation_context(request)
        self.assertIn('recommendation_count', context)
        self.assertEqual(context['recommendation_count'], 0)

        # Test with recommendations
        RecommendedProduct.objects.create(name="Test Rec 1", product_url="http://example.com")
        RecommendedProduct.objects.create(name="Test Rec 2", product_url="http://example.com")
        context = recommendation_context(request)
        self.assertEqual(context['recommendation_count'], 2)

    def test_recommendation_count_in_template(self):
         """ Test if count is accessible in a rendered template via client. """
         # Test with no recommendations
         response = self.client.get(reverse('portfolio:index')) # Request any page
         self.assertEqual(response.context['recommendation_count'], 0)

         # Test with recommendations
         RecommendedProduct.objects.create(name="Test Rec 1", product_url="http://example.com")
         response = self.client.get(reverse('portfolio:index'))
         self.assertEqual(response.context['recommendation_count'], 1)

