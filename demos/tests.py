# demos/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils.text import slugify
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock
import numpy as np

# Import models and forms from the current app
from .models import Demo
from .forms import ImageUploadForm, SentimentAnalysisForm # Import both forms

# --- Mock Objects (Keep as before) ---
class MockPredictionResult:
    def __init__(self, label, score): self.label = label; self.score = score
    def __getitem__(self, key): return getattr(self, key)
MOCK_SENTIMENT_RESULT = [{'label': 'POSITIVE', 'score': 0.998}]
MOCK_IMAGE_PREDICTIONS = [('mock_id_1', 'mock_label_1', 0.95), ('mock_id_2', 'mock_label_2', 0.04)]
# Tiny valid GIF header/content
MINIMAL_GIF_CONTENT = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"


class DemoModelTests(TestCase):
    # ... (model tests remain the same) ...
    def test_demo_creation(self):
        demo = Demo.objects.create(title="Test Image Demo", description="A test demo.", demo_url_name="demos:image_classifier")
        self.assertEqual(str(demo), "Test Image Demo"); self.assertTrue(demo.is_featured); self.assertEqual(demo.order, 0)
    def test_demo_slug_generation(self):
        demo = Demo.objects.create(title="Sentiment Test", demo_url_name="demos:sentiment_analyzer")
        self.assertEqual(demo.slug, "sentiment-test")
    def test_demo_get_absolute_url(self):
        demo = Demo.objects.create(title="Image Test", demo_url_name="demos:image_classifier")
        self.assertEqual(demo.get_absolute_url(), reverse("demos:image_classifier"))
    def test_demo_get_absolute_url_invalid_name(self):
        demo = Demo.objects.create(title="Invalid Test", demo_url_name="invalid:name")
        self.assertEqual(demo.get_absolute_url(), "#")


class DemoViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.img_demo = Demo.objects.create(title="Image Classifier", slug="image-classifier", demo_url_name="demos:image_classifier")
        cls.sent_demo = Demo.objects.create(title="Sentiment Analyzer", slug="sentiment-analyzer", demo_url_name="demos:sentiment_analyzer")

    def setUp(self):
        self.client = Client()

    # --- Image Classifier Tests ---
    def test_image_classifier_view_get(self):
        """ Test GET request for image classifier. """
        url = reverse('demos:image_classifier')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/image_classification_demo.html')
        self.assertIn('form', response.context)
        self.assertIsInstance(response.context['form'], ImageUploadForm)
        self.assertIsNone(response.context.get('prediction_results'))

    @patch('demos.views.image_model.predict')
    @patch('demos.views.decode_predictions')
    def test_image_classifier_view_post_success(self, mock_decode, mock_predict):
        """ Test POST request with valid image upload, mocking the ML model. """
        mock_predict.return_value = np.array([[0.1, 0.95, 0.05]])
        mock_decode.return_value = [[('mock_imagenet_id', 'mock_label', 0.95)]]

        url = reverse('demos:image_classifier')
        # *** Use minimal valid GIF content for the dummy file ***
        dummy_image = SimpleUploadedFile("test.gif", MINIMAL_GIF_CONTENT, content_type="image/gif")
        form_data = {'image': dummy_image}

        response = self.client.post(url, form_data)

        # Check form validity first
        form_in_context = response.context.get('form')
        self.assertTrue(form_in_context.is_valid(), msg=f"Form expected to be valid but had errors: {form_in_context.errors if form_in_context else 'No form found'}")

        # Now check other aspects
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/image_classification_demo.html')
        mock_predict.assert_called_once()
        mock_decode.assert_called_once()
        self.assertIn('prediction_results', response.context)
        self.assertIsNotNone(response.context['prediction_results'])
        self.assertEqual(len(response.context['prediction_results']), 1)
        self.assertEqual(response.context['prediction_results'][0]['label'], 'mock label')
        self.assertAlmostEqual(response.context['prediction_results'][0]['probability'], 95.0)
        self.assertIn('uploaded_image_url', response.context)
        # Check if the URL starts correctly (might end differently due to filename sanitization)
        self.assertTrue(response.context['uploaded_image_url'].startswith('/media/temp_demos/test'))

    def test_image_classifier_view_post_invalid_form(self):
        """ Test POST request with no image file. """
        url = reverse('demos:image_classifier')
        response = self.client.post(url, {}) # Empty data
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/image_classification_demo.html')
        # Check the form instance from the context
        form_in_context = response.context.get('form')
        self.assertIsNotNone(form_in_context)
        self.assertFalse(form_in_context.is_valid()) # Ensure form is invalid
        self.assertIn('image', form_in_context.errors) # Check 'image' field has errors
        self.assertFormError(form_in_context, 'image', 'This field is required.') # Check specific error
        self.assertIsNone(response.context.get('prediction_results'))

    # --- Sentiment Analyzer Tests ---
    def test_sentiment_analyzer_view_get(self):
        """ Test GET request for sentiment analyzer. """
        url = reverse('demos:sentiment_analyzer')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/sentiment_analysis_demo.html')
        self.assertIn('form', response.context)
        self.assertIsNone(response.context.get('sentiment_result'))

    @patch('demos.views.sentiment_pipeline')
    def test_sentiment_analyzer_view_post_success(self, mock_pipeline):
        """ Test POST request with valid text, mocking the ML pipeline. """
        mock_pipeline.return_value = [{'label': 'POSITIVE', 'score': 0.9987}]
        url = reverse('demos:sentiment_analyzer')
        test_text = "This is a wonderful test!"
        form_data = {'text_input': test_text}
        response = self.client.post(url, form_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/sentiment_analysis_demo.html')
        mock_pipeline.assert_called_once_with(test_text)
        self.assertIn('sentiment_result', response.context)
        self.assertEqual(response.context['sentiment_result']['label'], 'POSITIVE')
        self.assertAlmostEqual(response.context['sentiment_result']['score'], 99.9)
        self.assertEqual(response.context['submitted_text'], test_text)

    def test_sentiment_analyzer_view_post_invalid_form(self):
        """ Test POST request with empty text input. """
        url = reverse('demos:sentiment_analyzer')
        response = self.client.post(url, {}) # Empty data
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'demos/sentiment_analysis_demo.html')
        # Check the form instance from the context
        form_in_context = response.context.get('form')
        self.assertIsNotNone(form_in_context)
        self.assertFalse(form_in_context.is_valid()) # Ensure form is invalid
        self.assertIn('text_input', form_in_context.errors) # Check 'text_input' field has errors
        self.assertFormError(form_in_context, 'text_input', 'This field is required.')
        self.assertIsNone(response.context.get('sentiment_result'))

