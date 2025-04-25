# blog/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify
from .models import BlogPost

class BlogPostModelTests(TestCase):

    def test_blog_post_creation(self):
        """ Test basic BlogPost creation and defaults. """
        post = BlogPost.objects.create(title="My First Blog Post", content="Some content.")
        self.assertEqual(str(post), "My First Blog Post")
        self.assertEqual(post.status, 'published') # Check default status
        self.assertTrue(isinstance(post, BlogPost))
        self.assertTrue(post.published_date <= timezone.now()) # Check default date

    def test_slug_generation_on_save(self):
        """ Test slug generation for blog posts. """
        post = BlogPost.objects.create(title="Another Post Title", content="Content")
        expected_slug = slugify("Another Post Title")
        self.assertEqual(post.slug, expected_slug)

    # Add test for get_absolute_url if you implement it later

class BlogViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.now = timezone.now()
        cls.post1 = BlogPost.objects.create(
            title="Published Post 1", content="Content 1", status='published', published_date=cls.now - timezone.timedelta(days=1)
        )
        cls.post2 = BlogPost.objects.create(
            title="Published Post 2", content="Content 2", status='published', published_date=cls.now - timezone.timedelta(hours=1)
        )
        cls.draft_post = BlogPost.objects.create(
            title="Draft Post", content="Draft Content", status='draft'
        )
        cls.future_post = BlogPost.objects.create(
            title="Future Post", content="Future Content", status='published', published_date=cls.now + timezone.timedelta(days=1)
        )

    def setUp(self):
        self.client = Client()

    def test_blog_post_list_view(self):
        """ Test the blog post list view. """
        url = reverse('blog:blog_post_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'blog/blog_list.html')
        self.assertIn('posts', response.context)
        # Check that only published, past posts are shown, ordered correctly
        self.assertEqual(len(response.context['posts']), 2)
        self.assertEqual(response.context['posts'][0], self.post2) # Most recent published
        self.assertEqual(response.context['posts'][1], self.post1)
        self.assertContains(response, self.post1.title)
        self.assertNotContains(response, self.draft_post.title)
        self.assertNotContains(response, self.future_post.title)

    def test_blog_post_detail_view(self):
        """ Test the blog post detail view for a published post. """
        url = reverse('blog:blog_post_detail', kwargs={'slug': self.post1.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'blog/blog_detail.html')
        self.assertEqual(response.context['post'], self.post1)
        self.assertContains(response, self.post1.title)
        self.assertContains(response, self.post1.content)

    def test_blog_post_detail_view_404_draft(self):
        """ Test that accessing a draft post detail view returns 404. """
        url = reverse('blog:blog_post_detail', kwargs={'slug': self.draft_post.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_blog_post_detail_view_404_future(self):
        """ Test that accessing a future post detail view returns 404. """
        url = reverse('blog:blog_post_detail', kwargs={'slug': self.future_post.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    def test_blog_post_detail_view_404_invalid_slug(self):
        """ Test that accessing a non-existent slug returns 404. """
        url = reverse('blog:blog_post_detail', kwargs={'slug': 'invalid-slug'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

