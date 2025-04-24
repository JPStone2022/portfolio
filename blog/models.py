from django.db import models
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse

# Create your models here.
# New BlogPost Model
class BlogPost(models.Model):
    """
    Represents a single blog post.
    """
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=250, unique=True, blank=True, help_text="URL-friendly version of the title (auto-generated).")
    # Optional: Link to author (if you have user accounts)
    # author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='blog_posts')
    content = models.TextField(help_text="The main content of the blog post (can use Markdown or HTML).")
    published_date = models.DateTimeField(default=timezone.now, help_text="The date and time the post was published.")
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    # Optional: Status field for drafts
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='published')

    class Meta:
        ordering = ['-published_date'] # Order by most recent first

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        # Define this if you create a detail view for blog posts later
        # return reverse('portfolio:blog_post_detail', kwargs={'slug': self.slug})
        pass # Placeholder for now

    def save(self, *args, **kwargs):
        """Auto-generate slug."""
        if not self.slug:
            self.slug = slugify(self.title)
            # Ensure slug uniqueness
            original_slug = self.slug
            counter = 1
            while BlogPost.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f'{original_slug}-{counter}'
                counter += 1
        super().save(*args, **kwargs)