# demos/models.py
from django.db import models
from django.urls import reverse
from django.utils.text import slugify

class Demo(models.Model):
    """ Represents an interactive demo showcased on the site. """
    title = models.CharField(max_length=200, help_text="The title of the demo (e.g., Image Classifier).")
    slug = models.SlugField(max_length=220, unique=True, blank=True, help_text="URL-friendly version (auto-generated).")
    description = models.TextField(help_text="A brief description explaining the demo.")
    # URL to the actual demo page (e.g., /demos/image-classifier/)
    # Use reverse() in get_absolute_url or hardcode if simpler
    demo_url_name = models.CharField(
        max_length=100,
        help_text="The URL name for the demo view (e.g., 'demos:image_classifier')."
    )
    # Optional: URL for a representative image/thumbnail
    image_url = models.URLField(max_length=500, blank=True, null=True, help_text="URL for a preview image.")
    order = models.PositiveIntegerField(default=0, help_text="Order for display on feature lists.")
    is_featured = models.BooleanField(default=True, help_text="Feature this demo on the homepage?")

    class Meta:
        ordering = ['order', 'title']

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        """ Returns the URL to the actual demo page. """
        try:
            # Assumes the demo_url_name is like 'app_name:view_name'
            return reverse(self.demo_url_name)
        except Exception:
            # Fallback if URL name is invalid or app not configured
            return "#" # Or return None or log error

    def save(self, *args, **kwargs):
        """ Auto-generate slug. """
        if not self.slug:
            self.slug = slugify(self.title)
            original_slug = self.slug
            counter = 1
            while Demo.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f'{original_slug}-{counter}'
                counter += 1
        super().save(*args, **kwargs)
