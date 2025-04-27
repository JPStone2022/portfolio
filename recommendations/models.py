# recommendations/models.py

from django.db import models
from django.utils.text import slugify
from django.urls import reverse # Import reverse

class RecommendedProduct(models.Model):
    """ Represents a recommended product (book, tool, course, etc.). """
    name = models.CharField(max_length=200, help_text="Name of the product.")
    # Add slug field
    slug = models.SlugField(max_length=220, unique=True, blank=True, help_text="URL-friendly version of the name (auto-generated).")
    description = models.TextField(help_text="Brief description or why you recommend it.")
    product_url = models.URLField(max_length=500, help_text="Link to the product (e.g., store page, affiliate link).")
    image_url = models.URLField(max_length=500, blank=True, null=True, help_text="URL for the product image.")
    category = models.CharField(max_length=100, blank=True, help_text="Optional category (e.g., Book, Course, Software, Hardware).")
    order = models.PositiveIntegerField(default=0, help_text="Order in which to display products.")

    class Meta:
        ordering = ['order', 'name'] # Default ordering

    def __str__(self):
        return self.name

    # Add get_absolute_url method
    def get_absolute_url(self):
        """ Returns the URL to access a detail page for this recommendation. """
        return reverse('recommendations:recommendation_detail', kwargs={'slug': self.slug})

    # Override save method to generate slug
    def save(self, *args, **kwargs):
        """ Auto-generate slug if one doesn't exist. """
        if not self.slug:
            self.slug = slugify(self.name)
            # Ensure slug uniqueness
            original_slug = self.slug
            counter = 1
            while RecommendedProduct.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f'{original_slug}-{counter}'
                counter += 1
        super().save(*args, **kwargs)

