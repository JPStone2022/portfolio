# recommendations/models.py

from django.db import models
from django.utils.text import slugify

class RecommendedProduct(models.Model):
    """ Represents a recommended product (book, tool, course, etc.). """
    name = models.CharField(max_length=200, help_text="Name of the product.")
    description = models.TextField(help_text="Brief description or why you recommend it.")
    product_url = models.URLField(max_length=500, help_text="Link to the product (e.g., store page, affiliate link).")
    # Using URLField for image simplicity, could use ImageField for uploads
    image_url = models.URLField(max_length=500, blank=True, null=True, help_text="URL for the product image.")
    category = models.CharField(max_length=100, blank=True, help_text="Optional category (e.g., Book, Course, Software, Hardware).")
    order = models.PositiveIntegerField(default=0, help_text="Order in which to display products.")

    class Meta:
        ordering = ['order', 'name'] # Default ordering

    def __str__(self):
        return self.name

    # Optional: Add get_absolute_url if you plan a detail page per product later
    # def get_absolute_url(self):
    #     # return reverse('recommendations:product_detail', args=[self.slug]) # If using slugs
    #     pass
