# demos/sitemaps.py

from django.contrib.sitemaps import Sitemap
# Import your Demo model
from .models import Demo

class DemoSitemap(Sitemap):
    """ Sitemap for individual demo pages. """
    changefreq = "monthly"  # How often demo pages might change
    priority = 0.6         # Priority relative to other pages

    def items(self):
        """ Returns a queryset of all Demo objects to include. """
        return Demo.objects.all()

    # No need to define location(self, obj) because the Demo model
    # already has a get_absolute_url() method which the sitemap
    # framework will use automatically.

    # Optional: Add lastmod if your Demo model has an update timestamp
    # def lastmod(self, obj):
    #     return obj.updated_date
