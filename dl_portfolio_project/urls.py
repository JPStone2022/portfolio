# dl_portfolio_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
# Import sitemaps view and your sitemap classes
from django.contrib.sitemaps.views import sitemap
from portfolio.sitemaps import StaticViewSitemap, ProjectSitemap
from blog.sitemaps import BlogStaticSitemap, BlogPostSitemap
from skills.sitemaps import SkillsStaticSitemap, SkillSitemap
# Import TemplateView for robots.txt
from django.views.generic import TemplateView

# Combine all sitemaps into a dictionary
sitemaps = {
    'static': StaticViewSitemap,
    'projects': ProjectSitemap,
    'blogstatic': BlogStaticSitemap,
    'blogposts': BlogPostSitemap,
    'skillsstatic': SkillsStaticSitemap,
    'skills': SkillSitemap,
}

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('skills/', include('skills.urls', namespace='skills')),
    path('recommendations/', include('recommendations.urls', namespace='recommendations')), # Include recommendations URLs
    path('demos/', include('demos.urls', namespace='demos')), # Include demos URLs

    # Add the sitemap URL pattern
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps},
            name='django.contrib.sitemaps.views.sitemap'),

    # Add the robots.txt URL pattern using TemplateView
    path(
        'robots.txt',
        TemplateView.as_view(template_name="robots.txt", content_type="text/plain")
    ),

    # Keep portfolio URLs at the root (should be last for catch-all)
    path('', include('portfolio.urls')),
]

# Media file serving for development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

