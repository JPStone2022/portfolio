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
from demos.sitemaps import DemoSitemap
# Import TemplateView for robots.txt
from django.views.generic import TemplateView

# # Combine all sitemaps into a dictionary
# sitemaps = {
#     'static': StaticViewSitemap,
#     'projects': ProjectSitemap,
#     'blogstatic': BlogStaticSitemap,
#     'blogposts': BlogPostSitemap,
#     'skillsstatic': SkillsStaticSitemap,
#     'skills': SkillSitemap,
#     'demos': DemoSitemap,
    
# }

# --- Sitemap Imports ---
try: from portfolio.sitemaps import StaticViewSitemap, ProjectSitemap
except ImportError: StaticViewSitemap, ProjectSitemap = None, None
try: from blog.sitemaps import BlogStaticSitemap, BlogPostSitemap
except ImportError: BlogStaticSitemap, BlogPostSitemap = None, None
try: from skills.sitemaps import SkillsStaticSitemap, SkillSitemap
except ImportError: SkillsStaticSitemap, SkillSitemap = None, None
try: from demos.sitemaps import DemoSitemap
except ImportError: DemoSitemap = None
try: from topics.sitemaps import TopicListSitemap, TopicSitemap # Import from topics app
except ImportError: TopicListSitemap, TopicSitemap = None, None
# --- End Sitemap Imports ---

sitemaps = {}
if StaticViewSitemap: sitemaps['static'] = StaticViewSitemap
if ProjectSitemap: sitemaps['projects'] = ProjectSitemap
if BlogStaticSitemap: sitemaps['blogstatic'] = BlogStaticSitemap
if BlogPostSitemap: sitemaps['blogposts'] = BlogPostSitemap
if SkillsStaticSitemap: sitemaps['skillsstatic'] = SkillsStaticSitemap
if SkillSitemap: sitemaps['skills'] = SkillSitemap
if DemoSitemap: sitemaps['demos'] = DemoSitemap
if TopicListSitemap: sitemaps['topiclist'] = TopicListSitemap # Add topic list sitemap
if TopicSitemap: sitemaps['topics'] = TopicSitemap # Add topic detail sitemap

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')),
    path('skills/', include('skills.urls', namespace='skills')),
    path('recommendations/', include('recommendations.urls', namespace='recommendations')), # Include recommendations URLs
    path('demos/', include('demos.urls', namespace='demos')), # Include demos URLs
    path('topics/', include('topics.urls', namespace='topics')), # Include topics URLs

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

