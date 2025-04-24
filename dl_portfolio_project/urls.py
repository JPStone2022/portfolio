# dl_portfolio_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings # Import settings
from django.conf.urls.static import static # Import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', include('blog.urls', namespace='blog')), # Include blog URLs under /blog/
    path('skills/', include('skills.urls', namespace='skills')), # Include skills URLs
    # Include URLs from the portfolio app
    path('', include('portfolio.urls')), # Make sure 'portfolio.urls' is correct
]

# Add this block to serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Note: In production, your web server (e.g., Nginx) should be configured
# to serve files from MEDIA_ROOT directly. This block should not be used.
