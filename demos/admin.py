# demos/admin.py
from django.contrib import admin
from .models import Demo
    # Import Skill model safely for checking if app exists
try:
    from skills.models import Skill
    SKILLS_APP_ENABLED = True
except ImportError:
    SKILLS_APP_ENABLED = False
    Skill = None # Define Skill as None if app doesn't exist

@admin.register(Demo)
class DemoAdmin(admin.ModelAdmin):
    list_display = ('title', 'demo_url_name', 'order', 'is_featured')
    # Add skills to filter only if the Skill model could be imported
    list_filter = ('is_featured',) + (('skills',) if SKILLS_APP_ENABLED else ())
    search_fields = ('title', 'description') + (('skills__name',) if SKILLS_APP_ENABLED else ()) # Search related skills
    list_editable = ('order', 'is_featured')
    prepopulated_fields = {'slug': ('title',)}
    # Add skills field to the form display
    fields = ('title', 'slug', 'description', 'demo_url_name', 'image_url', 'skills', 'order', 'is_featured')
    # Use filter_horizontal for the skills ManyToManyField if Skill model exists
    filter_horizontal = ('skills',) if SKILLS_APP_ENABLED else ()
    