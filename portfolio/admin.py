# portfolio/admin.py

from django.contrib import admin
# Import models from this app
from .models import Project, Certificate, ProjectTopic

# Import Skill model safely
try:
    from skills.models import Skill
except ImportError:
    Skill = None

# Register ProjectTopic Admin
@admin.register(ProjectTopic)
class ProjectTopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'order')
    list_editable = ('order',)
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ('name', 'description')

# Update Project Admin
@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'date_created', 'order')
    # Add topics to filter
    list_filter = ('date_created', 'skills', 'topics')
    # Add topics to search
    search_fields = ('title', 'description', 'skills__name', 'topics__name')
    list_editable = ('order',)
    prepopulated_fields = {'slug': ('title',)}
    fieldsets = (
        (None, {
            'fields': ('title', 'slug', 'description', 'image_url')
        }),
        # Combine Topics & Skills
        ('Topics & Skills', {
            'fields': ('topics', 'skills') # Add topics field
        }),
        ('Project Outcomes', {
            'classes': ('collapse',),
            'fields': ('results_metrics', 'challenges', 'lessons_learned')
        }),
        ('Code Snippet', {
            'classes': ('collapse',),
            'fields': ('code_snippet', 'code_language')
        }),
        ('Links', {
            'fields': ('github_url', 'demo_url', 'paper_url')
        }),
        ('Details', {
            'fields': ('order', 'date_created')
        }),
    )
    # Add topics to filter_horizontal
    filter_horizontal = ('skills', 'topics',) if Skill else ('topics',)

# Certificate Admin (remains the same)
@admin.register(Certificate)
class CertificateAdmin(admin.ModelAdmin):
    list_display = ('title', 'issuer', 'date_issued', 'order', 'certificate_file')
    list_filter = ('issuer', 'date_issued')
    search_fields = ('title', 'issuer')
    list_editable = ('order',)
    fields = ('title', 'issuer', 'date_issued', 'certificate_file', 'order')

