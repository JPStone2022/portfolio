# portfolio/admin.py

from django.contrib import admin
from portfolio.models import Project, Certificate

try:
    from skills.models import Skill
except ImportError:
    Skill = None

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'date_created', 'order')
    list_filter = ('date_created', 'skills')
    search_fields = ('title', 'description', 'skills__name')
    list_editable = ('order',)
    prepopulated_fields = {'slug': ('title',)}
    fieldsets = (
        (None, {
            'fields': ('title', 'slug', 'description', 'image_url')
        }),
        ('Associated Skills', {
            'fields': ('skills',)
        }),
        ('Project Outcomes', {
            'classes': ('collapse',),
            'fields': ('results_metrics', 'challenges', 'lessons_learned')
        }),
        # Add Section for Code Snippet
        ('Code Snippet', {
            'classes': ('collapse',), # Optional: Make collapsible
            'fields': ('code_snippet', 'code_language') # Add new fields
        }),
        ('Links', {
            'fields': ('github_url', 'demo_url', 'paper_url')
        }),
        ('Details', {
            'fields': ('order', 'date_created')
        }),
    )
    filter_horizontal = ('skills',) if Skill else ()

@admin.register(Certificate)
class CertificateAdmin(admin.ModelAdmin):
    list_display = ('title', 'issuer', 'date_issued', 'order', 'certificate_file')
    list_filter = ('issuer', 'date_issued')
    search_fields = ('title', 'issuer')
    list_editable = ('order',)
    fields = ('title', 'issuer', 'date_issued', 'certificate_file', 'order')

