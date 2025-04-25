# portfolio/admin.py

from django.contrib import admin
from portfolio.models import Project, Certificate # Import BlogPost # Import both models
# Import Skill model safely
try:
    from skills.models import Skill
except ImportError:
    Skill = None

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'date_created', 'order') # Removed links for brevity here
    list_filter = ('date_created',) # Removed technologies filter for now
    search_fields = ('title', 'description') # Removed technologies search for now
    list_editable = ('order',)
    prepopulated_fields = {'slug': ('title',)}
    # Add 'skills' and new text fields to fieldsets
    fieldsets = (
        (None, {
            'fields': ('title', 'slug', 'description', 'image_url') # Basic info
        }),
        ('Associated Skills', { # Section for skills relationship
            'fields': ('skills',)
        }),
        # New Section for Outcomes
        ('Project Outcomes', {
            'classes': ('collapse',), # Optional: Make the section collapsible
            'fields': ('results_metrics', 'challenges', 'lessons_learned')
        }),
        ('Links', { # Project links
            'fields': ('github_url', 'demo_url', 'paper_url')
        }),
        ('Details', { # Other details
            # Removed deprecated 'technologies' field
            'fields': ('order', 'date_created')
        }),
    )
    # Ensure filter_horizontal is used for the skills ManyToManyField
    filter_horizontal = ('skills',) if Skill else () # Apply only if Skill model is available


# Register the Certificate model
@admin.register(Certificate)
class CertificateAdmin(admin.ModelAdmin):
    # Update list_display and fields to use certificate_file
    list_display = ('title', 'issuer', 'date_issued', 'order', 'certificate_file') # Changed credential_url to certificate_file
    list_filter = ('issuer', 'date_issued')
    search_fields = ('title', 'issuer')
    list_editable = ('order',)
    # Update fields list
    fields = ('title', 'issuer', 'date_issued', 'certificate_file', 'logo_image', 'order') # Changed credential_url to certificate_file
