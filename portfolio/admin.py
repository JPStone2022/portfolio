# portfolio/admin.py

from django.contrib import admin
from .models import Project, Certificate # Import BlogPost # Import both models

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'date_created', 'order', 'github_url', 'demo_url')
    list_filter = ('date_created', 'technologies')
    search_fields = ('title', 'description', 'technologies')
    list_editable = ('order',)
    prepopulated_fields = {'slug': ('title',)}
    fieldsets = (
        (None, {
            'fields': ('title', 'slug', 'description', 'image_url')
        }),
        ('Links', {
            'fields': ('github_url', 'demo_url', 'paper_url')
        }),
        ('Details', {
            'fields': ('technologies', 'order', 'date_created')
        }),
    )

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
