# portfolio/models.py

from django.db import models
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse
# Import the Skill model from the skills app
try:
    from skills.models import Skill
except ImportError:
    Skill = None # Handle case where skills app might not exist yet

# Existing Project Model...
class Project(models.Model):
    # ... (Project model fields remain the same) ...
    title = models.CharField(max_length=200, help_text="The title of the project.")
    slug = models.SlugField(max_length=250, unique=True, blank=True, help_text="URL-friendly version of the title (auto-generated).")
    description = models.TextField(help_text="A detailed description of the project.")
    image_url = models.URLField(max_length=500, blank=True, null=True, help_text="URL for the project's main image or GIF.")

    # --- New Fields Added ---
    results_metrics = models.TextField(
        blank=True,
        help_text="Specific results, metrics, or outcomes achieved (e.g., accuracy improvement, performance gains)."
    )
    challenges = models.TextField(
        blank=True,
        help_text="Key challenges faced during the project and how they were addressed."
    )
    lessons_learned = models.TextField(
        blank=True,
        help_text="Important takeaways, insights, or lessons learned from the project."
    )
    # --- End New Fields ---
    # --- New Field for Code Snippet ---
    code_snippet = models.TextField(
        blank=True,
        help_text="An interesting code snippet related to the project (paste code here)."
    )
    # Optional: Field to specify the language for syntax highlighting
    code_language = models.CharField(
        max_length=50,
        blank=True,
        default='python', # Default to python
        help_text="Language for syntax highlighting (e.g., python, javascript, html, css, sql)."
    )
    # --- End New Field ---
    
    # Deprecated field - keep for now if data migration is needed later
    technologies = models.CharField(
        max_length=300,
        default='',
        blank=True,
        # help_text="Comma-separated list of key technologies used (e.g., PyTorch, CNN, OpenCV)."
        help_text="DEPRECATED: Comma-separated list. Use the 'Skills' field instead." # Mark old field as deprecated
    )
    # Add the ManyToManyField to Skill
    # ManyToManyField to Skill (ensure Skill model is imported)
    skills = models.ManyToManyField(
        Skill,
        blank=True, # Allow projects to have no skills linked
        related_name="projects", # Allows accessing project_set as skill.projects
        verbose_name="Associated Skills"
    )
    github_url = models.URLField(max_length=500, blank=True, null=True, help_text="Link to the project's GitHub repository.")
    demo_url = models.URLField(max_length=500, blank=True, null=True, help_text="Link to a live demo, if available.")
    paper_url = models.URLField(max_length=500, blank=True, null=True, help_text="Link to a relevant paper, if available.")
    date_created = models.DateField(default=timezone.now, help_text="The date the project entry was created.")
    order = models.PositiveIntegerField(default=0, help_text="Order in which to display projects (lower numbers first).")

    class Meta:
        ordering = ['order', '-date_created']

    def __str__(self):
        return self.title

    def get_technologies_list(self):
         # Prioritize new skills field if available
        if self.skills.exists():
             return [skill.name for skill in self.skills.all()]
        # Fallback to old field if needed (optional)
        if self.technologies:
             print(f"Warning: Project '{self.title}' is using the deprecated 'technologies' field.")
             return [tech.strip() for tech in self.technologies.split(',') if tech.strip()]
        return []

    def get_absolute_url(self):
        return reverse('portfolio:project_detail', kwargs={'slug': self.slug})

    def save(self, *args, **kwargs):
        # Check if slug is missing or title changed (handle potential AttributeError if self.pk is None)
        needs_slug = False
        if not self.slug:
            needs_slug = True
        elif self.pk is not None: # Check if object exists in DB
             try:
                 orig = Project.objects.get(pk=self.pk)
                 if orig.title != self.title:
                     needs_slug = True
             except Project.DoesNotExist:
                 needs_slug = True # Should not happen if pk exists, but safe check
        else: # New object being created
             needs_slug = True

        if needs_slug:
            self.slug = slugify(self.title)
            # Ensure slug uniqueness (append number if needed)
            original_slug = self.slug
            counter = 1
            # Adjust query to handle new objects (self.pk is None)
            queryset = Project.objects.filter(slug=self.slug)
            if self.pk:
                queryset = queryset.exclude(pk=self.pk)

            while queryset.exists():
                self.slug = f'{original_slug}-{counter}'
                counter += 1
                # Update queryset condition for the loop
                queryset = Project.objects.filter(slug=self.slug)
                if self.pk:
                    queryset = queryset.exclude(pk=self.pk)

        super().save(*args, **kwargs) # Call the "real" save() method.


# Updated Certificate Model
class Certificate(models.Model):
    """
    Represents a certificate or accomplishment.
    """
    title = models.CharField(max_length=250, help_text="The name of the certificate or accomplishment.")
    issuer = models.CharField(max_length=150, help_text="The organization that issued the certificate.")
    date_issued = models.DateField(blank=True, null=True, help_text="The date the certificate was issued.")
    # Replace credential_url with FileField for PDF/other file uploads
    certificate_file = models.FileField(
        upload_to='certificate_files/', # Subdirectory within MEDIA_ROOT for certificate files
        blank=True,
        null=True,
        help_text="Upload the certificate file (e.g., PDF)."
    )
    # Keep logo_image if you still want a separate logo upload
    logo_image = models.ImageField(
        upload_to='certificate_logos/',
        blank=True,
        null=True,
        help_text="Upload a logo image (optional)."
    )
    order = models.PositiveIntegerField(default=0, help_text="Order in which to display certificates (lower numbers first).")

    class Meta:
        ordering = ['order', '-date_issued'] # Default ordering

    def __str__(self):
        return f"{self.title} - {self.issuer}"

