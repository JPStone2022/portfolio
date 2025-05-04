# portfolio/models.py

from django.db import models
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse

# Import Skill model safely
try:
    from skills.models import Skill
except ImportError:
    Skill = None

# Removed ProjectTopic import from here

class Project(models.Model):
    """ Represents a single project in the portfolio. """
    title = models.CharField(max_length=200, help_text="The title of the project.")
    slug = models.SlugField(max_length=250, unique=True, blank=True, help_text="URL-friendly version of the title (auto-generated).")
    description = models.TextField(help_text="A detailed description of the project.")
    image_url = models.URLField(max_length=500, blank=True, null=True, help_text="URL for the project's main image or GIF.")
    results_metrics = models.TextField(blank=True, help_text="Specific results, metrics, or outcomes achieved.")
    challenges = models.TextField(blank=True, help_text="Key challenges faced during the project.")
    lessons_learned = models.TextField(blank=True, help_text="Important takeaways or lessons learned.")
    code_snippet = models.TextField(blank=True, help_text="An interesting code snippet related to the project.")
    code_language = models.CharField(max_length=50, blank=True, default='python', help_text="Language for syntax highlighting.")

    # --- Add ManyToManyField to ProjectTopic ---
    topics = models.ManyToManyField(
        'topics.ProjectTopic',
        blank=True,
        related_name="projects", # Allows topic.projects access
        verbose_name="Project Topics"
    )
    # --- End Topic Field ---

    # Deprecated field
    technologies = models.CharField(max_length=300, default='', blank=True, help_text="DEPRECATED: Use 'Skills' field.")
    # Skills relationship
    skills = models.ManyToManyField(
        Skill if Skill else 'skills.Skill',
        blank=True, related_name="projects", verbose_name="Associated Skills"
    )

    github_url = models.URLField(max_length=500, blank=True, null=True)
    demo_url = models.URLField(max_length=500, blank=True, null=True)
    paper_url = models.URLField(max_length=500, blank=True, null=True)
    date_created = models.DateField(default=timezone.now)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', '-date_created']

    def __str__(self):
        return self.title

    def get_technologies_list(self): # Keep for compatibility or remove if fully migrated
        if hasattr(self, 'skills') and self.skills.exists():
             return [skill.name for skill in self.skills.all()]
        if self.technologies:
             print(f"Warning: Project '{self.title}' using deprecated 'technologies'.")
             return [tech.strip() for tech in self.technologies.split(',') if tech.strip()]
        return []

    def get_absolute_url(self):
        return reverse('portfolio:project_detail', kwargs={'slug': self.slug})

    def save(self, *args, **kwargs):
        # --- Slug generation logic ---
        needs_slug = False
        if not self.slug: needs_slug = True
        elif self.pk is not None:
             try:
                 orig = Project.objects.get(pk=self.pk)
                 if orig.title != self.title: needs_slug = True
             except Project.DoesNotExist: needs_slug = True
        else: needs_slug = True
        if needs_slug:
            self.slug = slugify(self.title)
            original_slug = self.slug; counter = 1
            queryset = Project.objects.filter(slug=self.slug)
            if self.pk: queryset = queryset.exclude(pk=self.pk)
            while queryset.exists():
                self.slug = f'{original_slug}-{counter}'; counter += 1
                queryset = Project.objects.filter(slug=self.slug)
                if self.pk: queryset = queryset.exclude(pk=self.pk)
        # --- End Slug generation ---
        super().save(*args, **kwargs)

# --- Certificate Model ---
class Certificate(models.Model):
    title = models.CharField(max_length=250)
    issuer = models.CharField(max_length=150)
    date_issued = models.DateField(blank=True, null=True)
    certificate_file = models.FileField(upload_to='certificate_files/', blank=True, null=True)
    # Add ImageField for logo
    logo_image = models.ImageField(
        upload_to='certificate_logos/', # Subdirectory within MEDIA_ROOT
        blank=True,
        null=True,
        help_text="Upload a logo image for the issuer or certificate (optional)."
    )    
    order = models.PositiveIntegerField(default=0)
    class Meta: ordering = ['order', '-date_issued']
    def __str__(self): return f"{self.title} - {self.issuer}"
# --- End Certificate Model ---
