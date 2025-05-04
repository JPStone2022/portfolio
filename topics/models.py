from django.db import models
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse

# Create your models here.


# --- New ProjectTopic Model ---
class ProjectTopic(models.Model):
    """ Represents a broader topic area for projects (e.g., NLP, Computer Vision). """
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True, blank=True)
    description = models.TextField(blank=True, help_text="Optional description of the topic.")
    order = models.PositiveIntegerField(default=0, help_text="Order for display.")

    class Meta:
        ordering = ['order', 'name']
        verbose_name = "Project Topic"
        verbose_name_plural = "Project Topics"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        """ Auto-generate slug. """
        if not self.slug:
            self.slug = slugify(self.name)
            # Basic uniqueness check (more robust needed for high concurrency)
            original_slug = self.slug
            counter = 1
            while ProjectTopic.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f'{original_slug}-{counter}'
                counter += 1
        super().save(*args, **kwargs)

    # Add get_absolute_url method
    def get_absolute_url(self):
        """ Returns the URL to access a detail page for this topic. """
        return reverse('topics:topic_detail', kwargs={'topic_slug': self.slug})
    
    # Optional: Add get_absolute_url if you create a page listing projects by topic
    # def get_absolute_url(self):
    #     return reverse('portfolio:projects_by_topic', kwargs={'topic_slug': self.slug})
# --- End ProjectTopic Model ---
