# recommendations/admin.py

from django.contrib import admin
from .models import RecommendedProduct

@admin.register(RecommendedProduct)
class RecommendedProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'order', 'product_url')
    list_filter = ('category',)
    search_fields = ('name', 'description', 'category')
    list_editable = ('order', 'category')
    # Define fields shown in the edit/add view
    fields = ('name', 'slug', 'description', 'category', 'product_url', 'image_url', 'order')
    # Automatically populate slug from name in the add form
    prepopulated_fields = {'slug': ('name',)}
    # Remove slug from readonly_fields to allow prepopulation to work
    # The model's save() method ensures it's generated correctly anyway.
    # readonly_fields = ('slug',) # REMOVED THIS LINE

