# recommendations/admin.py

from django.contrib import admin
from .models import RecommendedProduct

@admin.register(RecommendedProduct)
class RecommendedProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'order', 'product_url')
    list_filter = ('category',)
    search_fields = ('name', 'description', 'category')
    list_editable = ('order', 'category')
    fields = ('name', 'description', 'category', 'product_url', 'image_url', 'order')

