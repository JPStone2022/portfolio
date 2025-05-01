# # demos/admin.py
# from django.contrib import admin
# from .models import Demo

# @admin.register(Demo)
# class DemoAdmin(admin.ModelAdmin):
#     list_display = ('title', 'demo_url_name', 'order', 'is_featured')
#     list_filter = ('is_featured',)
#     search_fields = ('title', 'description')
#     list_editable = ('order', 'is_featured')
#     prepopulated_fields = {'slug': ('title',)}
#     fields = ('title', 'slug', 'description', 'demo_url_name', 'image_url', 'order', 'is_featured')

