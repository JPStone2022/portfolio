# blog/views.py

from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import BlogPost

def blog_post_list(request):
    """Displays a list of published blog posts."""
    posts = BlogPost.objects.filter(
        status='published',
        published_date__lte=timezone.now()
    ).order_by('-published_date')
    context = {
        'posts': posts,
        'page_title': 'Blog',
    }
    return render(request, 'blog/blog_list.html', context)

def blog_post_detail(request, slug):
    """Displays a single blog post."""
    post = get_object_or_404(
        BlogPost,
        slug=slug,
        status='published',
        published_date__lte=timezone.now()
    )
    context = {
        'post': post,
        'page_title': post.title,
    }
    return render(request, 'blog/blog_detail.html', context)