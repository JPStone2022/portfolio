# dl_portfolio_project/settings.py

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/stable/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-=your-secret-key-here' # Replace with a real secret key

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True # Set to False for production

ALLOWED_HOSTS = [] # Add your domain/IP in production


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin', # Needs auth and contenttypes
    'django.contrib.auth', # Core authentication framework
    'django.contrib.contenttypes', # Django content type system
    'django.contrib.sessions', # Session framework
    'django.contrib.messages', # Messaging framework
    'django.contrib.staticfiles', # Manages static files
    'portfolio', # Your app
]

# Add the default middleware stack
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware', # Manages sessions across requests
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', # Cross Site Request Forgery protection
    'django.contrib.auth.middleware.AuthenticationMiddleware', # Associates users with requests using sessions (REQUIRED FOR ADMIN)
    'django.contrib.messages.middleware.MessageMiddleware', # Enables message framework
    'django.middleware.clickjacking.XFrameOptionsMiddleware', # Clickjacking protection
]

ROOT_URLCONF = 'dl_portfolio_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True, # Allows Django to find admin templates
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request', # Required by admin
                'django.contrib.auth.context_processors.auth', # Adds user to template context (Required by admin)
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'dl_portfolio_project.wsgi.application'


# Database
# https://docs.djangoproject.com/en/stable/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/stable/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    { 'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator', },
    { 'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', },
    { 'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator', },
    { 'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator', },
]


# Internationalization
# https://docs.djangoproject.com/en/stable/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/stable/howto/static-files/
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles' # For collectstatic in production


# Media files (User-uploaded content)
# https://docs.djangoproject.com/en/stable/howto/static-files/#serving-files-uploaded-by-a-user-during-development
MEDIA_URL = '/media/' # Base URL for serving media files
MEDIA_ROOT = BASE_DIR / 'mediafiles' # Absolute filesystem path to the directory for user uploads


# Default primary key field type
# https://docs.djangoproject.com/en/stable/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
