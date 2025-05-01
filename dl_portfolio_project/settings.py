# dl_portfolio_project/settings.py

import os
from pathlib import Path
import dj_database_url
# --- Add these lines ---
from dotenv import load_dotenv
load_dotenv() # Loads variables from .env file into environment
# -----------------------
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Construct the path to the .env file
dotenv_path = BASE_DIR / '.env'

# Load .env only if it exists (indicating development)
if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path)
    DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        # Configure production database using environment variables if needed
        # ssl_require=True if not DEBUG else False # Example for Heroku PG
    }
}
else:
    # Database configuration (using dj-database-url example)
    
    DATABASES = {
        'default': dj_database_url.config(
            default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
            conn_max_age=600,
            # ssl_require=True if not DEBUG else False # Example for Heroku PG
        )
    }

# SECURITY WARNING: keep the secret key used in production secret!
# Read secret key from environment variable in production
#SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-=your-default-development-key-here') # Replace fallback key

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'False') == 'True' # Default to True for dev, set env var to 'False' in prod
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY and os.environ.get('DJANGO_SETTINGS_MODULE') == 'dl_portfolio_project.settings':
    SECRET_KEY = 'django-insecure-=replace-with-your-local-dev-key-here'
#DEBUG = os.environ.get('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = []
RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
if RENDER_EXTERNAL_HOSTNAME: ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
# Allow pythonanywhere domain if set
PYTHONANYWHERE_DOMAIN = os.environ.get('PYTHONANYWHERE_DOMAIN')
if PYTHONANYWHERE_DOMAIN: ALLOWED_HOSTS.append(PYTHONANYWHERE_DOMAIN)
if DEBUG: ALLOWED_HOSTS.extend(['localhost', '127.0.0.1'])
# CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN')
# if CUSTOM_DOMAIN: ALLOWED_HOSTS.append(CUSTOM_DOMAIN)

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize', # For template filters like naturaltime
    'django.contrib.sitemaps', # Add this line
    'portfolio', # Your portfolio app
    'blog',      # Your blog app
    'skills',    # Your skills app
    'recommendations', # Add the new app
    #'demos',
    # Add whitenoise.runserver_nostatic if DEBUG is True for easier local static serving
    'whitenoise.runserver_nostatic', # Optional for development convenience
]

# Conditionally add 'demos' app only if DEBUG is True
if DEBUG:
    INSTALLED_APPS += [
        'demos',
    ]
# --- End Conditional App ---

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    # Add WhiteNoise middleware right after SecurityMiddleware
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'dl_portfolio_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # 'DIRS': [],
        'DIRS': [BASE_DIR / 'templates'], # <--- UPDATE THIS LINE
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # Add the path to your recommendations context processor
                'recommendations.context_processors.recommendation_context',

            ],
        },
    },
]

WSGI_APPLICATION = 'dl_portfolio_project.wsgi.application'


# Database used in development

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#         # Configure production database using environment variables if needed
#         # ssl_require=True if not DEBUG else False # Example for Heroku PG
#     }
# }

# Database used in production

# # Database configuration (using dj-database-url example)
# DATABASES = {
#     'default': dj_database_url.config(
#         default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
#         conn_max_age=600,
#         # ssl_require=True if not DEBUG else False # Example for Heroku PG
#     )
# }

#print("DEBUG DATABASES:", DATABASES)


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

# # Staticfiles storage using WhiteNoise (Recommended for Render)
# # For Django 4.2+
# STORAGES = {
#     "staticfiles": {
#         "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
#     },
# # If you use django-storages for media files (e.g., S3), configure default here:
#     "default": {
#      "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",
#     },
#     }
# # For Django < 4.2, use this instead of STORAGES["staticfiles"]:
# # STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
# https://docs.djangoproject.com/en/stable/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# Email Configuration
# --------------------------------------------------------------------------
# Choose ONE backend.

# Option 1: Console backend (for development - prints emails to console)
# EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Option 2: SMTP backend (for production - e.g., Gmail, SendGrid, Mailgun, etc.)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com') # e.g., 'smtp.gmail.com' or your provider's SMTP server
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587)) # 587 for TLS, 465 for SSL, 25 for unencrypted (not recommended)
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True') == 'True' # Use True for port 587
EMAIL_USE_SSL = os.environ.get('EMAIL_USE_SSL', 'False') == 'True' # Use True for port 465 (TLS and SSL are mutually exclusive)
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', 'your_email@example.com') # Your email address or username
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', 'your_password_or_app_password') # ** STORE SECURELY - Use env var! **

# Default email address for 'from' field in emails sent by Django (e.g., error reports)
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', EMAIL_HOST_USER)
# Email address for site admins to receive error notifications etc.
SERVER_EMAIL = os.environ.get('SERVER_EMAIL', EMAIL_HOST_USER)
# ADMINS = [('Your Name', 'your_admin_email@example.com')] # Optional: For site error notifications

# --------------------------------------------------------------------------

# Security settings for production (when DEBUG=False)
# Uncomment and configure these as needed, especially if using HTTPS
# CSRF_COOKIE_SECURE = True
# SESSION_COOKIE_SECURE = True
# SECURE_SSL_REDIRECT = True
# SECURE_HSTS_SECONDS = 31536000 # e.g., 1 year
# SECURE_HSTS_INCLUDE_SUBDOMAINS = True
# SECURE_HSTS_PRELOAD = True
# SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https') # If behind a proxy like Render's
