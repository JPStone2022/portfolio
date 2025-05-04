# your_app_name/migrations/000X_create_initial_superuser.py
from django.db import migrations
import os # Import os to access environment variables

def create_superuser(apps, schema_editor):
    # Get the User model from the 'auth' app
    User = apps.get_model('auth', 'User')

    # Get credentials from environment variables
    username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
    email = os.environ.get('DJANGO_SUPERUSER_EMAIL')
    password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')

    # Only proceed if all variables are set
    if not all([username, email, password]):
        print("Superuser environment variables not fully set. Skipping superuser creation.")
        return # Exit the function if variables aren't set

    # Check if superuser already exists
    if User.objects.filter(username=username).exists():
        print(f"Superuser '{username}' already exists. Skipping creation.")
        return # Exit if user exists

    # Create the superuser
    print(f"Creating superuser '{username}'...")
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f"Superuser '{username}' created successfully.")

class Migration(migrations.Migration):

    # Add dependency on the previous migration in this app, if any
    # dependencies = [
    #     ('your_app_name', '000Y_previous_migration'),
    # ]
    # If this is the first migration in the app, dependencies might just be
    dependencies = [
         ('portfolio', '0003_alter_project_table'), # Example dependency from auth app
         # Add dependency on your custom user model migration if you have one
    ]


    operations = [
        # Run the Python function `create_superuser`
        migrations.RunPython(create_superuser),
    ]