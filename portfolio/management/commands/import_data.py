# portfolio/management/commands/import_data.py
# (Place this file in portfolio/management/commands/import_data.py)

import csv
import os
from django.core.management.base import BaseCommand, CommandError
from django.utils.text import slugify
from django.db import transaction # For atomic operations
from django.db.models import Q # *** ADD THIS IMPORT FOR COMPLEX LOOKUPS ***
from django.utils import timezone # For date parsing

# Import your models (adjust paths as needed)
# It's safer to use try-except blocks in case apps aren't always present
try:
    from portfolio.models import Project, Certificate
    PORTFOLIO_APP_EXISTS = True
except ImportError:
    Project, Certificate = None, None
    PORTFOLIO_APP_EXISTS = False

try:
    from blog.models import BlogPost
    BLOG_APP_EXISTS = True
except ImportError:
    BlogPost = None
    BLOG_APP_EXISTS = False

try:
    from skills.models import Skill, SkillCategory
    SKILLS_APP_EXISTS = True
except ImportError:
    Skill, SkillCategory = None, None
    SKILLS_APP_EXISTS = False

try:
    from topics.models import ProjectTopic
    TOPICS_APP_EXISTS = True
except ImportError:
    ProjectTopic = None
    TOPICS_APP_EXISTS = False

try:
    from recommendations.models import RecommendedProduct
    RECOMMENDATIONS_APP_EXISTS = True
except ImportError:
    RecommendedProduct = None
    RECOMMENDATIONS_APP_EXISTS = False


# Define which model corresponds to which file type/name
# Only include models that were successfully imported
MODEL_MAP = {}
if PORTFOLIO_APP_EXISTS:
    MODEL_MAP['projects'] = Project
    MODEL_MAP['certificates'] = Certificate
    MODEL_MAP['topics'] = ProjectTopic
if BLOG_APP_EXISTS:
    MODEL_MAP['blogposts'] = BlogPost
if SKILLS_APP_EXISTS:
    MODEL_MAP['skills'] = Skill
    MODEL_MAP['skillcategories'] = SkillCategory
if RECOMMENDATIONS_APP_EXISTS:
    MODEL_MAP['recommendations'] = RecommendedProduct


class Command(BaseCommand):
    help = 'Imports data from a specified CSV file into the database.'

    def add_arguments(self, parser):
        parser.add_argument('csv_filepath', type=str, help='The path to the CSV file to import.')
        parser.add_argument(
            '--model_type',
            type=str,
            help=f"The type of model to import (e.g., {', '.join(MODEL_MAP.keys())}). Must match keys in MODEL_MAP.",
            required=True # Make specifying the model type mandatory
        )
        parser.add_argument(
            '--update',
            action='store_true', # Makes this a flag, default is False
            help='Update existing records based on a unique field (e.g., slug or title) instead of just creating new ones.',
        )
        parser.add_argument(
            '--unique_field',
            type=str,
            default='slug', # Default field to check for updates
            help='The unique field name to use for matching when updating (default: slug). Use "title" or "name" if slug is auto-generated.',
        )


    @transaction.atomic # Wrap the whole import in a transaction
    def handle(self, *args, **options):
        csv_filepath = options['csv_filepath']
        model_type = options['model_type'].lower()
        update_existing = options['update']
        unique_field = options['unique_field']

        # Validate model_type
        if model_type not in MODEL_MAP:
            raise CommandError(f"Invalid model_type '{model_type}'. Valid types are: {', '.join(MODEL_MAP.keys())}")

        TargetModel = MODEL_MAP[model_type]
        if TargetModel is None: # Check if model failed to import earlier
             raise CommandError(f"Model for type '{model_type}' could not be imported. Is the app in INSTALLED_APPS?")


        # Check if file exists
        if not os.path.exists(csv_filepath):
            raise CommandError(f"File not found at path: {csv_filepath}")

        self.stdout.write(self.style.SUCCESS(f"Starting import for '{model_type}' from '{csv_filepath}'..."))
        if update_existing:
            self.stdout.write(self.style.WARNING(f"Update mode enabled. Matching on field '{unique_field}'."))

        created_count = 0
        updated_count = 0
        skipped_count = 0

        try:
            with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile: # Use utf-8-sig to handle potential BOM
                # Use DictReader for easier access by column header
                reader = csv.DictReader(csvfile)

                # Check for required headers based on model type (add more checks as needed)
                required_headers = []
                if model_type == 'projects': required_headers = ['title', 'description']
                elif model_type == 'skills': required_headers = ['name']
                elif model_type == 'topics': required_headers = ['name']
                elif model_type == 'certificates': required_headers = ['title']
                # Add checks for other models...

                if not reader.fieldnames:
                    raise CommandError(f"CSV file '{csv_filepath}' appears to be empty or has no headers.")

                if required_headers and not all(h in reader.fieldnames for h in required_headers):
                     raise CommandError(f"CSV missing required headers for '{model_type}'. Expected: {', '.join(required_headers)}. Found: {', '.join(reader.fieldnames)}")
                if update_existing and unique_field not in reader.fieldnames:
                     raise CommandError(f"CSV missing the unique field '{unique_field}' needed for updating.")


                for row_num, row in enumerate(reader, start=1):
                    try:
                        # --- Prepare data dictionary ---
                        data = {}
                        unique_value = row.get(unique_field, '').strip()
                        skill_names_or_slugs = [] # Initialize M2M lists
                        topic_names_or_slugs = []

                        # --- Specific Logic per Model Type ---
                        if model_type == 'projects':
                            if not PORTFOLIO_APP_EXISTS: continue # Skip if app not available
                            data['title'] = row.get('title', '').strip()
                            if not data['title']: raise ValueError("Missing required field: title")
                            data['description'] = row.get('description', '').strip()
                            data['image_url'] = row.get('image_url', '').strip() or None
                            data['results_metrics'] = row.get('results_metrics', '').strip()
                            data['challenges'] = row.get('challenges', '').strip()
                            data['lessons_learned'] = row.get('lessons_learned', '').strip()
                            data['code_snippet'] = row.get('code_snippet', '').strip()
                            data['code_language'] = row.get('code_language', 'python').strip() or 'python'
                            data['github_url'] = row.get('github_url', '').strip() or None
                            data['demo_url'] = row.get('demo_url', '').strip() or None
                            data['paper_url'] = row.get('paper_url', '').strip() or None
                            try: data['order'] = int(row.get('order', 0))
                            except (ValueError, TypeError): data['order'] = 0
                            if unique_field == 'slug' and unique_value: data['slug'] = unique_value
                            elif unique_field == 'title': unique_value = data['title']

                            skill_names_or_slugs = [s.strip() for s in row.get('skills', '').split(',') if s.strip()]
                            topic_names_or_slugs = [t.strip() for t in row.get('topics', '').split(',') if t.strip()]

                        elif model_type == 'skills':
                            if not SKILLS_APP_EXISTS: continue
                            data['name'] = row.get('name', '').strip()
                            if not data['name']: raise ValueError("Missing required field: name")
                            data['description'] = row.get('description', '').strip()
                            try: data['order'] = int(row.get('order', 0))
                            except (ValueError, TypeError): data['order'] = 0
                            if unique_field == 'slug' and unique_value: data['slug'] = unique_value
                            elif unique_field == 'name': unique_value = data['name']

                            category_name = row.get('category_name', '').strip()
                            if category_name:
                                category, created = SkillCategory.objects.get_or_create(name=category_name)
                                if created: self.stdout.write(f"  Created SkillCategory: {category_name}")
                                data['category'] = category

                        elif model_type == 'topics':
                             if not TOPICS_APP_EXISTS: continue
                             data['name'] = row.get('name', '').strip()
                             if not data['name']: raise ValueError("Missing required field: name")
                             data['description'] = row.get('description', '').strip()
                             try: data['order'] = int(row.get('order', 0))
                             except (ValueError, TypeError): data['order'] = 0
                             if unique_field == 'slug' and unique_value: data['slug'] = unique_value
                             elif unique_field == 'name': unique_value = data['name']

                        elif model_type == 'certificates':
                            if not PORTFOLIO_APP_EXISTS: continue
                            data['title'] = row.get('title', '').strip()
                            if not data['title']: raise ValueError("Missing required field: title")
                            data['issuer'] = row.get('issuer', '').strip()
                            # Handle date conversion carefully
                            date_str = row.get('date_issued', '').strip()
                            if date_str:
                                try: data['date_issued'] = timezone.datetime.strptime(date_str, '%Y-%m-%d').date() # Example format
                                except ValueError: raise ValueError(f"Invalid date format for date_issued: '{date_str}'. Use YYYY-MM-DD.")
                            else: data['date_issued'] = None
                            # certificate_file needs special handling if uploading via CSV (not directly supported here)
                            try: data['order'] = int(row.get('order', 0))
                            except (ValueError, TypeError): data['order'] = 0
                            # Set unique value for update matching (usually title for certificates)
                            if unique_field == 'title':
                                unique_value = data['title']
                            # Add elif for other unique fields if needed
                        # --- Add logic for other model_types (BlogPosts, Recommendations etc.) here ---

                        # Example for Recommendations:
                        elif model_type == 'recommendations':
                            if not RECOMMENDATIONS_APP_EXISTS: continue
                            data['name'] = row.get('name', '').strip()
                            if not data['name']: raise ValueError("Missing required field: name")
                            data['description'] = row.get('description', '').strip()
                            data['product_url'] = row.get('product_url', '').strip()
                            if not data['product_url']: raise ValueError("Missing required field: product_url")
                            data['image_url'] = row.get('image_url', '').strip() or None
                            data['category'] = row.get('category', '').strip()
                            try: data['order'] = int(row.get('order', 0))
                            except (ValueError, TypeError): data['order'] = 0
                            if unique_field == 'name': unique_value = data['name'] # Match on name for recommendations


                        # --- Create or Update Logic ---
                        instance = None
                        if update_existing and unique_value:
                            try:
                                lookup_params = {unique_field: unique_value}
                                instance = TargetModel.objects.get(**lookup_params)
                                for key, value in data.items():
                                    setattr(instance, key, value)
                                instance.save()
                                updated_count += 1
                                self.stdout.write(f"  Updated {model_type}: {instance}")
                            except TargetModel.DoesNotExist:
                                if unique_field == 'slug': data['slug'] = unique_value
                                instance = TargetModel.objects.create(**data)
                                created_count += 1
                                self.stdout.write(f"  Created {model_type}: {instance}")
                        else:
                            instance = TargetModel.objects.create(**data)
                            created_count += 1
                            self.stdout.write(f"  Created {model_type}: {instance}")

                        # --- Handle ManyToMany Post-Save (Example for Projects) ---
                        if instance and model_type == 'projects':
                             if update_existing:
                                 instance.skills.clear()
                                 instance.topics.clear()

                             # Add Skills
                             if SKILLS_APP_EXISTS and Skill:
                                 for skill_identifier in skill_names_or_slugs:
                                     try:
                                         skill = Skill.objects.get(Q(slug=skill_identifier) | Q(name=skill_identifier))
                                         instance.skills.add(skill)
                                     except Skill.DoesNotExist: self.stdout.write(self.style.WARNING(f"    Skill '{skill_identifier}' not found. Skipping."))
                                     except Skill.MultipleObjectsReturned: self.stdout.write(self.style.WARNING(f"    Multiple skills found for '{skill_identifier}'. Skipping."))

                             # Add Topics
                             if PORTFOLIO_APP_EXISTS and ProjectTopic:
                                 for topic_identifier in topic_names_or_slugs:
                                     try:
                                         topic = ProjectTopic.objects.get(Q(slug=topic_identifier) | Q(name=topic_identifier))
                                         instance.topics.add(topic)
                                     except ProjectTopic.DoesNotExist: self.stdout.write(self.style.WARNING(f"    Topic '{topic_identifier}' not found. Skipping."))
                                     except ProjectTopic.MultipleObjectsReturned: self.stdout.write(self.style.WARNING(f"    Multiple topics found for '{topic_identifier}'. Skipping."))

                    except ValueError as ve:
                        self.stdout.write(self.style.ERROR(f"Skipping row {row_num}: Invalid data - {ve}"))
                        skipped_count += 1
                        continue
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"Error processing row {row_num}: {e} - Data: {row}"))
                        skipped_count += 1
                        continue


        except FileNotFoundError:
            raise CommandError(f"File not found: '{csv_filepath}'")
        except Exception as e:
            # Rollback happens automatically due to @transaction.atomic
            raise CommandError(f"An error occurred during import: {e}")

        self.stdout.write(self.style.SUCCESS(f"Import finished. Created: {created_count}, Updated: {updated_count}, Skipped: {skipped_count}"))

