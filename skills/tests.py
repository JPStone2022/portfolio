# skills/tests.py

from django.test import TestCase, Client
from django.urls import reverse
from django.utils.text import slugify
from .models import Skill, SkillCategory

class SkillCategoryModelTests(TestCase):

    def test_skill_category_creation(self):
        """ Test basic SkillCategory creation and defaults. """
        category = SkillCategory.objects.create(name="Programming Languages")
        self.assertEqual(str(category), "Programming Languages")
        self.assertEqual(category.order, 0)
        self.assertTrue(isinstance(category, SkillCategory))

class SkillModelTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.category1 = SkillCategory.objects.create(name="Frameworks", order=1)
        cls.skill = Skill.objects.create(
            name="Django Skill Test",
            description="Testing the Django skill.",
            category=cls.category1
        )

    def test_skill_creation(self):
        """ Test basic Skill creation and defaults. """
        self.assertEqual(str(self.skill), "Django Skill Test")
        self.assertEqual(self.skill.order, 0)
        self.assertEqual(self.skill.category, self.category1)
        self.assertTrue(isinstance(self.skill, Skill))

    def test_slug_generation_on_save(self):
        """ Test slug generation for skills. """
        expected_slug = slugify("Django Skill Test")
        self.assertEqual(self.skill.slug, expected_slug)

    def test_get_absolute_url(self):
        """ Test the get_absolute_url method for skills. """
        expected_url = reverse('skills:skill_detail', kwargs={'slug': self.skill.slug})
        self.assertEqual(self.skill.get_absolute_url(), expected_url)

class SkillViewTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.cat_web = SkillCategory.objects.create(name="Web Development", order=0)
        cls.cat_data = SkillCategory.objects.create(name="Data Science", order=1)
        cls.skill_dj = Skill.objects.create(name="Django", category=cls.cat_web, description="Web framework")
        cls.skill_py = Skill.objects.create(name="Python", category=cls.cat_web, description="Programming language")
        cls.skill_sql = Skill.objects.create(name="SQL", description="Database language") # Uncategorized

    def setUp(self):
        self.client = Client()

    def test_skill_list_view(self):
        """ Test the skill list view. """
        url = reverse('skills:skill_list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'skills/skill_list.html')
        self.assertIn('categories', response.context)
        self.assertIn('uncategorized_skills', response.context)
        # Check if categories and skills are present
        self.assertEqual(len(response.context['categories']), 2)
        self.assertEqual(response.context['categories'][0], self.cat_web)
        self.assertEqual(len(response.context['uncategorized_skills']), 1)
        self.assertEqual(response.context['uncategorized_skills'][0], self.skill_sql)
        self.assertContains(response, self.skill_dj.name)
        self.assertContains(response, self.cat_data.name) # Category name should appear

    def test_skill_detail_view(self):
        """ Test the skill detail view. """
        url = reverse('skills:skill_detail', kwargs={'slug': self.skill_dj.slug})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'skills/skill_detail.html')
        self.assertEqual(response.context['skill'], self.skill_dj)
        self.assertContains(response, self.skill_dj.name)
        self.assertContains(response, self.skill_dj.description)
        self.assertContains(response, self.skill_dj.category.name)

    def test_skill_detail_view_404(self):
        """ Test skill detail view returns 404 for invalid slug. """
        url = reverse('skills:skill_detail', kwargs={'slug': 'non-existent-skill'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

