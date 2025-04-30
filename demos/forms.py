# demos/forms.py
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload an Image',
        help_text='(JPEG, PNG, etc.)',
        widget=forms.ClearableFileInput(attrs={
            'class': 'block w-full text-sm text-gray-500 dark:text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 dark:file:bg-blue-900 file:text-blue-700 dark:file:text-blue-300 hover:file:bg-blue-100 dark:hover:file:bg-blue-800 cursor-pointer border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none'
        })
    )

# New form for Sentiment Analysis
class SentimentAnalysisForm(forms.Form):
    text_input = forms.CharField(
        label='Enter Text for Analysis',
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400 transition-colors duration-300 ease-in-out',
            'rows': 5,
            'placeholder': 'Type or paste text here...'
        }),
        max_length=1000, # Limit input length
        required=True
    )

