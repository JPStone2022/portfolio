# demos/views.py

import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm, SentimentAnalysisForm # Import new form

# --- TensorFlow / Keras Imports (for Image Classification) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    import numpy as np
    TF_AVAILABLE = True
    try:
        # Load model only if needed for the image demo
        # Consider lazy loading if memory is a concern
        image_model = MobileNetV2(weights='imagenet')
        IMAGE_MODEL_LOADED = True
    except Exception as e:
        print(f"Error loading MobileNetV2 model: {e}")
        IMAGE_MODEL_LOADED = False
except ImportError:
    print("TensorFlow not found. Image Classification demo disabled.")
    TF_AVAILABLE = False
    IMAGE_MODEL_LOADED = False

# --- Hugging Face Transformers Imports (for Sentiment Analysis) ---
try:
    # Use pipeline for easy sentiment analysis
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    try:
        # Load pipeline once on startup (or use caching/lazy loading)
        # Using a distilled version for potentially faster/smaller footprint
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        SENTIMENT_MODEL_LOADED = True
    except Exception as e:
        print(f"Error loading sentiment analysis pipeline: {e}")
        SENTIMENT_MODEL_LOADED = False
except ImportError:
    print("Transformers library not found. Install using 'pip install transformers[torch]' or 'transformers[tf]'. Sentiment Analysis demo disabled.")
    TRANSFORMERS_AVAILABLE = False
    SENTIMENT_MODEL_LOADED = False


# --- Image Classification View ---
def image_classification_view(request):
    form = ImageUploadForm()
    prediction_results = None
    uploaded_image_url = None
    error_message = None
    uploaded_image_path = None

    if not TF_AVAILABLE: error_message = "TensorFlow library not installed."
    elif not IMAGE_MODEL_LOADED: error_message = "Image classification model could not be loaded."

    if request.method == 'POST' and TF_AVAILABLE and IMAGE_MODEL_LOADED:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_demos')
            os.makedirs(temp_dir, exist_ok=True)
            fs = FileSystemStorage(location=temp_dir)
            safe_filename = fs.get_valid_name(uploaded_image.name)
            filename = fs.save(safe_filename, uploaded_image)
            uploaded_image_path = fs.path(filename)
            uploaded_image_url = os.path.join(settings.MEDIA_URL, 'temp_demos', filename).replace("\\", "/")

            try:
                img = image.load_img(uploaded_image_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_array_expanded)
                predictions = image_model.predict(img_preprocessed)
                decoded = decode_predictions(predictions, top=3)[0]
                prediction_results = [{'label': label.replace('_', ' '), 'probability': float(prob) * 100} for (_, label, prob) in decoded]
            except Exception as e:
                error_message = f"Error processing image or making prediction: {e}"
                if uploaded_image_path and os.path.exists(uploaded_image_path):
                    try: os.remove(uploaded_image_path)
                    except OSError as oe: print(f"Error removing temp file {uploaded_image_path}: {oe}")
                uploaded_image_url = None
        else: error_message = "Invalid form submission. Please upload a valid image."

    context = { 'form': form, 'prediction_results': prediction_results, 'uploaded_image_url': uploaded_image_url, 'error_message': error_message, 'page_title': 'Image Classification Demo', }
    return render(request, 'demos/image_classification_demo.html', context=context)


# --- Sentiment Analysis View (NEW) ---
def sentiment_analysis_view(request):
    form = SentimentAnalysisForm()
    sentiment_result = None
    submitted_text = None
    error_message = None

    if not TRANSFORMERS_AVAILABLE:
        error_message = "Transformers library not installed. This demo cannot function."
    elif not SENTIMENT_MODEL_LOADED:
        error_message = "Sentiment analysis model could not be loaded. Please check server logs."

    if request.method == 'POST' and TRANSFORMERS_AVAILABLE and SENTIMENT_MODEL_LOADED:
        form = SentimentAnalysisForm(request.POST)
        if form.is_valid():
            submitted_text = form.cleaned_data['text_input']
            try:
                # Run text through the pipeline
                # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
                results = sentiment_pipeline(submitted_text)
                if results:
                    sentiment_result = results[0] # Get the first result dictionary
                    # Convert score to percentage for display
                    sentiment_result['score'] = round(sentiment_result['score'] * 100, 1)
                else:
                    error_message = "Could not analyze sentiment for the provided text."

            except Exception as e:
                error_message = f"Error during sentiment analysis: {e}"
        else:
            error_message = "Please enter some text to analyze."

    context = {
        'form': form,
        'sentiment_result': sentiment_result,
        'submitted_text': submitted_text,
        'error_message': error_message,
        'page_title': 'Sentiment Analysis Demo',
    }
    return render(request, 'demos/sentiment_analysis_demo.html', context=context)

