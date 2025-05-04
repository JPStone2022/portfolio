# demos/views.py

import os
import io # For handling dataframe info in memory
import uuid # For unique filenames
import base64
from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm, SentimentAnalysisForm, CSVUploadForm, ExplainableAIDemoForm # Import new form
import numpy as np
# Import Demo model
from .models import Demo

# --- TensorFlow / Keras Imports (for Image Classification) ---
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    
    TF_AVAILABLE = True
    try:
        # Load model only if needed for the image demo
        # Consider lazy loading if memory is a concern
        image_model = MobileNetV2(weights='imagenet')
        IMAGE_MODEL_LOADED = True
    except Exception as e:
        print(f"Error loading MobileNetV2 model: {e}")
        IMAGE_MODEL_LOADED = False
        image_model = None # Define as None on error
except ImportError:
    print("TensorFlow not found. Image Classification demo disabled.")
    TF_AVAILABLE = False
    IMAGE_MODEL_LOADED = False
    image_model = None # Define as None if TF not available

# --- Hugging Face Transformers Imports (for Sentiment Analysis) ---
sentiment_pipeline = None # Initialize as None at module level
SENTIMENT_MODEL_LOADED = False
TRANSFORMERS_AVAILABLE = False
try:
    # Use pipeline for easy sentiment analysis
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    try:
        # Load pipeline once on startup (or use caching/lazy loading)
        # Using a distilled version for potentially faster/smaller footprint
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        SENTIMENT_MODEL_LOADED = True
        print("Sentiment analysis pipeline loaded successfully.") # Confirmation
    except Exception as e:
        print(f"Error loading sentiment analysis pipeline: {e}")
        # sentiment_pipeline remains None
        SENTIMENT_MODEL_LOADED = False
except ImportError:
    print("Transformers library not found. Install using 'pip install transformers[torch]' or 'transformers[tf]'. Sentiment Analysis demo disabled.")
    # sentiment_pipeline remains None
    TRANSFORMERS_AVAILABLE = False
    SENTIMENT_MODEL_LOADED = False

# --- Pandas / Matplotlib / Seaborn Imports ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib # Use Agg backend for non-interactive plotting
    matplotlib.use('Agg')
    DATA_LIBS_AVAILABLE = True
except ImportError:
    print("Pandas, Matplotlib, or Seaborn not found. Data Analysis demo disabled.")
    DATA_LIBS_AVAILABLE = False

# --- Scikit-learn Imports (for XAI Demo) ---
try:
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn import tree # For plotting if needed later
    SKLEARN_AVAILABLE = True
    # Load Iris data and train a simple tree ONCE on startup
    # In production, load a pre-saved model instead.
    try:
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        # Train a simple Decision Tree
        decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42) # Limit depth for simplicity
        decision_tree_model.fit(X_iris, y_iris)
        TREE_MODEL_LOADED = True
        print("Decision Tree model trained successfully.")
    except Exception as e:
        print(f"Error loading Iris data or training Decision Tree: {e}")
        TREE_MODEL_LOADED = False
        decision_tree_model = None
        iris = None

except ImportError:
    print("Scikit-learn not found. Explainable AI demo disabled.")
    SKLEARN_AVAILABLE = False
    TREE_MODEL_LOADED = False
    decision_tree_model = None
    iris = None

# Statsmodels (for Causal Inference Demo)
try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Statsmodels not found. Causal Inference demo disabled."); STATSMODELS_AVAILABLE = False

# SciPy (for Optimization Demo)
try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    print("SciPy not found. Optimization demo disabled."); SCIPY_AVAILABLE = False


# --- All Demos List View (NEW) ---
def all_demos_view(request):
    """ Displays a list of all available interactive demos. """
    all_demos_list = Demo.objects.all() # Fetch all, ordered by model Meta
    context = {
        'demos': all_demos_list,
        'page_title': 'All Interactive Demos',
        'meta_description': "Explore interactive demonstrations showcasing various data science, machine learning, and AI concepts.",
        'meta_keywords': "demo, interactive, machine learning, data science, AI, portfolio",
    }
    return render(request, 'demos/all_demos.html', context=context)



# --- Image Classification View ---
def image_classification_view(request):
    form = ImageUploadForm()
    prediction_results = None
    uploaded_image_url = None
    error_message = None
    uploaded_image_path = None

    if not TF_AVAILABLE: error_message = "TensorFlow library is not installed."
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
                predictions = image_model.predict(img_preprocessed) # Use image_model
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
                # Check if pipeline object actually exists before calling
                if sentiment_pipeline:
                    results = sentiment_pipeline(submitted_text)
                    if results:
                        sentiment_result = results[0] # Get the first result dictionary
                        sentiment_result['score'] = round(sentiment_result['score'] * 100, 1)
                    else:
                        error_message = "Could not analyze sentiment for the provided text."
                else:
                    # This case should ideally not be reached if MODEL_LOADED is True, but added for safety
                    error_message = "Sentiment analysis pipeline is not available."

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



# --- Data Analysis View (NEW) ---
def data_analysis_view(request):
    form = CSVUploadForm()
    analysis_results = None
    error_message = None
    plot_image_url = None

    if not DATA_LIBS_AVAILABLE:
        error_message = "Required libraries (Pandas, Matplotlib, Seaborn) not installed."

    if request.method == 'POST' and DATA_LIBS_AVAILABLE:
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']

            # Basic validation (size, type)
            if csv_file.size > 5 * 1024 * 1024: # Max 5 MB
                error_message = "File size exceeds 5MB limit."
            elif not csv_file.name.lower().endswith('.csv'):
                error_message = "Invalid file type. Please upload a CSV file."
            else:
                try:
                    # Read CSV into Pandas DataFrame
                    df = pd.read_csv(csv_file)

                    # --- Perform Basic Analysis ---
                    # 1. Get DataFrame Info (capture output)
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    df_info = buffer.getvalue()

                    # 2. Get Descriptive Statistics (convert to HTML)
                    df_describe_html = None
                    try:
                        # Select only numeric columns for describe()
                        numeric_df = df.select_dtypes(include=np.number)
                        if not numeric_df.empty:
                             df_describe_html = numeric_df.describe().to_html(
                                 classes='w-full text-sm text-left text-gray-500 dark:text-gray-400 border border-collapse border-gray-200 dark:border-gray-700', # Tailwind classes
                                 border=0 # Remove default border
                             )
                    except Exception as desc_e:
                        print(f"Error generating describe table: {desc_e}")


                    # 3. Generate a Plot (Example: Histogram of the first numerical column)
                    plot_filename = None
                    numerical_cols = df.select_dtypes(include=np.number).columns
                    if not numerical_cols.empty:
                        col_to_plot = numerical_cols[0] # Plot the first numerical column
                        plt.figure(figsize=(8, 4)) # Create a figure
                        sns.histplot(df[col_to_plot], kde=True)
                        plt.title(f'Distribution of {col_to_plot}')
                        plt.xlabel(col_to_plot)
                        plt.ylabel('Frequency')
                        plt.tight_layout()

                        # Save plot to a temporary file in media
                        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_demos')
                        os.makedirs(temp_dir, exist_ok=True)
                        plot_filename = f"plot_{uuid.uuid4()}.png"
                        plot_filepath = os.path.join(temp_dir, plot_filename)
                        plt.savefig(plot_filepath)
                        plt.close() # Close the figure to free memory

                        plot_image_url = os.path.join(settings.MEDIA_URL, 'temp_demos', plot_filename).replace("\\", "/")
                    else:
                        print("No numerical columns found for plotting.")


                    # --- Prepare results for template ---
                    analysis_results = {
                        'filename': csv_file.name,
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'head': df.head().to_html(classes='w-full text-sm text-left text-gray-500 dark:text-gray-400', border=0, index=False),
                        'info': df_info,
                        'describe_html': df_describe_html,
                        'plot_url': plot_image_url
                    }

                except pd.errors.EmptyDataError:
                    error_message = "The uploaded CSV file is empty."
                except Exception as e:
                    error_message = f"Error processing CSV file: {e}"
        else:
            error_message = "Invalid form submission. Please upload a valid CSV file."

    context = {
        'form': form,
        'analysis_results': analysis_results,
        'error_message': error_message,
        'page_title': 'Simple CSV Data Analyzer',
    }
    return render(request, 'demos/data_analysis_demo.html', context=context)


# --- Data Wrangling View (NEW) ---
def data_wrangling_view(request):
    form = CSVUploadForm() # Reuse the CSV upload form
    wrangling_results = None
    error_message = None

    if not DATA_LIBS_AVAILABLE:
        error_message = "Required libraries (Pandas, NumPy) not installed."

    if request.method == 'POST' and DATA_LIBS_AVAILABLE:
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = form.cleaned_data['csv_file']

            # Basic validation
            if csv_file.size > 5 * 1024 * 1024: # Max 5 MB
                error_message = "File size exceeds 5MB limit."
            elif not csv_file.name.lower().endswith('.csv'):
                error_message = "Invalid file type. Please upload a CSV file."
            else:
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file)
                    original_head_html = df.head().to_html(classes='w-full text-sm text-left text-gray-500 dark:text-gray-400', border=0, index=False)
                    original_columns = df.columns.tolist()
                    original_shape = df.shape

                    # --- Apply Wrangling Steps ---
                    steps_applied = []
                    df_wrangled = df.copy() # Work on a copy

                    # 1. Handle Missing Numerical Values (Example: fill with median)
                    numeric_cols = df_wrangled.select_dtypes(include=np.number).columns
                    for col in numeric_cols:
                        if df_wrangled[col].isnull().any():
                            median_val = df_wrangled[col].median()
                            df_wrangled[col].fillna(median_val, inplace=True)
                            steps_applied.append(f"Filled missing values in numerical column '{col}' with median ({median_val:.2f}).")

                    # 2. Handle Missing Categorical Values (Example: fill with 'Unknown')
                    categorical_cols = df_wrangled.select_dtypes(include='object').columns
                    for col in categorical_cols:
                        if df_wrangled[col].isnull().any():
                            df_wrangled[col].fillna('Unknown', inplace=True)
                            steps_applied.append(f"Filled missing values in categorical column '{col}' with 'Unknown'.")

                    # 3. Rename a Column (Example: if 'QuantitySold' exists)
                    if 'QuantitySold' in df_wrangled.columns:
                        df_wrangled.rename(columns={'QuantitySold': 'Units_Sold'}, inplace=True)
                        steps_applied.append("Renamed column 'QuantitySold' to 'Units_Sold'.")
                    elif 'Quantity' in df_wrangled.columns: # Alternative common name
                            df_wrangled.rename(columns={'Quantity': 'Units_Sold'}, inplace=True)
                            steps_applied.append("Renamed column 'Quantity' to 'Units_Sold'.")


                    # 4. Create a Derived Column (Example: Price Category)
                    if 'Price' in df_wrangled.columns:
                        # Ensure Price is numeric first
                        df_wrangled['Price'] = pd.to_numeric(df_wrangled['Price'], errors='coerce')
                        df_wrangled.dropna(subset=['Price'], inplace=True) # Drop rows where conversion failed

                        bins = [0, 50, 200, np.inf] # Define price ranges
                        labels = ['Low', 'Medium', 'High']
                        df_wrangled['Price_Category'] = pd.cut(df_wrangled['Price'], bins=bins, labels=labels, right=False)
                        steps_applied.append("Created 'Price_Category' column based on 'Price' (Low: <50, Medium: 50-199, High: >=200).")

                    # --- Prepare results ---
                    wrangled_head_html = df_wrangled.head().to_html(classes='w-full text-sm text-left text-gray-500 dark:text-gray-400', border=0, index=False)

                    wrangling_results = {
                        'filename': csv_file.name,
                        'original_shape': original_shape,
                        'original_columns': original_columns,
                        'original_head': original_head_html,
                        'wrangled_shape': df_wrangled.shape,
                        'wrangled_columns': df_wrangled.columns.tolist(),
                        'wrangled_head': wrangled_head_html,
                        'steps_applied': steps_applied,
                    }

                except pd.errors.EmptyDataError:
                    error_message = "The uploaded CSV file is empty."
                except Exception as e:
                    error_message = f"Error processing CSV file: {e}"
                    print(f"Wrangling Error: {e}") # Log error for debugging
        else:
            error_message = "Invalid form submission. Please upload a valid CSV file."

    context = {
        'form': form,
        'wrangling_results': wrangling_results,
        'error_message': error_message,
        'page_title': 'Simple Data Wrangling Demo',
    }
    return render(request, 'demos/data_wrangling_demo.html', context=context)


# --- Explainable AI View (UPDATED) ---
def explainable_ai_view(request):
    form = ExplainableAIDemoForm()
    prediction = None
    # prediction_proba = None # Replaced by probability_list
    explanation_rules = None
    feature_importances = None
    input_features_dict = None # Store cleaned form data
    probability_list = None # New list to hold zipped data
    error_message = None

    if not SKLEARN_AVAILABLE: error_message = "Scikit-learn library not installed."
    elif not TREE_MODEL_LOADED: error_message = "Decision Tree model or Iris data could not be loaded."

    if request.method == 'POST' and SKLEARN_AVAILABLE and TREE_MODEL_LOADED:
        form = ExplainableAIDemoForm(request.POST)
        if form.is_valid():
            input_features_dict = form.cleaned_data # Store for display
            try:
                input_features_array = np.array([[ # Create numpy array for prediction
                    input_features_dict['sepal_length'],
                    input_features_dict['sepal_width'],
                    input_features_dict['petal_length'],
                    input_features_dict['petal_width']
                ]])

                pred_index = decision_tree_model.predict(input_features_array)[0]
                prediction = iris.target_names[pred_index]
                prediction_proba_raw = decision_tree_model.predict_proba(input_features_array)[0] * 100

                # *** Zip probabilities and names together HERE ***
                probability_list = [
                    {'name': name, 'probability': prob}
                    for name, prob in zip(iris.target_names, prediction_proba_raw)
                ]
                # *** End Zipping ***

                # --- Generate Explanation: Decision Path ---
                node_indicator = decision_tree_model.decision_path(input_features_array)
                leaf_id = decision_tree_model.apply(input_features_array)
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
                explanation_rules = []
                tree_ = decision_tree_model.tree_
                feature_names = iris.feature_names
                for i, node_id in enumerate(node_index):
                    if leaf_id[0] == node_id:
                        value = tree_.value[node_id][0]; class_dist = [f"{iris.target_names[i]} ({int(v)})" for i, v in enumerate(value)]
                        explanation_rules.append(f"<b>Leaf Node {node_id}:</b> Reached Leaf. Prediction based on majority class here (counts: {', '.join(class_dist)}).")
                        break
                    feature_idx = tree_.feature[node_id]; threshold = round(tree_.threshold[node_id], 2)
                    feature_name = feature_names[feature_idx]; input_val = input_features_array[0, feature_idx]
                    if input_val <= threshold:
                        decision = f"<= {threshold}"; next_node_id = tree_.children_left[node_id]
                        explanation_rules.append(f"<b>Node {node_id}:</b> Check <i>{feature_name}</i> ({input_val:.2f}). Is {input_val:.2f} {decision}? <b>Yes</b>. Go to Node {next_node_id}.")
                    else:
                        decision = f"> {threshold}"; next_node_id = tree_.children_right[node_id]
                        explanation_rules.append(f"<b>Node {node_id}:</b> Check <i>{feature_name}</i> ({input_val:.2f}). Is {input_val:.2f} {decision}? <b>No</b>. Go to Node {next_node_id}.")
                # --- End Explanation ---

                importances = decision_tree_model.feature_importances_
                feature_importances = sorted(zip(feature_names, importances * 100), key=lambda x: x[1], reverse=True)

            except Exception as e: error_message = f"Error during prediction or explanation: {e}"; print(f"XAI Error: {e}")
        else: error_message = "Please enter valid numerical values for all features."

    context = {
        'form': form,
        'prediction': prediction,
        # 'prediction_proba': prediction_proba, # Removed
        'probability_list': probability_list, # Pass the zipped list instead
        'explanation_rules': explanation_rules,
        'feature_importances': feature_importances,
        'input_features': input_features_dict, # Pass cleaned dict
        # 'target_names': iris.target_names if iris else [], # No longer needed separately
        'error_message': error_message,
        'page_title': 'Explainable AI (Decision Tree)',
    }
    return render(request, 'demos/explainable_ai_demo.html', context=context)


# --- Flask API Demo View (NEW) ---
def flask_api_demo_view(request):
    """ Renders the page explaining Flask for simple ML APIs. """
    context = {
        'page_title': 'Flask for Machine Learning APIs',
        'meta_description': "Learn how the Flask microframework is often used to create simple APIs for serving machine learning models.",
        'meta_keywords': "Flask, API, machine learning, deployment, Python, microframework",
    }
    return render(request, 'demos/flask_api_demo.html', context=context)

# --- Django Concepts Demo View (NEW) ---
def django_concepts_demo_view(request):
    """ Renders the page explaining key Django concepts. """
    context = {
        'page_title': 'Key Django Concepts',
        'meta_description': "Explore key features of the Django web framework, including its ORM, Admin, Forms, and Template system.",
        'meta_keywords': "Django, web framework, ORM, admin, forms, templates, Python",
    }
    return render(request, 'demos/django_concepts_demo.html', context=context)

# --- Causal Inference Demo View (NEW) ---
def causal_inference_demo_view(request):
    """
    Demonstrates Causal Inference using Regression Adjustment
    on simulated marketing campaign data.
    """
    results = None
    error_message = None
    plot_url = None

    # Check dependencies
    if not DATA_LIBS_AVAILABLE or not STATSMODELS_AVAILABLE or not SKLEARN_AVAILABLE:
         error_message = "Required libraries (Pandas, Statsmodels, Scikit-learn, Matplotlib) not installed."
         context = {'error_message': error_message, 'page_title': 'Causal Inference Demo'}
         return render(request, 'demos/causal_inference_demo.html', context)

    try:
        # 1. Simulate Data with Confounding
        np.random.seed(42) # for reproducibility
        n_customers = 1000
        # Confounder: 'engagement_score' (influences both treatment and outcome)
        engagement_score = np.random.normal(50, 15, n_customers).clip(1, 100)
        # Treatment Assignment (Promotion): More engaged customers are more likely to get promo
        prob_promo = 1 / (1 + np.exp(-( -2.5 + 0.05 * engagement_score))) # Sigmoid function
        received_promo = (np.random.rand(n_customers) < prob_promo).astype(int) # 1 if promo, 0 otherwise
        # Outcome (Spending): Depends on engagement, promo (true effect=20), and noise
        true_ate = 20
        spending = 50 + 0.8 * engagement_score + true_ate * received_promo + np.random.normal(0, 10, n_customers)
        spending = spending.clip(10) # Min spending of 10

        df = pd.DataFrame({
            'customer_id': range(n_customers),
            'engagement': engagement_score.round(1),
            'received_promo': received_promo, # Treatment (0 or 1)
            'spending': spending.round(2)      # Outcome
        })

        # 2. Naive Comparison (Incorrect due to confounding)
        naive_diff = df[df['received_promo'] == 1]['spending'].mean() - \
                     df[df['received_promo'] == 0]['spending'].mean()

        # 3. Regression Adjustment
        # Model: spending ~ engagement + received_promo
        # Fit OLS model using statsmodels
        # Use C(received_promo) if you want explicit categorical treatment
        ols_formula = 'spending ~ engagement + received_promo'
        ols_model = smf.ols(formula=ols_formula, data=df).fit()

        # Predict potential outcomes
        # Predict spending if EVERYONE received promo
        df_promo_all = df.assign(received_promo=1)
        pred_spending_if_promo = ols_model.predict(df_promo_all)

        # Predict spending if NO ONE received promo
        df_no_promo_all = df.assign(received_promo=0)
        pred_spending_if_no_promo = ols_model.predict(df_no_promo_all)

        # Calculate Average Treatment Effect (ATE)
        ate_estimate = (pred_spending_if_promo - pred_spending_if_no_promo).mean()

        # 4. Prepare results for template
        results = {
            'n_customers': n_customers,
            'naive_difference': round(naive_diff, 2),
            'confounder_info': "Higher engagement scores increase both the chance of receiving a promotion AND baseline spending.",
            'method_used': "Regression Adjustment",
            'regression_formula': ols_formula,
            'ate_estimate': round(ate_estimate, 2),
            'true_ate': true_ate, # For comparison in the demo
            'ols_summary': ols_model.summary().as_html() # Get model summary as HTML table
        }

        # 5. Generate a simple plot (Optional)
        try:
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=df, x='engagement', y='spending', hue='received_promo', alpha=0.6)
            plt.title('Spending vs Engagement (Colored by Promo)')
            plt.xlabel("Engagement Score")
            plt.ylabel("Customer Spending ($)")
            plt.legend(title='Received Promo', loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # Save plot to buffer and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close() # Close the figure
            results['plot_url'] = f"data:image/png;base64,{plot_url}"

        except Exception as plot_e:
            print(f"Error generating plot: {plot_e}")
            results['plot_url'] = None # Handle plot error gracefully

    except Exception as e:
        error_message = f"An error occurred during analysis: {e}"
        print(f"Causal Inference Error: {e}") # Log for debugging

    context = {
        'results': results,
        'error_message': error_message,
        'page_title': 'Causal Inference Demo',
        'meta_description': "Demonstration of causal inference using regression adjustment to estimate treatment effects in the presence of confounding.",
        'meta_keywords': "causal inference, regression adjustment, ATE, confounding, data science, demo",
    }
    return render(request, 'demos/causal_inference_demo.html', context=context)

# --- Optimization Demo View (NEW) ---
def optimization_demo_view(request):
    """
    Demonstrates finding the minimum of a function using SciPy's optimize module.
    """
    results = None
    error_message = None
    plot_url = None

    if not SCIPY_AVAILABLE or not DATA_LIBS_AVAILABLE or not np:
        error_message = "Required libraries (SciPy, NumPy, Matplotlib) not installed."
        context = {'error_message': error_message, 'page_title': 'Optimization with SciPy'}
        return render(request, 'demos/optimization_demo.html', context)

    try:
        # 1. Define the function to minimize (Himmelblau's function)
        # This function has multiple local minima.
        def himmelblau(p):
            x, y = p
            # f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
            term1 = (x**2 + y - 11)**2
            term2 = (x + y**2 - 7)**2
            return term1 + term2

        function_str = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2"
        start_point = np.array([0.0, 0.0]) # Where the optimization starts

        # 2. Perform Optimization
        # Use scipy.optimize.minimize. 'Nelder-Mead' is a common gradient-free method.
        optimization_result = optimize.minimize(
            himmelblau,
            start_point,
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'disp': False} # Tolerance and display options
        )

        # 3. Prepare results
        if optimization_result.success:
            found_minimum_x = optimization_result.x
            found_minimum_value = optimization_result.fun
            results = {
                'function': function_str,
                'start_point': start_point.tolist(),
                'method': 'Nelder-Mead',
                'success': optimization_result.success,
                'message': optimization_result.message,
                'found_minimum_point': [round(coord, 4) for coord in found_minimum_x],
                'found_minimum_value': round(found_minimum_value, 4),
                'iterations': optimization_result.nit,
            }
        else:
            error_message = f"Optimization failed: {optimization_result.message}"
            results = {'success': False, 'message': optimization_result.message}


        # 4. Generate Contour Plot
        try:
            x_range = np.arange(-5.0, 5.0, 0.1)
            y_range = np.arange(-5.0, 5.0, 0.1)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            Z_grid = himmelblau([X_grid, Y_grid])

            plt.figure(figsize=(7, 6))
            # Use contourf for filled contours, contour for lines
            contour_plot = plt.contourf(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 3, 15), cmap='viridis', alpha=0.8)
            plt.colorbar(contour_plot, label='Function Value (log scale)')
            plt.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 3, 15), colors='white', linewidths=0.5, alpha=0.5)

            # Mark known minima for Himmelblau's function
            known_minima = [
                (3.0, 2.0),
                (-2.805118, 3.131312),
                (-3.779310, -3.283186),
                (3.584428, -1.848126)
            ]
            for km in known_minima:
                plt.plot(km[0], km[1], 'r*', markersize=10, label='Known Minimum' if km == known_minima[0] else "")

            # Mark start point and found minimum
            plt.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start Point')
            if optimization_result.success:
                plt.plot(found_minimum_x[0], found_minimum_x[1], 'yo', markersize=8, label='Found Minimum')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("Optimization of Himmelblau's Function")
            plt.legend(fontsize='small')
            plt.grid(True, linestyle=':', alpha=0.4)
            plt.axis('equal') # Ensure aspect ratio is equal

            # Save plot to buffer and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close() # Close the figure
            if results: results['plot_url'] = f"data:image/png;base64,{plot_url}"

        except Exception as plot_e:
            print(f"Error generating plot: {plot_e}")
            if results: results['plot_url'] = None # Handle plot error gracefully

    except Exception as e:
        error_message = f"An error occurred during optimization setup: {e}"
        print(f"Optimization Demo Error: {e}") # Log for debugging

    context = {
        'results': results,
        'error_message': error_message,
        'page_title': 'Optimization with SciPy',
        'meta_description': "Demonstration of finding function minima using SciPy's optimization tools.",
        'meta_keywords': "scipy, optimization, minimize, Nelder-Mead, Himmelblau, data science, demo",
    }
    return render(request, 'demos/optimization_demo.html', context=context)

# --- Django Security Demo View (NEW) ---
def django_security_demo_view(request):
    """ Renders the page explaining key Django security features. """
    context = {
        'page_title': 'Django Security Features',
        'meta_description': "Learn about built-in security features in the Django framework like CSRF protection, XSS prevention, and SQL injection prevention.",
        'meta_keywords': "Django, security, web framework, CSRF, XSS, SQL injection, protection",
    }
    return render(request, 'demos/django_security_demo.html', context=context)



# --- Django Testing Demo View (NEW) ---
def django_testing_demo_view(request):
    """ Renders the page explaining Django's testing framework. """
    context = {
        'page_title': 'Django Testing Framework',
        'meta_description': "Learn how Django's built-in testing framework helps ensure application reliability by testing models, views, and forms.",
        'meta_keywords': "Django, testing, unit testing, TestCase, web framework, Python",
    }
    return render(request, 'demos/django_testing_demo.html', context=context)

# --- AI Tools Demo View (NEW) ---
def ai_tools_demo_view(request):
    """ Renders the page explaining the use and risks of AI tools in development. """
    context = {
        'page_title': 'AI Tools in Machine Learning and Data Science Development',
        'meta_description': "Exploring the benefits and potential dangers of using AI code assistants and large language models in machine learning and data science workflows.",
        'meta_keywords': "AI tools, LLM, code assistant, Copilot, ChatGPT, machine learning, data science, productivity, risks, ethics",
    }
    return render(request, 'demos/ai_tools_demo.html', context=context)

# --- Python Concepts Demo View (NEW) ---
def python_concepts_demo_view(request):
    """ Renders the page explaining/demonstrating core Python concepts. """
    context = {
        'page_title': 'Core Python Concepts for ML/DS',
        'meta_description': "Interactive examples showcasing Python lists, dictionaries, loops, and functions and their relevance to data science and machine learning.",
        'meta_keywords': "Python, core concepts, data structures, list, dictionary, function, loop, machine learning, data science, demo",
    }
    return render(request, 'demos/python_concepts_demo.html', context=context)


# --- R Concepts Demo View (NEW) ---
def r_concepts_demo_view(request):
    """ Renders the page explaining core R concepts for Data Science. """
    context = {
        'page_title': 'R Language for Data Science',
        'meta_description': "Explore key concepts and packages in the R programming language commonly used for statistical analysis, data visualization, and machine learning.",
        'meta_keywords': "R language, data science, statistics, dplyr, ggplot2, data visualization, machine learning, demo",
    }
    return render(request, 'demos/r_concepts_demo.html', context=context)


# --- Go Concepts Demo View (NEW) ---
def go_concepts_demo_view(request):
    """ Renders the page explaining Go's role in ML/DS infrastructure. """
    context = {
        'page_title': 'Go (Golang) in ML/DS Infrastructure',
        'meta_description': "Learn how the Go programming language is used for building performant backend systems, APIs, and infrastructure components supporting machine learning workflows.",
        'meta_keywords': "Go, Golang, machine learning, data science, infrastructure, performance, concurrency, API",
    }
    return render(request, 'demos/go_concepts_demo.html', context=context)


# --- Scala Concepts Demo View (NEW) ---
def scala_concepts_demo_view(request):
    """ Renders the page explaining Scala's role, especially with Spark. """
    context = {
        'page_title': 'Scala in Big Data & ML Ecosystem',
        'meta_description': "Learn how Scala, running on the JVM, is used with Apache Spark for large-scale data processing and machine learning tasks.",
        'meta_keywords': "Scala, Spark, big data, machine learning, data engineering, JVM, functional programming",
    }
    return render(request, 'demos/scala_concepts_demo.html', context=context)


# --- Java Concepts Demo View (NEW) ---
def java_concepts_demo_view(request):
    """ Renders the page explaining Java's role in the ML/DS ecosystem. """
    context = {
        'page_title': 'Java in Big Data & Enterprise AI',
        'meta_description': "Learn how the Java programming language and its ecosystem are used in large-scale data processing (Hadoop, Spark), enterprise systems, and for deploying ML models.",
        'meta_keywords': "Java, machine learning, data science, big data, Hadoop, Spark, enterprise, JVM",
    }
    return render(request, 'demos/java_concepts_demo.html', context=context)


# --- Language Comparison Demo View (NEW) ---
def language_comparison_demo_view(request):
    """ Renders the page comparing languages used in ML/DS/AI. """
    context = {
        'page_title': 'Languages in ML, AI & Data Science',
        'meta_description': "Comparing the roles and use cases of Python, R, Scala, Java, C++, and other languages in the machine learning, AI, and data science ecosystem.",
        'meta_keywords': "programming languages, Python, R, Scala, Java, C++, machine learning, data science, AI, comparison",
    }
    return render(request, 'demos/language_comparison_demo.html', context=context)


# --- Ruby Concepts Demo View (NEW) ---
def ruby_concepts_demo_view(request):
    """ Renders the page explaining Ruby's role relative to ML/DS. """
    context = {
        'page_title': 'Ruby & the ML/AI/DS Ecosystem',
        'meta_description': "Understanding the role of the Ruby programming language, primarily known for web development (Ruby on Rails), in relation to data science and AI.",
        'meta_keywords': "Ruby, Ruby on Rails, machine learning, data science, AI, web development",
    }
    return render(request, 'demos/ruby_concepts_demo.html', context=context)


# --- OOP Concepts Demo View (NEW) ---
def oop_concepts_demo_view(request):
    """ Renders the page explaining OOP concepts and their relevance to ML/DS. """
    context = {
        'page_title': 'Demo: OOP Concepts in ML/DS',
        'meta_description': "Understanding Object-Oriented Programming (OOP) principles like classes, objects, inheritance, and encapsulation and their application in Python for data science and machine learning.",
        'meta_keywords': "OOP, Object-Oriented Programming, Python, machine learning, data science, classes, objects, inheritance, encapsulation, polymorphism",
    }
    return render(request, 'demos/oop_concepts_demo.html', context=context)


# --- Kotlin Concepts Demo View (NEW) ---
def kotlin_concepts_demo_view(request):
    """ Renders the page explaining Kotlin's role, mainly in Android ML integration. """
    context = {
        'page_title': 'Demo: Kotlin & On-Device AI (Android)',
        'meta_description': "Understanding Kotlin's primary role in Android development and how machine learning models (like TensorFlow Lite) are integrated for on-device AI features.",
        'meta_keywords': "Kotlin, Android, machine learning, AI, TensorFlow Lite, on-device ML, mobile AI",
    }
    return render(request, 'demos/kotlin_concepts_demo.html', context=context)


# --- Jupyter Demo View (NEW) ---
def jupyter_demo_view(request):
    """ Renders the page explaining Jupyter Notebooks for ML/DS. """
    context = {
        'page_title': 'Demo: Jupyter Notebooks in ML/DS',
        'meta_description': "Understanding the use of Jupyter Notebooks for interactive data exploration, analysis, model prototyping, and sharing results in data science and machine learning.",
        'meta_keywords': "Jupyter Notebook, data science, machine learning, interactive computing, Python, R, Julia, EDA",
    }
    return render(request, 'demos/jupyter_demo.html', context=context)


# --- PySpark Concepts Demo View (NEW) ---
def pyspark_concepts_demo_view(request):
    """ Renders the page explaining PySpark for Big Data ML/DS. """
    context = {
        'page_title': 'Demo: PySpark for Big Data & ML',
        'meta_description': "Understanding PySpark (Python API for Apache Spark) and its use for distributed data processing, analysis, and machine learning on large datasets.",
        'meta_keywords': "PySpark, Spark, big data, distributed computing, data engineering, machine learning, data science, Python",
    }
    return render(request, 'demos/pyspark_concepts_demo.html', context=context)


# --- PyTorch Concepts Demo View (NEW) ---
def pytorch_concepts_demo_view(request):
    """ Renders the page explaining PyTorch for Deep Learning. """
    context = {
        'page_title': 'Demo: PyTorch for Deep Learning',
        'meta_description': "Understanding the PyTorch framework, its core concepts like Tensors and Autograd, and its use in building and training neural networks for AI and machine learning.",
        'meta_keywords': "PyTorch, deep learning, machine learning, AI, tensors, autograd, neural networks, Python",
    }
    return render(request, 'demos/pytorch_concepts_demo.html', context=context)


# --- Data Security Demo View (NEW) ---
def data_security_demo_view(request):
    """ Renders the page explaining security considerations in ML/DS/AI. """
    context = {
        'page_title': 'Security in ML, AI & Data Science',
        'meta_description': "Understanding key security risks and considerations in machine learning, AI, and data science, including data privacy, model security, and infrastructure protection.",
        'meta_keywords': "security, machine learning, data science, AI, data privacy, model security, adversarial attacks, infrastructure security",
    }
    return render(request, 'demos/data_security_demo.html', context=context)


# --- Ethical Hacking Concepts Demo View (NEW) ---
def ethical_hacking_demo_view(request):
    """ Renders the page explaining ethical hacking concepts relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Ethical Hacking Mindset for AI/ML Security',
        'meta_description': "Understanding how ethical hacking principles apply to securing data pipelines, machine learning models, and AI infrastructure against potential threats.",
        'meta_keywords': "ethical hacking, security, machine learning, AI, data science, adversarial attacks, penetration testing, vulnerability",
    }
    return render(request, 'demos/ethical_hacking_demo.html', context=context)

# --- Django to Heroku Deployment Guide View (NEW) ---
def deploying_to_heroku_view(request):
    """ Renders the page Django to Heroku Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django to Heroku Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django to Heroku Deployment Guide.",
        'meta_keywords': "Django, Heroku, Deployment, Guide",
    }
    return render(request, 'demos/deploying_django_app_to_heroku.html', context=context)

# --- Django to Render Deployment Guide View (NEW) ---
def deploying_to_render_view(request):
    """ Renders the page Django to Render Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django to Render Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django to Render Deployment Guide.",
        'meta_keywords': "Django, Render, Deployment, Guide",
    }
    return render(request, 'demos/deploying_django_app_to_render.html', context=context)

# --- Django to Python Anywhere Deployment Guide View (NEW) ---
def deploying_to_python_anywhere_view(request):
    """ Renders the page Django to Python Anywhere Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django to Python Anywhere Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django to Python Anywhere Deployment Guide.",
        'meta_keywords': "Django, Python Anywhere, Deployment, Guide",
    }
    return render(request, 'demos/deploying_django_app_to_pythonanywhere.html', context=context)

# --- Django to Google App Engine Deployment Guide View (NEW) ---
def deploying_to_google_app_engine_view(request):
    """ Renders the page Django to Google App Engine Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django to Google App Engine Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django to Google App Engine Deployment Guide.",
        'meta_keywords': "Django, Google App Engine, Deployment, Guide",
    }
    return render(request, 'demos/deploying_django_app_to_google_app_engine.html', context=context)

# --- Django to AWS Elastic Beanstalk Deployment Guide View (NEW) ---
def deploying_to_aws_elastic_beanstalk_view(request):
    """ Renders the page Django to AWS Elastic Beanstalk Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django to AWS Elastic Beanstalk Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django to AWS Elastic Beanstalk Deployment Guide.",
        'meta_keywords': "Django, AWS Elastic Beanstalk, Deployment, Guide",
    }
    return render(request, 'demos/deploying_django_app_to_aws_elastic_beanstalk.html', context=context)

# --- Django Deployment Options View (NEW) ---
def deploying_options_view(request):
    """ Renders the page Django Deployment Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django Deployment Guide for AI/ML Security',
        'meta_description': "Understanding how Django Deployment Guide.",
        'meta_keywords': "Django, Deployment, Options, Guide",
    }
    return render(request, 'demos/deploying_django_options.html', context=context)

# --- Django Deployment Comparisons View (NEW) ---
def deploying_comparisons_view(request):
    """ Renders the page Django Deployment Comparisons Guide relevant to ML/DS/AI. """
    context = {
        'page_title': 'Demo: Django Deployment Comparisons Guide for AI/ML Security',
        'meta_description': "Understanding Django Deployment Comparisons Guide.",
        'meta_keywords': "Django, Deployment, Comparisons, Guide",
    }
    return render(request, 'demos/deploying_django_comparisons.html', context=context)

