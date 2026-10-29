# Review Sentiment Studio

**An interactive sentiment analysis application for Amazon product reviews, powered by TF-IDF and a PyTorch neural network.**

Review Sentiment Studio classifies product reviews into **Negative, Neutral, or Positive** sentiment. It provides an interactive interface for analyzing individual reviews, viewing predicted class probabilities, and processing multiple reviews in batches.

## Overview

Understanding customer feedback is important for evaluating product experiences and identifying recurring issues. This project explores how Natural Language Processing (NLP) and machine learning can be used to classify the sentiment expressed in product reviews.

The project follows an end-to-end machine learning workflow, including exploratory data analysis, text preprocessing, TF-IDF feature extraction, model training, evaluation, and application development.

## Features

- **Single Review Analysis:** Enter or paste a product review and receive a sentiment prediction.
- **Three-Class Classification:** Predict Negative, Neutral, or Positive sentiment.
- **Class Probabilities:** View the model's predicted probability for each sentiment class.
- **Example Reviews:** Try sample reviews to explore the application's behaviour.
- **Batch Analysis:** Upload a CSV containing reviews or paste multiple reviews for batch processing.
- **Prediction Explanations:** View word-level indicators intended to help interpret the prediction.
- **Interactive Interface:** Use a dark-themed dashboard built for convenient review analysis.

## Application Preview

The interface includes three main sections:

- **Analyze Review:** Enter a review and inspect its predicted sentiment and class probabilities.
- **Batch Analysis:** Process multiple reviews and inspect their predictions.
- **About the Model:** Explore information about the model and its approach.

## Machine Learning Approach

### 1. Dataset and Exploratory Data Analysis

The project uses a labelled product-review dataset with review text and sentiment labels.

Exploratory analysis and data cleaning were performed to understand the dataset, examine missing values and duplicates, and inspect the distribution of sentiment classes.

### 2. Text Representation

Review text is converted into numerical features using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

TF-IDF represents reviews using weighted word features that can be used by machine learning models.

### 3. Models

Two approaches were explored:

- **Logistic Regression:** A traditional machine learning baseline.
- **PyTorch Multilayer Perceptron (MLP):** A feed-forward neural network trained on TF-IDF features.

### 4. Evaluation

Models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Because the sentiment classes are imbalanced, macro F1 and class-wise metrics were also considered.

The MLP's final reported test results were:

| Metric | Test Score |
|---|---:|
| Accuracy | 0.77 |
| Macro F1 | 0.61 |
| Weighted F1 | 0.80 |
| Negative F1 | 0.64 |
| Neutral F1 | 0.30 |
| Positive F1 | 0.87 |

These results indicate that the model performs better on Positive reviews than on Neutral reviews. Neutral sentiment remains a key limitation of the current model.

## Technology Stack

- **Python** — core programming language
- **Pandas and NumPy** — data handling
- **scikit-learn** — TF-IDF, Logistic Regression, and evaluation metrics
- **PyTorch** — neural network implementation and training
- **Joblib** — saving and loading the fitted TF-IDF vectorizer
- **Streamlit** — interactive application interface
- **Git and GitHub** — version control and source code hosting

## Project Structure

```text
amazon-reviews-sentiment-analysis/
├── data/
│   ├── cleaned_reviews.csv
│   └── Reviews.csv
├── models/
│   ├── sentiment_mlp.pth
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_pytorch_mlp.ipynb
│   └── 05_model_comparison.ipynb
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
└── README.md
```

*The structure above describes the main project files; individual filenames may differ depending on the final repository contents.*

## Run Locally

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies
- Git

### 1. Clone the repository

```bash
git clone https://github.com/mohithgowdadb/amazon-reviews-sentiment-analysis.git
cd amazon-reviews-sentiment-analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

The application should open in your browser at the local URL shown in the terminal.

Ensure the trained model and fitted TF-IDF vectorizer are available at the paths expected by `app.py`.

## Limitations and Future Improvements

- Improve classification of Neutral reviews.
- Explore class weighting and other approaches to class imbalance.
- Experiment with alternative text representations and neural network architectures.
- Evaluate calibration and reliability of predicted probabilities.
- Improve prediction explanations and validate the method used to generate word-level indicators.
- Add more robust input validation and error handling.
- Deploy the application so that it can be accessed publicly.

## Learning Outcomes

This project provided practical experience with:

- Exploratory data analysis and data cleaning
- Text preprocessing and TF-IDF feature extraction
- Traditional machine learning and neural networks
- PyTorch training and evaluation
- Imbalanced multiclass classification
- Model persistence and inference pipelines
- Building an interactive machine learning application

## Author

**Mohith Gowda D B**

GitHub: [@mohithgowdadb](https://github.com/mohithgowdadb)

---

