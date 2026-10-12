# customer-churn-prediction-
# Customer Churn Prediction

## Overview

Customer Churn Prediction is a machine learning project designed to predict whether a customer is likely to leave a service.

The project combines a trained machine learning pipeline with a Streamlit interface, allowing users to enter customer information and obtain a prediction.

## Features

* Machine learning-based churn prediction.
* Interactive Streamlit user interface.
* Saved model pipeline for making predictions.
* Python-based machine learning workflow.
* Jupyter Notebook for exploring the model development process.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Project Structure

* `app.py` — Streamlit application.
* `churn_prediction_corrected.ipynb` — Machine learning notebook.
* `churn_prediction_pipeline.joblib` — Saved trained pipeline.
* `churn_project_metadata.joblib` — Saved project metadata.
* `requirements.txt` — Python dependencies.

## How to Run

### 1. Download the repository

Clone the repository or download it as a ZIP file from GitHub.

### 2. Install the required libraries

Open a terminal in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the application

Run:

```bash
python -m streamlit run app.py
```

The application should open in your web browser.

## Purpose

This project demonstrates how machine learning can be combined with an interactive interface to explore customer churn prediction.

Businesses can use churn predictions as one input when identifying customers who may need additional attention or retention efforts.

## Disclaimer

Predictions depend on the model, the training data, and the customer information provided. The application is intended for educational and demonstration purposes.
