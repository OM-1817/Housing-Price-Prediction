# Housing Price Prediction

This project builds a Machine Learning model to predict **median house
values** using the California Housing dataset. It demonstrates a
complete ML workflow including data preprocessing, pipeline creation,
model training, saving the trained model, and making predictions.

## Project Overview

The project trains a **Random Forest Regressor** using Scikit‑Learn
pipelines. It handles both numerical and categorical features and saves
the trained model so predictions can be generated without retraining.

## Features

-   Data preprocessing using Scikit‑Learn Pipelines
-   Missing value handling with SimpleImputer
-   Feature scaling with StandardScaler
-   Categorical encoding using OneHotEncoder
-   Stratified dataset splitting
-   Model training using RandomForestRegressor
-   Model persistence using Joblib
-   Batch prediction on new data

## Technologies Used

-   Python
-   NumPy
-   Pandas
-   Scikit‑Learn
-   Joblib

## Project Structure

    project-folder
    │
    ├── housing.csv        # Original dataset
    ├── input.csv          # Test input data
    ├── output.csv         # Model predictions
    │
    ├── model.pkl          # Saved trained model
    ├── pipeline.pkl       # Saved preprocessing pipeline
    │
    └── main.py            # Main training and prediction script

## Installation

Install the required dependencies:

    pip install -r requirements.txt

## Usage

### 1. Train the Model

Place the dataset `housing.csv` in the project directory and run:

    python main.py

This will:

-   Train the model
-   Save `model.pkl`
-   Save `pipeline.pkl`
-   Generate `input.csv` for testing

### 2. Generate Predictions

Run the script again:

    python main.py

This will:

-   Load the saved model and pipeline
-   Predict housing prices
-   Save results to `output.csv`

## Output

The output file will contain predicted `median_house_value` for each row
in the input data.

## Future Improvements

-   Hyperparameter tuning
-   Model evaluation metrics (RMSE, MAE)
-   Visualization dashboards
-   REST API for predictions
-   Model deployment

## Author

Om Vaghani
