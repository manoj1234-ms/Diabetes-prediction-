# Diabetes Prediction using Machine Learning

A machine learning project that predicts whether a person has diabetes based on various health metrics using Support Vector Machine (SVM) classifier.

## Dataset Description

The dataset used in this project is the [Pima Indians Diabetes Database](diabetes.csv) which contains the following features:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration 
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0: No diabetes, 1: Has diabetes)

## Project Overview

This project implements:
- Data preprocessing using StandardScaler
- Training/test split (80-20)
- SVM classifier with linear kernel
- Model evaluation using accuracy metrics

## Model Performance

The model achieves:
- Training accuracy: 78.3%
- Test accuracy: 77.3%

## Requirements

```python
# Required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

## Usage

1. Load and preprocess the data:
```python
# Load the dataset
db = pd.read_csv('diabetes.csv')

# Split features and target
X = db.drop(columns='Outcome', axis=1)
y = db['Outcome']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

2. Train the model:
```python
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Create and train the classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
```

3. Make predictions:
```python
# Example input data
input_data = (7,147,76,0,0,39.4,0.257,43)
input_array = np.asarray(input_data).reshape(1,-1)
std_data = scaler.transform(input_array)
prediction = classifier.predict(std_data)
```

## Future Improvements

- Implement cross-validation
- Try different ML algorithms
- Handle class imbalance
- Feature selection/engineering
- Hyperparameter tuning
