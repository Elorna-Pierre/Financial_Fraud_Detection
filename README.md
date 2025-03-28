

# Financial Fraud Detection Project

## Overview:

The purpose of this project is to detect fraudulent transactions using machine learning techniques. The dataset contains various transaction details, and the focus is on identifying fraud based on key features like transaction type, amount, and account balances.
Dataset:

* Source: Cleaned financial transaction dataset
* Key Features:
    * step: Represents time in hours from the start of the dataset
    * type: Transaction type (e.g., TRANSFER, CASH_OUT)
    * amount: Transaction amount
    * oldbalanceOrg: Balance before the transaction
    * newbalanceOrig: Balance after the transaction
    * oldbalanceDest: Destination account balance before the transaction
    * newbalanceDest: Destination account balance after the transaction
    * isFraud: Fraud label (1 = Fraudulent, 0 = Not Fraud)


## Data Preprocessing:

* Focused on TRANSFER and CASH_OUT transactions (where fraud is most common)
* Converted categorical variables to numerical format
* Scaled numerical features using StandardScaler
* Sampled 30% of the dataset for faster processing


## Exploratory Data Analysis (EDA):

* Visualizations:
    * Bar and line plots of hourly transaction activity
    * Pair plots to analyze transaction amounts
    * Scatter plots to examine balance changes
    * Confusion matrices for model evaluation

![alt text](image-3.png) 



* Model Training & Evaluation
    * Models Used:
    * Random Forest Classifier: Used as a baseline model
    * Gradient Boosting Classifier: Used for improved predictive performance
    * Hyperparameter Tuning:
    * Applied RandomizedSearchCV to optimize model parameters
    * Selected best parameters for improved F1-score
* Performance Metrics:
    * Accuracy
    * Precision
    * Recall
    * F1-score
    * Confusion Matrix visualization

## Results

* The optimized Gradient Boosting Classifier achieved the best F1-score
* Important features influencing fraud detection include transaction amount, balance differences, and transaction type

![alt text](image.png) 

<prev>
Random Forest Performance:
Accuracy: 0.9994
F1 Score: 0.8840
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    165726
           1       0.99      0.80      0.88       499

    accuracy                           1.00    166225
   macro avg       0.99      0.90      0.94    166225
weighted avg       1.00      1.00      1.00    166225
</prev>

## Future Improvements

* Use deep learning to improve fraud detection.
* Detect fraud in real time with streaming data.
* Test different anomaly detection methods.

## How to Run

1. Install Python libaries (pandas, numpy, matplotlib, seaborn, sklearn, scipy etc)
2. Run the data preprocessing notebook to clean and prepare the dataset
3. Execute the model training notebook to train and evaluate fraud detection models