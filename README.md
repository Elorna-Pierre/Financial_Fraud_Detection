# Financial Fraud Detection Project

### Overview:

The purpose of this project is to detect fraudulent transactions using machine learning techniques. The dataset contains various transaction details, and the focus is on identifying fraud based on key features like transaction type, amount, and account balances.

### Dataset:

- Source: Cleaned financial transaction dataset

 - Key Features:

    - step: Represents time in hours from the start of the dataset

    - type: Transaction type (e.g., TRANSFER, CASH_OUT)

    - amount: Transaction amount

    - oldbalanceOrg: Balance before the transaction

    - newbalanceOrig: Balance after the transaction

    - oldbalanceDest: Destination account balance before the transaction

    - newbalanceDest: Destination account balance after the transaction

    - isFraud: Fraud label (1 = Fraudulent, 0 = Not Fraud)

### Data Preprocessing:

- Focused on TRANSFER and CASH_OUT transactions (where fraud is most common)

- Converted categorical variables to numerical format

- Scaled numerical features using StandardScaler

- Created an HourOfDay feature from the step column to analyze hourly transaction trends

- Sampled 10% of the dataset for faster processing

### Exploratory Data Analysis (EDA): 

- Visualizations:

   - Bar and line plots of hourly transaction activity

   - Box plots to analyze transaction amounts

   - Scatter plots to examine balance changes

   - Confusion matrices for model evaluation

- Model Training & Evaluation

   - Models Used:

    - Random Forest Classifier: Used as a baseline model

    - Gradient Boosting Classifier: Used for improved predictive performance

   - Hyperparameter Tuning:

    - Applied RandomizedSearchCV to optimize model parameters

Selected best parameters for improved F1-score

Performance Metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix visualization

Results

The optimized Gradient Boosting Classifier achieved the best F1-score

Important features influencing fraud detection include transaction amount, balance differences, and transaction type

Future Improvements

Implement deep learning techniques for enhanced fraud detection

Incorporate real-time fraud detection using streaming data

Experiment with anomaly detection methods

How to Run

Ensure you have Python installed along with required libraries (pandas, numpy, matplotlib, seaborn, sklearn)

Run the data preprocessing notebook to clean and prepare the dataset

Execute the model training notebook to train and evaluate fraud detection models
