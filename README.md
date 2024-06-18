# IIT-GUWAHATI-DATAHACK

### Brief Explanation of Method and Approach

The objective of this code is to develop a predictive model for vaccine uptake using survey data. The model predicts the likelihood of respondents receiving two types of vaccines: xyz_vaccine and seasonal_vaccine. The following steps are taken:

1. **Data Loading**:
    - Features and targets are loaded from CSV files.
    - The two datasets are merged using `respondent_id` to form a comprehensive dataset.

2. **Feature and Target Separation**:
    - The merged dataset is divided into features (`X`) and target variables (`y`).
    - `respondent_id` and the target columns (`xyz_vaccine`, `seasonal_vaccine`) are excluded from the feature set.

3. **Feature Identification**:
    - Numerical and categorical features are identified based on their data types.

4. **Preprocessing Pipelines**:
    - Two preprocessing pipelines are created: one for numerical features and one for categorical features.
    - **Numerical Features**:
        - Missing values are imputed with the median.
        - Features are standardized using `StandardScaler`.
    - **Categorical Features**:
        - Missing values are imputed with the most frequent value.
        - Features are one-hot encoded to convert categorical variables into a format suitable for machine learning algorithms.

5. **Combining Preprocessing Steps**:
    - The preprocessing steps for numerical and categorical features are combined into a single `ColumnTransformer`.

6. **Model Pipeline**:
    - A pipeline is defined that first applies the preprocessing steps and then fits a `GradientBoostingClassifier` wrapped in a `MultiOutputClassifier` to handle the multi-label classification task.

7. **Training and Testing Split**:
    - The data is split into training and testing sets using `train_test_split`.

8. **Model Training**:
    - The model is trained on the training data.

9. **Prediction and Evaluation**:
    - The model predicts probabilities for the test set.
    - The ROC AUC score is calculated to evaluate the model's performance.

10. **Preparation for Submission**:
    - Test data is preprocessed similarly to the training data.
    - Predictions are made for the test data.
    - The results are formatted into a submission DataFrame and saved to a CSV file.

### Detailed Steps:

1. **Data Loading**:
    - Load feature and target datasets.
    - Merge the datasets on `respondent_id` to create a unified dataset.

2. **Feature and Target Separation**:
    - Drop the `respondent_id` and target columns from the features.
    - Store the targets in a separate DataFrame.

3. **Feature Identification**:
    - Identify columns containing numerical data (`int64`, `float64`).
    - Identify columns containing categorical data (`object`).

4. **Preprocessing Pipelines**:
    - For numerical features, handle missing values by imputing the median and standardize the features.
    - For categorical features, impute missing values with the most frequent value and apply one-hot encoding.

5. **Combining Preprocessing Steps**:
    - Use `ColumnTransformer` to apply the numerical and categorical preprocessing pipelines to their respective features.

6. **Model Pipeline**:
    - Create a pipeline that first preprocesses the data and then fits a multi-output gradient boosting classifier.

7. **Training and Testing Split**:
    - Use `train_test_split` to divide the data into training and testing sets.

8. **Model Training**:
    - Fit the model on the training data.

9. **Prediction and Evaluation**:
    - Predict probabilities for the test data.
    - Evaluate the model using the ROC AUC score to assess its performance.

10. **Preparation for Submission**:
    - Apply the same preprocessing steps to the test data.
    - Predict probabilities for the test data.
    - Format the results into a submission file and save it.
