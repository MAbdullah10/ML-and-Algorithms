import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
train_data = pd.read_csv('./train.csv', low_memory=False)
test_data = pd.read_csv('./test.csv', low_memory=False)

# Display column names
print("Training Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# Assume 'Credit_Score' is the target variable for this example
target_column = 'Credit_Score'

# Ensure 'target' column is present in the training data
if target_column not in train_data.columns:
    raise KeyError(f"'{target_column}' column not found in training data")

# Separate features and target variable from training data
X_train = train_data.drop(target_column, axis=1)
y_train = train_data[target_column]

# Separate features and target variable from test data
if target_column in test_data.columns:
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
else:
    X_test = test_data.copy()
    y_test = None

# Preprocessing for numerical data
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Preprocess the test data
X_test_processed = preprocessor.transform(X_test)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the preprocessed training data
model.fit(X_train_processed, y_train)

# Predict on the preprocessed test data
y_pred_train = model.predict(X_train_processed)
y_pred_test = model.predict(X_test_processed)

# Evaluate the model on the training set
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Print training set performance
print(f"Training Mean Squared Error: {train_mse}")
print(f"Training R-squared: {train_r2}")

# Evaluate the model on the test set if true labels are available
if y_test is not None:
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    # Print test set performance
    print(f"Test Mean Squared Error: {test_mse}")
    print(f"Test R-squared: {test_r2}")

# Get feature importances from the model
importances = model.feature_importances_

# Get the feature names
feature_names = numerical_features.tolist() + preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Additional Analysis (if y_test is available)
if y_test is not None:
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of perfect prediction
    plt.xlabel('Actual Credit Score')
    plt.ylabel('Predicted Credit Score')
    plt.title('Actual vs Predicted Credit Score')
    plt.grid(True)
    plt.show()
