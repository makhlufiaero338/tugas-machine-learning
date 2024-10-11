# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset (replace 'FinancialMarket.csv' with the path to your file if needed)
data = pd.read_csv('FinancialMarket.csv')

# Preview the first few rows of the dataset
print(data.head())

# Split the dataset into features (X) and target (y)
X = data[['x']]  # Independent variable (feature)
y = data['combined_data']  # Dependent variable (target)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
# Initialize the linear regression model
linear_model = LinearRegression()

# Train the linear regression model on the training data
linear_model.fit(X_train, y_train)

# Predict the target values for the test data using the linear model
y_pred_linear = linear_model.predict(X_test)

# Calculate performance metrics for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Print the performance metrics for Linear Regression
print("Linear Regression Results:")
print(f"Mean Squared Error (MSE): {mse_linear}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear}")
print(f"R-squared (R2): {r2_linear}\n")

# 2. Polynomial (Basis Function) Regression Model with degree 2
# Create polynomial features (degree=2)
poly_features = PolynomialFeatures(degree=2)

# Transform the training and testing data into polynomial features
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Initialize the linear regression model (for polynomial features)
poly_model = LinearRegression()

# Train the polynomial regression model on the transformed training data
poly_model.fit(X_train_poly, y_train)

# Predict the target values for the test data using the polynomial model
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate performance metrics for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print the performance metrics for Polynomial Regression
print("Polynomial Regression Results (Degree 2):")
print(f"Mean Squared Error (MSE): {mse_poly}")
print(f"Root Mean Squared Error (RMSE): {rmse_poly}")
print(f"R-squared (R2): {r2_poly}")

# Comparison Summary
print("\nModel Comparison Summary:")
print(f"Linear Regression RMSE: {rmse_linear}, R2: {r2_linear}")
print(f"Polynomial Regression (Degree 2) RMSE: {rmse_poly}, R2: {r2_poly}")
