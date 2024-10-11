# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For statistical data visualization

# Import scikit-learn modules
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LogisticRegression  # For logistic regression algorithm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)  # For evaluation metrics

# Load the dataset
# For demonstration purposes, let's create a synthetic dataset
# In practice, you would load your dataset using pd.read_csv or similar methods
from sklearn.datasets import make_classification  # To create a synthetic classification dataset

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000,  # Number of samples
                           n_features=10,   # Number of features
                           n_informative=5,  # Number of informative features
                           n_redundant=2,   # Number of redundant features
                           n_classes=2,     # Number of classes
                           random_state=42)  # For reproducibility

# Convert to DataFrame for easier manipulation
feature_names = [f'feature_{i}' for i in range(X.shape[1])]  # Create feature names
data = pd.DataFrame(X, columns=feature_names)  # Feature data
data['quality'] = y  # Target variable

# Explore the dataset
print(data.head())  # Display the first few rows of the dataset
print(data['quality'].value_counts())  # Check the distribution of the target variable

# Split the dataset into features and target variable
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,   # Features and target
                                                    test_size=0.3,  # 30% for testing
                                                    random_state=42)  # For reproducibility

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)  # max_iter increased for convergence

# Train the model
model.fit(X_train, y_train)  # Fit the model to the training data

# Make predictions on the test set
y_pred = model.predict(X_test)  # Predicted classes
y_prob = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
precision = precision_score(y_test, y_pred)  # Calculate precision
recall = recall_score(y_test, y_pred)  # Calculate recall
f1 = f1_score(y_test, y_pred)  # Calculate F1 score
auc = roc_auc_score(y_test, y_prob)  # Calculate AUC

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')  # Display accuracy
print(f'Precision: {precision:.2f}')  # Display precision
print(f'Recall: {recall:.2f}')  # Display recall
print(f'F1 Score: {f1:.2f}')  # Display F1 score
print(f'AUC: {auc:.2f}')  # Display AUC

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')  # Plot confusion matrix using seaborn
plt.title('Confusion Matrix')  # Title of the plot
plt.xlabel('Predicted')  # X-axis label
plt.ylabel('Actual')  # Y-axis label
plt.show()  # Display the plot

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Compute ROC curve data
plt.plot(fpr, tpr, label='Logistic Regression')  # Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Plot random guess line
plt.xlabel('False Positive Rate')  # X-axis label
plt.ylabel('True Positive Rate')  # Y-axis label
plt.title('Receiver Operating Characteristic (ROC) Curve')  # Title of the plot
plt.legend()  # Display legend
plt.show()  # Display the plot

# Print a detailed classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))  # Display precision, recall, f1-score, and support
