from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X is your feature matrix and y is your target variable
wine_data = pd.read_csv('wine123.csv')

# Separate features and target variable
X = wine_data.drop('Wine', axis=1)  # Features
y = wine_data['Wine']  # Target variable

# Number of variations
num_variations = 10

rf_accuracies = []
rf_mses = []
svm_accuracies = []
svm_mses = []

for i in range(num_variations):
    # Split the data into training and testing sets with a different random seed each time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Random Forest
    rf_model = RandomForestRegressor(random_state=i)  # Use the same random seed for model initialization
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_rounded = np.round(rf_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred_rounded)
    rf_mse = mean_squared_error(y_test, rf_pred)

    # Support Vector Machine (SVM)
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_pred_rounded = np.round(svm_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred_rounded)
    svm_mse = mean_squared_error(y_test, svm_pred)

    # Append the results to the lists
    rf_accuracies.append(rf_accuracy)
    rf_mses.append(rf_mse)
    svm_accuracies.append(svm_accuracy)
    svm_mses.append(svm_mse)

# Print the results
print("Random Forest Accuracies:", rf_accuracies)
print("Random Forest Mean Accuracy:", np.mean(rf_accuracies))
print("\nRandom Forest Mean Squared Errors:", rf_mses)
print("Random Forest Mean MSE:", np.mean(rf_mses))

print("\nSVM Accuracies:", svm_accuracies)
print("SVM Mean Accuracy:", np.mean(svm_accuracies))
print("\nSVM Mean Squared Errors:", svm_mses)
print("SVM Mean MSE:", np.mean(svm_mses))

# Scatter plot with regression line function
def plot_scatter_with_regression(y_true, y_pred, model_name):
    # Create a DataFrame for seaborn
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    
    # Scatter plot with regression line
    sns.regplot(x='Actual', y='Predicted', data=df, scatter_kws={'alpha':0.5})
    
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Actual Wine Quality')
    plt.ylabel('Predicted Wine Quality')
    plt.show()

# Random Forest Scatter Plot with Regression Line
plot_scatter_with_regression(y_test, rf_pred, 'Random Forest')

# SVM Scatter Plot with Regression Line
plot_scatter_with_regression(y_test, svm_pred, 'SVM')
