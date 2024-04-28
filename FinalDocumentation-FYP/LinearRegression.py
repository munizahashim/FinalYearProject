import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Importing the dataset
dataset = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/Preprocessing Data/Data.csv', encoding='utf-8')

# Splitting the dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X_train, y_train)
# Initialize the model
linear_regression = LinearRegression()

# Fit the model to the training data
linear_regression.fit(X_train, y_train)

# Making predictions on the Test set
# Making predictions on the Test set
y_pred = linear_regression.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)  # R-squared calculation
rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)
print("R-squared:", r_squared)  # Printing the R-squared value
print("Root Mean Squared Error:", rmse) 
print("Mean Absolute Percentage Error:", mape, "%")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Save the model to disk
with open('linear_regression.pkl', 'wb') as file:
    pickle.dump(linear_regression, file)