import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb


# Importing the dataset
dataset = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/Preprocessing Data/Data.csv', encoding='utf-8')

# Splitting the dataset into independent (X) and dependent (y) variables
X = dataset.drop('condition', axis=1).values  # Drop the 'condition' column to get the features
y = dataset['condition'].values               # Set the 'condition' column as the target

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model 1
# Gradient Boosting Algorithm
# Initialize the model
#gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model to the training data
#gradient_boosting_regressor.fit(X_train, y_train)

# Making predictions on the Test set
#y_pred = gradient_boosting_regressor.predict(X_test)


#model 2
# Random Forest Regressor 

# Initialize the model with n_estimators as the number of trees
#random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


# Fit the model to the training data
#random_forest_regressor.fit(X_train, y_train)

# Making predictions on the Test set
#y_pred = random_forest_regressor.predict(X_test)


#model 3
# LGBM Regressor

# Initialize the model
lgb_regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model to the training data
lgb_regressor.fit(X_train, y_train)


# Making predictions on the Test set
y_pred = lgb_regressor.predict(X_test)


# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Calculate R-squared

print("Mean Absolute Error:", mae)
print("R-squared:", r2)  # This gives an indication of the goodness of fit 

