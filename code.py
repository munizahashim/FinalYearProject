import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the dataset
df = pd.read_csv('C:/Users/muniza.hashim/Desktop/senior/FYP/FYP progress/House Prices/house_kg_10K_ads.csv')

# Display the first 10 rows of the dataset
print(df.head(10))

# Get the shape of the dataset
print("Shape of the dataset:", df.shape)

# Count categorical variables
obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

# Count integer variables
int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

# Count float variables
fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Check for missing values
print("Missing values:")
print("=============================")
print(df.isnull().sum())

# Check the number of unique values in each column
print("=============================")
print("Number of unique values in each column:")
print(df.nunique())

# Drop rows with missing values in the 'district' column
df.dropna(subset=['district'], inplace=True)

# Check for duplicate rows and print the count
duplicate_rows = df.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())

# Extract the year from the 'date' column and create a new column 'date_year'
df['date_year'] = pd.to_datetime(df['date']).dt.year
print(df.head(1))

# Create a histogram for the 'price' column
df['price'].hist()

# Identify unique values in the 'rooms' column
unique_rooms = df['rooms'].unique()
print("Unique values in 'rooms' column:", unique_rooms)

# Replace 'свободная планировка' with 0 (free layout)
df.loc[df['rooms'] == "свободная планировка", 'rooms'] = 0

# Replace '6 и более' with 6 (6 and more)
df.loc[df['rooms'] == "6 и более", 'rooms'] = 6

# Convert the 'rooms' column to integer data type
df["rooms"] = df["rooms"].astype(int)

# Create a scatter plot for 'rooms' vs. 'price'
df.plot.scatter('rooms', 'price')

# Create a new column 'is_top_floor' based on floor and floors
df['is_top_floor'] = (df['floor'] == df['floors']) & (df['floor'] != 1)
df['is_bottom_floor'] = (df['floor'] == 1) & (df['floor'] != df['floors'])

# Create a bar plot for average price by floor type
avg_price_by_floor_type = df.groupby(['is_top_floor', 'is_bottom_floor'])['price'].mean()
avg_price_by_floor_type.plot(kind='bar', color='blue')
plt.xlabel('Floor Type')
plt.ylabel('Average Price')
plt.xticks([0, 1, 2], ['Not Top/Bottom', 'Top Floor', 'Bottom Floor'])
plt.title('Average Price Comparison by Floor Type')
plt.show()

# Group by 'district' and calculate the minimum price in each district
min_price_by_district = df.groupby('district')['price'].min()

# Calculate the total minimum price
total_min_price = min_price_by_district.sum()

# Calculate the proportion of each district's minimum price relative to the total minimum price
proportions = min_price_by_district / total_min_price

# Create a pie chart to visualize the proportion of minimum price by district
plt.pie(proportions, labels=min_price_by_district.index, autopct='%1.1f%%', startangle=140,
        colors=['blue', 'red', 'green', 'orange'])
plt.axis('equal')
plt.title('Proportion of Minimum Price by District')
plt.show()

# Define the number of top micro districts to display
top_n = 10

# Group by 'square' and calculate the average price, then select the top N micro districts
top_micro_districts = df.groupby('square')['price'].mean().sort_values(ascending=False).head(top_n)

# Create a bar plot to show the top micro districts by price
top_micro_districts.plot(kind='bar')
plt.xlabel('Square')
plt.ylabel('Average Price')
plt.title('Top Micro Districts by Price')
plt.xticks(rotation=45)
plt.show()

# Create a heatmap to visualize correlations in the dataset
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(21, 12))
sns.heatmap(numeric_df.corr(), cmap='BrBG', annot=True, fmt='.2f', linewidths=2)
plt.show()

# Function to show pair plots for numeric data
def show_graphs(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    sns.set(style="ticks")
    sns.pairplot(df[numeric_columns], diag_kind="kde", markers="o", plot_kws={"color": "orange"})
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Pairplot of Numeric Variables", fontsize=16)
    plt.show()

# Call the function to show the pair plot
show_graphs(numeric_df)

# Handling the 'year' column
print(f"Min year: {df['year'].min()}")
print(f"Max year: {df['year'].max()}")

# Fill missing values in 'year' with 0 and convert to integer
df['year'].fillna(0, inplace=True)
df["year"] = df["year"].astype(int)

# Define bins and labels for 'year' column
year_bin_names = ['missing', '1951-1960', '1961-1970', '1971-1980', '1981-1990',
                  '1991-2000', '2001-2010', '2011-2014', '2015-2020', '2021-2025']
year_bin = [0, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2014, 2020, 2025]

# Apply binning to 'year' and create a new column 'year_bin'
df['year'].fillna('missing', inplace=True)
df['year_bin'] = pd.cut(df['year'], bins=year_bin, labels=year_bin_names, right=True)

# Encoding categorical data in the 'district' column
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ... [previous code for loading and preprocessing data] ...

# Encoding categorical data in the 'district' column
labelencoder_X_1 = LabelEncoder()
df['district_encoded'] = labelencoder_X_1.fit_transform(df['district'])

# Encoding categorical data in the 'micro_district' column
labelencoder_X_2 = LabelEncoder()
df['micro_district_encoded'] = labelencoder_X_2.fit_transform(df['micro_district'])

# ... [other encoding and preprocessing steps] ...

# Calculate and fill the 'max_price_micro_district' column based on the maximum price in each micro district
df['max_price_micro_district'] = df.groupby('micro_district')['price'].transform('max')
df['max_price_micro_district'].fillna(0, inplace=True)

# One Hot Encoding for categorical columns
def encode_and_bind(df, encode_data):
    dummies = pd.get_dummies(df[[encode_data]])
    res = pd.concat([df, dummies], axis=1)
    cols = dummies.columns.tolist()
    return res, cols

# Apply One Hot Encoding to selected columns (district, building_type, condition, source, year_bin)
df, _ = encode_and_bind(df, 'district')
df, _ = encode_and_bind(df, 'building_type')
df, _ = encode_and_bind(df, 'condition')
df, _ = encode_and_bind(df, 'source')
df, _ = encode_and_bind(df, 'year_bin')

# Define independent and dependent variables
independent_variables = ["square", "rooms", "floors", "floor", "date_year", 'max_price_micro_district',
                         "district_encoded", "micro_district_encoded",
                        
                         "building_type_кирпичный", "building_type_монолитный",
                         "building_type_панельный", 'condition_под самоотделку (ПСО)',
                         'condition_хорошее', 'condition_среднее', 'condition_не достроено',
                         'condition_требует ремонта', 'condition_черновая отделка',
                         'condition_свободная планировка', 'source_Site', 'source_Android',
                         'source_iOS', 'year_bin_1951-1960', 'year_bin_1961-1970', 'year_bin_1971-1980',
                         'year_bin_1981-1990', 'year_bin_1991-2000', 'year_bin_2001-2010',
                         'year_bin_2011-2014', 'year_bin_2015-2020', 'year_bin_2021-2025'
                         ]
dependent_variable = ["price"]

# Standardize the selected columns
df[independent_variables] = (df[independent_variables] - df[independent_variables].mean()) / (df[independent_variables].std())
df[dependent_variable] = (df[dependent_variable] - df[dependent_variable].mean()) / (df[dependent_variable].std())

# Set up X and y based on dependent and independent variables
X = df[independent_variables]
X = sm.add_constant(X)
y = df[dependent_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Instantiate and fit a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate R-squared for the model using the test set
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Calculate Mean Squared Error (MSE) for the model using the test set
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Plot real vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Real Values')
plt.scatter(y_test, y_pred, color='red', alpha=0.5, label='Predicted Values')
plt.xlabel('Real Prices')
plt.ylabel('Predicted Prices')
plt.title('Real vs. Predicted Prices')
plt.legend()
plt.show()

# Instantiate and fit a Random Forest Regressor model
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = regressor_rf.predict(X_test)

# Calculate Mean Squared Error (MSE) for the Random Forest model using the test set
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Random Forest Mean Squared Error (MSE):", mse_rf)

# Calculate R-squared for the Random Forest model using the test set
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest R-squared:", r2_rf)

# Plot real vs. predicted prices for the Random Forest model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Real Values')
plt.scatter(y_test, y_pred_rf, color='red', alpha=0.5, label='Predicted Values (Random Forest)')
plt.xlabel('Real Prices')
plt.ylabel('Predicted Prices')
plt.title('Real vs. Predicted Prices (Random Forest)')
plt.legend()
plt.show()

# Instantiate and fit a Decision Tree Regressor model
regressor_dt = DecisionTreeRegressor(random_state=42)
regressor_dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = regressor_dt.predict(X_test)

# Calculate Mean Squared Error (MSE) for the Decision Tree model using the test set
mse_dt = mean_squared_error(y_test, y_pred_dt)
print("Decision Tree Mean Squared Error (MSE):", mse_dt)

# Calculate R-squared for the Decision Tree model using the test set
r2_dt = r2_score(y_test, y_pred_dt)
print("Decision Tree R-squared:", r2_dt)

# Plot real vs. predicted prices for the Decision Tree model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Real Values')
plt.scatter(y_test, y_pred_dt, color='red', alpha=0.5, label='Predicted Values (Decision Tree)')
plt.xlabel('Real Prices')
plt.ylabel('Predicted Prices')
plt.title('Real vs. Predicted Prices (Decision Tree)')
plt.legend()
plt.show()

# Save the trained Decision Tree Regressor model using pickle
with open('DecisionTreeRegressor.pkl', 'wb') as file:
    pickle.dump(regressor_dt, file)
