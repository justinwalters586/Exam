# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Step 2: Read the Excel file using pandas
df = pd.read_excel('C:\\Users\\WaltersJ07\\Downloads\\Restaurant Revenue.xlsx')

# Step 3: Check for any missing values and handle them (if any)
df = df.dropna()

# Step 4: Split the data into features (X) and the label (y)
X = df[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = df['Monthly_Revenue']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Create a multiple linear regression model using sklearn
model = LinearRegression()

# Step 7: Train the model using the training data
model.fit(X_train, y_train)

# Step 8: Evaluate the model using the testing data
y_pred = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Step 9: Print the regression coefficients and intercept
print('Coefficients: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Calculate the R-squared value for the model
r_squared = model.score(X_test, y_test)
print('R-squared:', r_squared)

print("Go Brewers")