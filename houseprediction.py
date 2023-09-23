# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'dataset.csv' with your dataset file path)
data = pd.read_csv('dataset.csv')

# Extract features (square footage, bedrooms, bathrooms) and target variable (house prices)
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict house prices for new data
# Replace the values below with the features of the new house
X_new = [[2000, 3, 2]]  # Example values for square footage, bedrooms, and bathrooms
predicted_price = model.predict(X_new)
print("Predicted House Price:", predicted_price[0])

# Visualize the predicted prices alongside actual prices
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual Prices vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Create a pair plot for all columns
sns.pairplot(data)
plt.show()
