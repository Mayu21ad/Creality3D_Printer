






import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Read the data
data = pd.read_csv("3Dprinter_data.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features (X) and target variable (y)
X = data.drop(['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion', 'Stringing', 'Poor Adhesion'], axis=1)
y = data[['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion', 'Stringing', 'Poor Adhesion']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict on a sample data point
sample_values = {'Nozzle Temp (C)': [215], 'Bed Temp (C)': [65], 'Fan Speed (RPM)': [7000], 'Print Speed (mm/s)': [60]}
sample_df = pd.DataFrame(sample_values)

y_sample_pred = dt_regressor.predict(sample_df)
print("Predicted Values (Warping, Nozzle Clogging, Over Extrusion, Under Extrusion, Stringing, Poor Adhesion):")
print(y_sample_pred)
