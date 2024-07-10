# import pandas as pd
# data = pd.read_csv("dataset.csv")
# data.info()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Assuming you have the data loaded into a DataFrame named 'data'

# # Splitting the data into features (X) and target variables (y)
# X = data[['Nozzle Temperature (C)', 'Bed Temperature (C)', 'Fan Speed (RPM)', 'Print Speed (mm/s)']]
# y = data[['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion']]

# # Splitting the data into training and testing sets (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Linear Regression model
# linear_regressor = LinearRegression()

# # Train the model
# linear_regressor.fit(X_train, y_train)

# # Predict on the test set
# y_pred = linear_regressor.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R-squared Score:", r2)

# print(X_test)
# print(y_pred)

# # Example prediction for a new sample (replace with your own values)
# new_sample = pd.DataFrame({'Nozzle Temperature (C)': [222], 'Bed Temperature (C)': [63], 'Fan Speed (RPM)': [7000], 'Print Speed (mm/s)': [80]})
# new_pred = linear_regressor.predict(new_sample)
# print("Predicted Values for New Sample:")
# print(new_pred)

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Load the data (replace 'path_to_your_file.csv' with the actual file path)
# data = pd.read_csv('3D_dataset_.csv')

# # # Prepare the data
# X = data.drop(['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion', 'Stringing', 'Poor Adhesion'], axis=1)
# y = data[['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion', 'Stringing', 'Poor Adhesion']]

# # # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Create and train the linear regression model
# linear_regressor = LinearRegression()
# linear_regressor.fit(X_train, y_train)

# # # Make predictions on the test data
# y_pred = linear_regressor.predict(X_test)

# # # Round the predicted values to 0 or 1 based on a threshold (e.g., 0.5)
# y_pred_rounded = np.round(y_pred)

# y_pred_rounded = np.where(y_pred_rounded == -0.0, 0.0, y_pred_rounded)

# # # Print the rounded predicted values
# print(y_pred_rounded)

# # # Example of predicting with your own sample values
# sample_values = np.array([[220, 55, 4000, 60]])  # Adjust these values according to your sample
# sample_pred = linear_regressor.predict(sample_values)
# sample_pred_rounded = np.round(sample_pred)
# print("Predicted Values for Sample Data:")
# print(sample_pred_rounded)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

# Load the data (replace 'path_to_your_file.csv' with the actual file path)
data = pd.read_csv('3Dprinter_data.csv')

# Drop rows with NaN values in the target variable 'y'
data.dropna(subset=['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion','Stringing','Poor Adhesion'], inplace=True)

# Prepare the data
X = data[['Nozzle Temp (C)', 'Bed Temp (C)', 'Fan Speed (RPM)', 'Print Speed (mm/s)']]
y = data[['Warping', 'Nozzle Clogging', 'Over Extrusion', 'Under Extrusion','Stringing','Poor Adhesion']]

# Impute missing values in X
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Impute missing values in y_train and y_test (if any)
y_imputer = SimpleImputer(strategy='mean')
y_train_imputed = y_imputer.fit_transform(y_train)
y_test_imputed = y_imputer.transform(y_test)

# Create and train the linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train_imputed)

# Make predictions for the sample data
sample_values = np.array([[220, 50, 4000, 55]])  # Input sample data
sample_values_imputed = imputer.transform(sample_values)
sample_pred = linear_regressor.predict(sample_values_imputed)

# Thresholding predictions to get binary values
sample_pred_binary = np.where(sample_pred >= 0.5, 1, 0)

# Print the predicted output
print("Predicted Values (Warping, Nozzle Clogging, Over Extrusion, Under Extrusion, Stringing, Poor Adhesion):")
print(sample_pred_binary)

