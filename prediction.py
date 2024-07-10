import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt  # Import for plotting

# Load data (replace with your actual data loading)
data = pd.read_csv('3Dprinter_data.csv')

features = ['Nozzle Temp (C)', 'Bed Temp (C)', 'Fan Speed (RPM)', 'Print Speed (mm/s)']
target = 'Vibrations'

X = data[features]
y = data[target]

# Impute missing values
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM model
import lightgbm as lgb

model = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train)

# Generate future data
future_data = []
for day in range(60):
    for hour in range(24):
        # Generate estimated printing parameters for each day/hour
        future_data.append([220 + np.random.randint(-10, 10),
                            60 + np.random.randint(-5, 5),
                            4000,  # Fan Speed (replace with your estimate)
                            60])  # Print Speed (replace with your estimate)

# future_data_imputed = imputer.transform(future_data)

# Predict future vibrations
predicted_future_vibrations = model.predict(future_data)

# Generate sample data for actual vibrations (replace with your actual data)
actual_vibrations = [y.iloc[i] for i in range(len(y))]  # Assuming data is in order

# Plot actual vs predicted vibrations
plt.plot(range(len(actual_vibrations)), actual_vibrations, label='Actual Vibrations')
plt.plot(range(len(predicted_future_vibrations)), predicted_future_vibrations, label='Predicted Future Vibrations')
plt.xlabel('Time Step')
plt.ylabel('Vibration Level')
plt.title('Actual vs Predicted Vibrations')
plt.legend()
plt.show()
