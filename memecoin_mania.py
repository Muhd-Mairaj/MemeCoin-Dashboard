from data_prep import prepare_data

"""# 5.1 LSTM Model Engineering

"""
# Get Cleaned Data
final_data = prepare_data()

train_data = final_data[final_data["coin"] == "pepe"].copy() #pepe, shiba_inu, dogecoin
train_data.drop(columns=["timestamp", "date", "coin"], inplace=True)
train_data.shape

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Scale the data to (0, 1)
scaler = MinMaxScaler()
train_data['close'] = scaler.fit_transform(train_data['close'].values.reshape(-1, 1))

# Create sequences for LSTM model
sequence_length = 50
X = []
y = []

for i in range(sequence_length, len(train_data)):
    X.append(train_data['close'].values[i-sequence_length:i])
    y.append(train_data['close'].values[i])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))






from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next price

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Training the model
model.fit(X, y, epochs=50, batch_size=32)





# Get the most recent data for prediction
# Extract last sequence_length rows
# Extract only the column used for training
test_data = train_data["close"].values[-sequence_length:]
# Reshape into (1, 50, 1)
test_data = test_data.reshape(1, sequence_length, 1).astype(np.float32)

# Predict the next price
predicted_price = model.predict(test_data)
predicted_price = scaler.inverse_transform(predicted_price)  # Undo scaling
print("Predicted PriceDate,Source,Keyword,Search_Score", predicted_price[0][0])

# Visualize predictions
import matplotlib.pyplot as plt

predictions = []
for i in range(len(X)):
    test_sequence = X[i].reshape(1, sequence_length, 1)
    prediction = model.predict(test_sequence)
    predictions.append(scaler.inverse_transform(prediction)[0][0])

plt.plot(scaler.inverse_transform(final_data['close'].values.reshape(-1, 1)), label="Actual Price")
plt.plot(np.arange(sequence_length, sequence_length + len(predictions)), predictions, label="Predicted Price")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# ✅ Step 1: Split features and target
X = train_data.drop(columns=["close"])
y = train_data["close"]

# ✅ Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 3: Scale features for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fill in missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Or median, most_frequent, etc.
X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)
X_test_scaled_imputed = imputer.fit_transform(X_test_scaled)
# Use X_train_scaled_imputed instead of X_train_scaled for training

X_test_scaled_imputed.shape

# ✅ SVR Model
svr = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.001)
svr.fit(X_train_scaled_imputed, y_train)
svr_preds = svr.predict(X_test_scaled_imputed)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_preds))

# ✅ Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# ✅ Results
print(f"SVR RMSE: {svr_rmse:.5f}")
print(f"Random Forest RMSE: {rf_rmse:.5f}")

import matplotlib.pyplot as plt
import numpy as np
# Predict using SVR and Random Forest
svr_preds = svr.predict(X_test_scaled_imputed)
rf_preds = rf.predict(X_test)

# Get test indices to locate actual prices in original data
_, X_test_indices = train_test_split(train_data.index, test_size=0.2, random_state=42)

# Sort the indices to plot in the correct time order
sorted_test_indices = np.sort(X_test_indices)

# Actual close prices for test set in correct order
actual_prices = train_data.loc[sorted_test_indices, "close"].values

# SVR and RF predictions need to be reordered according to sorted_test_indices
# But sklearn's train_test_split returns shuffled indices, so reorder preds accordingly:
# We'll reorder preds based on sorting the indices as train_test_split shuffles data
# The y_test, svr_preds, and rf_preds arrays correspond to X_test_indices order.
# So, get sorting order and reorder predictions accordingly:
sort_order = np.argsort(X_test_indices)

svr_preds_sorted = svr_preds[sort_order]
rf_preds_sorted = rf_preds[sort_order]

# Plot actual vs predictions
plt.figure(figsize=(14, 7))
plt.plot(sorted_test_indices, actual_prices, label="Actual Price", color='black', marker='o')
plt.plot(sorted_test_indices, svr_preds_sorted, label="SVR Predictions", color='blue', marker='x')
plt.plot(sorted_test_indices, rf_preds_sorted, label="Random Forest Predictions", color='green', marker='s')

plt.xlabel("Data Index (Time)")
plt.ylabel("Close Price")
plt.title("Actual vs SVR and Random Forest Predictions on Test Set")
plt.legend()
plt.grid(True)
plt.show()

"""# 5.1.1 Save Models"""

import joblib

# Save SVR model
joblib.dump(svr, 'svr_model.joblib')

# Save Random Forest model
joblib.dump(rf, 'rf_model.joblib')

# Assuming your LSTM model is called `model`
model.save('lstm_model.h5')  # saves architecture + weights + optimizer state

"""# 5.1.2 Grid Search Params for svr and random forest"""

from sklearn.model_selection import GridSearchCV

# SVR Hyperparameter Grid
svr_param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'epsilon': [0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

# Random Forest Hyperparameter Grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# SVR Grid Search
svr = SVR()
svr_grid = GridSearchCV(estimator=svr,
                        param_grid=svr_param_grid,
                        cv=5,
                        scoring='neg_mean_squared_error',
                        verbose=2,
                        n_jobs=-1)

X_scaled_imputed = imputer.fit_transform(X_train_scaled)

svr_grid.fit(X_scaled_imputed, y)
print("Best SVR params:", svr_grid.best_params_)
print("Best SVR RMSE:", np.sqrt(-svr_grid.best_score_))

# Random Forest Grid Search
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(estimator=rf,
                       param_grid=rf_param_grid,
                       cv=5,
                       scoring='neg_mean_squared_error',
                       verbose=2,
                       n_jobs=-1)

rf_grid.fit(X, y)
print("Best RF params:", rf_grid.best_params_)
print("Best RF RMSE:", np.sqrt(-rf_grid.best_score_))