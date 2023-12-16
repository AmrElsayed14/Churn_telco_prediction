import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Create and train a CNN model
import tensorflow as tf
from tensorflow.keras import models, layers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# Load the customer churn dataset
costumers_dataset = pd.read_csv("Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop the customer ID column
costumers_dataset.drop("customerID", axis = 'columns', inplace = True)

# Check if the 'TotalCharges' column contains any non-numeric values
non_numeric_charges = pd.to_numeric(costumers_dataset.TotalCharges, errors="coerce").isnull()

# If there are any non-numeric values in the 'TotalCharges' column, replace them with NaN
if sum(non_numeric_charges) > 0:
    costumers_dataset.TotalCharges.replace(' ', np.nan, inplace=True)
    costumers_dataset.dropna(subset=["TotalCharges"], inplace=True)

# Convert the 'TotalCharges' column to numeric values
costumers_dataset.TotalCharges = pd.to_numeric(costumers_dataset.TotalCharges)


# Replace 'No internet service' and 'No phone service' with 'No'
costumers_dataset.replace('No internet service', 'No', inplace=True)
costumers_dataset.replace('No phone service', 'No', inplace=True)

# Replace 'Yes' and 'No' in yes/no columns with 1 and 0, respectively
yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]

for col in yes_no_cols:
    costumers_dataset[col].replace('Yes', 1, inplace = True)
    costumers_dataset[col].replace('No', 0, inplace = True)

# Replace 'Female' with 1 and 'Male' with 0 in the 'gender' column
costumers_dataset['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

# Convert 'Partner' to numeric values
costumers_dataset.Partner = pd.to_numeric(costumers_dataset.Partner)

# One-hot encode categorical variables
final_costumer_dataset = pd.get_dummies(data=costumers_dataset, columns=["InternetService", "Contract", "PaymentMethod"])


# Scale numerical columns
cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = MinMaxScaler()

final_costumer_dataset[cols_to_scale] = scaler.fit_transform(final_costumer_dataset[cols_to_scale])

print(final_costumer_dataset.sample(5))
# # Split data into training and testing sets
X = final_costumer_dataset.drop('Churn', axis='columns')
y = final_costumer_dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Create the CNN model
model = models.Sequential([
    layers.Dense(20, input_shape=(26,), activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create the CNN model
model = models.Sequential([
    layers.Dense(20,input_shape = (26,), activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
model.fit(X_train, y_train, epochs=100)



X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
model.evaluate(X_test, y_test)


yp = model.predict(X_test)


pred_y = []
for element in yp:
    if element > .5:
        pred_y.append(1)
    else:
        pred_y.append(0)
        



print(classification_report(y_test, pred_y))
