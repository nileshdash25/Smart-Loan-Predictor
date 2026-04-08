import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==========================================
# PART 1: DATA PREPROCESSING
# ==========================================
print("Loading and preprocessing data...")
# FIXED: Replaced the dead URL with a reliable, working dataset URL
url = "https://raw.githubusercontent.com/sahutkarsh/loan-prediction-analytics-vidhya/master/train.csv"
df = pd.read_csv(url)

# Print columns to verify data loaded correctly
print("Data loaded successfully! Columns found:\n", df.columns.tolist())

# Handle missing values: Fill every column with its most frequent value (mode)
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode text data into numbers
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_columns:
    if col in df.columns:
        # Convert to string first to prevent mixed-type errors during encoding
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# Drop Loan_ID as it's useless for prediction
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

# Split Features (X) and Target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for ANN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl') # Save scaler for Django API

# ==========================================
# PART 2: BUILDING & TRAINING THE ANN
# ==========================================
print("\nBuilding the Artificial Neural Network...")

model = Sequential()

# Input Layer & 1st Hidden Layer (16 neurons, ReLU activation)
model.add(Dense(16, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2)) # Dropout prevents overfitting

# 2nd Hidden Layer (8 neurons)
model.add(Dense(8, activation='relu'))

# Output Layer (1 neuron, Sigmoid activation for Binary Output: 0 or 1)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("\nTraining started...")
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_data=(X_test_scaled, y_test), verbose=1)

# Evaluate model performance
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\n--- RESULTS ---")
print(f"Model Accuracy on Test Data: {accuracy*100:.2f}%")

# Save the trained model for Django integration
model.save('loan_model.keras')
print("\nModel saved successfully as 'loan_model.keras'")
print("Scaler saved successfully as 'scaler.pkl'")