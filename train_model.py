import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Load dataset
file_path = r"E:\CCET\6th sem\Data Mining and Analysis Lab\Flask\Flask workout\gym_members_exercise_tracking.csv"
df = pd.read_csv(file_path)

# Features and target
X = df.drop(columns=['Workout_Type'])
y = df['Workout_Type']

# Identify numerical and categorical features
numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                    'Resting_BPM', 'Session_Duration (hours)', 'Fat_Percentage',
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
categorical_features = ['Gender']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Full pipeline with model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Save pipeline
with open('workout_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as 'workout_model.pkl'")
