import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Define the path to your landmark data
DATA_PATH = "MP_Data"
labels = ["hello", "yes", "no", "iloveyou", "thankyou"]

X = []
y = []

# Load .npy landmark data
for label in labels:
    folder_path = os.path.join(DATA_PATH, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            X.append(data)
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels (e.g., "hello" -> 0, "yes" -> 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder for prediction use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest model (can be replaced with other models like SVM, MLP, etc.)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
with open('sign_language_model.h5', 'wb') as f:
    pickle.dump(model, f)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")
