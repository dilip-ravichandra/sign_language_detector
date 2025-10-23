import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Data
DATA_PATH = "Alphabet_Data"
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

X, y = [], []

for label in labels:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(folder, file)))
            y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=16)

# Save model + label encoder
model.save("sign_alphabet_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
