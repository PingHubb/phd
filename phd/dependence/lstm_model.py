import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess data
def load_data(file_name):
    data = pd.read_csv(file_name, sep=" ", header=None)
    labels = data.iloc[:, 0]  # First column is labels
    features = data.iloc[:, 1:]  # Remaining columns are features
    return features, labels

def main():
    features_1, labels_1 = load_data('/home/ping2/ros2_ws/src/phd/phd/resource/ai/gesture_1_data.txt')
    features_2, labels_2 = load_data('/home/ping2/ros2_ws/src/phd/phd/resource/ai/gesture_2_data.txt')

    # Combine data from both files
    features = pd.concat([features_1, features_2])
    labels = pd.concat([labels_1, labels_2])

    # Convert labels to categorical (one-hot encoding)
    labels = to_categorical(labels)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Reshape for LSTM [samples, time steps, features]
    features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 2: Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(labels.shape[1], activation='softmax'))  # Output layer

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 3: Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Step 4: Save the model
    model.save('/home/ping2/ros2_ws/src/phd/phd/resource/ai/gesture_recognition_model.keras')

if __name__ == '__main__':
    main()
