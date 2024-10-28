# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:37:53 2024

@author: RivVro0
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Function to load images and labels
def load_data(image_dir, labels_file):
    images = []
    labels = []
    
    # Load labels from CSV with semicolon delimiter
    label_data = np.loadtxt(labels_file, delimiter=';', skiprows=1, dtype=str)  # Read as string to process
    
    for row in label_data:
        image_filename = row[0]  # Assuming the first column contains the image filenames
        image_path = os.path.join(image_dir, image_filename)
        
        # Extract the grams of fresh clover from the corresponding column
        # Assuming the relevant grams value is in the 5th column (index 4)
        grams = float(row[4])  # Adjust the index according to your data structure

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))  # Resize to the input size of the model
            images.append(img)
            labels.append(grams)

    return np.array(images), np.array(labels)

# Load your dataset
image_dir = r'D:\02_Software Development\2_CloverGrass\Raw_data\biomass_data\train\images'
labels_file = r'D:\02_Software Development\2_CloverGrass\Raw_data\biomass_data\train\biomass_train_data.csv'  # This file should contain image names and corresponding weights
images, labels = load_data(image_dir, labels_file)

# Normalize images
images = images.astype('float32') / 255.0  # Normalize to [0, 1]

from sklearn.model_selection import train_test_split

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Single output for grams of clover
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Create the model
input_shape = (224, 224, 3)  # Adjust based on your image size
model = create_model(input_shape)

# Data augmentation (optional)
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=50)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Predict on the test set
y_pred = model.predict(X_test)

# Plotting predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Diagonal line
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Grams of Clover')
plt.ylabel('Predicted Grams of Clover')
plt.xlim([min(y_test), max(y_test)])
plt.ylim([min(y_test), max(y_test)])
plt.grid()
plt.show()


# Predict for a new image
def predict_new_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predicted_grams = model.predict(img)
    return predicted_grams[0][0]  # Return the predicted grams

# Example usage
new_image_path = 'path/to/your/new/image.jpg'
predicted_grams = predict_new_image(new_image_path)
print(f'Predicted grams of clover: {predicted_grams}')

# Plot training history
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
