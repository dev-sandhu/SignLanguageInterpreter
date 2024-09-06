import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from utils import load_additional_images

# Path to CSV files
train_csv_path = r"C:\Users\8noor\Downloads\archive (7)\sign_mnist_train.csv"
test_csv_path = r"C:\Users\8noor\Downloads\archive (7)\sign_mnist_test.csv"

# Load CSV Data
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Separate labels and pixel data from CSV
train_labels_csv = train_df['label'].values
train_images_csv = train_df.drop('label', axis=1).values
test_labels_csv = test_df['label'].values
test_images_csv = test_df.drop('label', axis=1).values

# Normalize CSV pixel values to range [0, 1]
train_images_csv = train_images_csv / 255.0
test_images_csv = test_images_csv / 255.0

# Reshape CSV images to 28x28 and add channel dimension (grayscale image, so 1 channel)
train_images_csv = train_images_csv.reshape(-1, 28, 28, 1)
test_images_csv = test_images_csv.reshape(-1, 28, 28, 1)

# One-hot encode CSV labels (since there are 26 possible labels)
train_labels_csv = to_categorical(train_labels_csv, num_classes=26)
test_labels_csv = to_categorical(test_labels_csv, num_classes=26)

# Load additional images from the provided paths
additional_images_paths = [
    (r"C:\Users\8noor\Downloads\archive (7)\amer_sign2.png", 0),  # Label 0 corresponds to 'A'
    (r"C:\Users\8noor\Downloads\archive (7)\amer_sign3.png", 1),  # Label 1 corresponds to 'B'
    (r"C:\Users\8noor\Downloads\archive (7)\american_sign_language.PNG", 2)  # Label 2 corresponds to 'C'
]

additional_images_data, additional_labels = load_additional_images(additional_images_paths)

# Combine CSV and additional image data
train_images_combined = np.concatenate((train_images_csv, additional_images_data), axis=0)
train_labels_combined = np.concatenate((train_labels_csv, additional_labels), axis=0)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 output classes for each alphabet letter
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the combined dataset
model.fit(train_images_combined, train_labels_combined, epochs=10, batch_size=32, validation_data=(test_images_csv, test_labels_csv))

# Save the model for future use
model.save('sign_language_model_combined.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images_csv, test_labels_csv)
print(f"Test accuracy: {test_acc}")
