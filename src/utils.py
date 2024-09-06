import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_additional_images(image_paths):
    """
    Load additional images from the provided paths and labels.
    
    Parameters:
    - image_paths: List of tuples containing image path and label.
    
    Returns:
    - additional_images_data: Numpy array of image data.
    - additional_labels: One-hot encoded labels.
    """
    additional_images_data = []
    additional_labels = []

    for image_path, label in image_paths:
        img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))  # Load and resize to 28x28
        img_array = img_to_array(img) / 255.0  # Normalize the image
        additional_images_data.append(img_array)
        additional_labels.append(label)

    # Convert lists to numpy arrays
    additional_images_data = np.array(additional_images_data)
    additional_labels = to_categorical(additional_labels, num_classes=26)

    return additional_images_data, additional_labels
