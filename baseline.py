import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from skimage.io import imread
from glob import glob

def load_images_from_folders(base_path):
    """
    Loads image byte data from subfolders and assigns labels.
    :param base_path: Root folder containing subdirectories for each class.
    :return: X (features as histograms of LBP), y (labels as class indices)
    """
    X, y = [], []
    class_labels = {}  # Map folder names to numerical labels
    label_idx = 0

    for folder in sorted(os.listdir(base_path)):  # Sort to maintain label consistency
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            if folder not in class_labels:
                class_labels[folder] = label_idx
                label_idx += 1
            label = class_labels[folder]

            for img_path in glob(os.path.join(folder_path, "*.jpg")):
                # Load the image in grayscale
                img = imread(img_path, as_gray=True)

                # Compute LBP (Local Binary Patterns)
                radius = 1  # Radius for LBP
                n_points = 8 * radius  # Number of points for LBP calculation
                lbp = local_binary_pattern(img, n_points, radius, method="uniform")

                # Compute histogram of LBP features
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

                # Normalize the histogram
                lbp_hist = lbp_hist.astype("float")
                lbp_hist /= (lbp_hist.sum() + 1e-6)

                X.append(lbp_hist)  # Add LBP histogram as a feature vector
                y.append(label)

    return np.array(X), np.array(y), class_labels


# Update the dataset path to point to your image folders
dataset_path = "/Users/maiankonabang/Desktop/CSCI467/outfit-aesthetic-classifier/outfit-aesthetic-classifier-dataset-web-scraping/test_images"  # Update this to your image directory path
X, y, class_labels = load_images_from_folders(dataset_path)

# Split dataset into training, development, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize One-vs-One and One-vs-All SVM classifiers
svm_ovo = SVC(kernel="rbf", decision_function_shape="ovo", C=1.0, gamma='scale')
svm_ova = SVC(kernel="rbf", decision_function_shape="ovr", C=1.0, gamma='scale')

# Train the models
svm_ovo.fit(X_train, y_train)
svm_ova.fit(X_train, y_train)

# Function to evaluate the model
def evaluate_model(model, X, y, set_name):
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"{set_name} Accuracy: {acc:.4f}")
    print(f"Number of support vectors: {len(model.support_)}")
    return acc

# Evaluate models on training, development, and test sets
print("\nOne-vs-One Results:")
evaluate_model(svm_ovo, X_train, y_train, "Training")
evaluate_model(svm_ovo, X_dev, y_dev, "Development")
evaluate_model(svm_ovo, X_test, y_test, "Test")

print("\nOne-vs-All Results:")
evaluate_model(svm_ova, X_train, y_train, "Training")
evaluate_model(svm_ova, X_dev, y_dev, "Development")
evaluate_model(svm_ova, X_test, y_test, "Test")

# Print class labels for reference
print("Class Labels:", class_labels)
