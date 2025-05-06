import os
import numpy as np
import matplotlib.pyplot as plt

def get_npy_dimensions(directory):
    """
    Reads all .npy files in a given directory, extracts their dimensions, and stores them in a list.
    :param directory: Path to the folder containing .npy files.
    :return: List of tuple dimensions.
    """
    dimensions = []

    for file in os.listdir(directory):
        if file.endswith(".npy"):
            file_path = os.path.join(directory, file)
            try:
                arr = np.load(file_path)  # Load the .npy file
                dimensions.append(arr.shape)  # Store the shape (dimensions) of the array
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return dimensions

def plot_dimensions(dimensions):
    """
    Plots the distribution of image dimensions.
    :param dimensions: List of tuple dimensions.
    """
    if not dimensions:
        print("No valid .npy files found.")
        return

    # Extract width and height
    widths = [dim[1] for dim in dimensions if len(dim) >= 2]  # Assuming at least 2D arrays
    heights = [dim[0] for dim in dimensions if len(dim) >= 2]

    plt.figure(figsize=(10, 5))
    plt.scatter(widths, heights, alpha=0.6, color="purple", label="Image Sizes")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Distribution of .npy Image Dimensions")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = "/Users/maiankonabang/Desktop/CSCI467/outfit-aesthetic-classifier/outfit-aesthetic-classifier-dataset-web-scraping/"  # Update this path
    dimensions = get_npy_dimensions(folder_path)
    print(f"Found {len(dimensions)} .npy files with extracted dimensions.")
    plot_dimensions(dimensions)
