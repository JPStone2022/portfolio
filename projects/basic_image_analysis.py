import numpy as np
# Import matplotlib for optional visualization
import matplotlib.pyplot as plt

def analyze_grayscale_image(image_array):
    """
    Analyzes a grayscale image represented as a NumPy array.

    Args:
        image_array (np.ndarray): A 2D NumPy array representing the grayscale image
                                  (pixel values typically 0-255).

    Returns:
        dict: A dictionary containing analysis results.
              Returns None if the input is not a valid 2D NumPy array.
    """
    # Input validation
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        print("Error: Input must be a 2D NumPy array.")
        return None

    print(f"--- Analyzing Image ---")
    print(f"Shape: {image_array.shape}") # (height, width)
    print(f"Data Type: {image_array.dtype}")
    print(f"Total Pixels: {image_array.size}")

    # 1. Basic Statistics using NumPy functions
    min_pixel = np.min(image_array)
    max_pixel = np.max(image_array)
    mean_pixel = np.mean(image_array)
    std_dev = np.std(image_array)

    print(f"\n--- Basic Statistics ---")
    print(f"Minimum Pixel Value: {min_pixel}")
    print(f"Maximum Pixel Value: {max_pixel}")
    print(f"Mean Pixel Value (Average Brightness): {mean_pixel:.2f}")
    print(f"Standard Deviation of Pixel Values: {std_dev:.2f}")

    # 2. Simple Image Manipulations (using vectorized operations)
    # Assume max pixel value is 255 for inversion
    max_value = 255
    inverted_image = max_value - image_array
    print("\n--- Image Manipulation Examples ---")
    # Displaying a small part of the inverted image for comparison
    print("Original Top-Left 3x3:\n", image_array[:3, :3])
    print("Inverted Top-Left 3x3:\n", inverted_image[:3, :3])

    # Increase brightness (clipping values to stay within 0-255 range)
    brightness_increase = 50
    brighter_image = np.clip(image_array + brightness_increase, 0, max_value)
    print("\nBrighter Top-Left 3x3 (Increased by 50):\n", brighter_image[:3, :3])

    # 3. Histogram Calculation
    # Calculate the frequency of each pixel intensity value (0-255)
    # np.histogram needs the array and the bin edges.
    # For 0-255 values, we need 256 bins (0, 1, ..., 255)
    # Bin edges should be 257 (0 to 256) to include the rightmost edge.
    hist, bin_edges = np.histogram(image_array.flatten(), bins=256, range=(0, 256))
    print("\n--- Histogram Data ---")
    print(f"Number of bins: {len(hist)}")
    # Print frequency of first few pixel values
    for i in range(5):
        print(f"Pixels with value {i}: {hist[i]}")
    print("...")

    # --- Optional: Visualization with Matplotlib ---
    # Uncomment the following lines and the import at the top to display plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Image Analysis Results')
    # Original Image
    im0 = axs[0].imshow(image_array, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Original Image')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    # Histogram
    axs[1].plot(bin_edges[:-1], hist) # Plot histogram data
    axs[1].set_title('Pixel Intensity Histogram')
    axs[1].set_xlabel('Pixel Value')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)
    # Inverted Image
    im2 = axs[2].imshow(inverted_image, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title('Inverted Image')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()
    # --- End Optional Visualization ---

    results = {
        'shape': image_array.shape,
        'dtype': str(image_array.dtype),
        'size': image_array.size,
        'min_pixel': min_pixel,
        'max_pixel': max_pixel,
        'mean_pixel': mean_pixel,
        'std_dev': std_dev,
        'histogram_counts': hist.tolist(), # Convert hist to list for easier use
        'histogram_bins': bin_edges[:-1].tolist() # Bins correspond to left edge
    }
    return results

# --- Example Usage ---
if __name__ == "__main__":
    # Create a simple sample grayscale image (10x10) as a NumPy array
    # Values range from dark (0) to light (255)
    sample_image = np.array([
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        [40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        [70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        [80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
        [90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
        [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    ], dtype=np.uint8) # Use unsigned 8-bit integer, common for grayscale

    # You could also load a real image using Pillow and convert to NumPy array
    # from PIL import Image
    # try:
    #     img = Image.open('path/to/your/grayscale_image.png').convert('L') # Convert to grayscale
    #     sample_image = np.array(img)
    # except FileNotFoundError:
    #     print("Sample image file not found, using generated array.")

    analysis_results = analyze_grayscale_image(sample_image)

    if analysis_results:
        print("\n--- Analysis Summary (Returned Dictionary) ---")
        for key, value in analysis_results.items():
            if key.startswith('histogram'): # Don't print full histogram list
                 print(f"{key}: Data calculated (length={len(value)})")
            else:
                 print(f"{key}: {value}")

