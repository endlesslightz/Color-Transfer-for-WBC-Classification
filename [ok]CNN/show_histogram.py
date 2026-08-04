import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to the image
image_path = os.path.join('data', 'may-grunwald.jpg')

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
else:
    # Read the image
    img_bgr = cv2.imread(image_path)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert BGR to Grayscale (Luma)
    img_luma = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    # histograms for R, G, B
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(10, 6))

    # Plot RGB histograms
    for i, color in enumerate(colors):
        # Calculate histogram for each channel (0=R, 1=G, 2=B in RGB image)
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f'{color.upper()} Channel')

    # Calculate Luma histogram
    hist_luma = cv2.calcHist([img_luma], [0], None, [256], [0, 256])
    plt.plot(hist_luma, color='black', linestyle='--', label='Luma (Grayscale)')

    plt.title('RGB and Luma Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])

    # Save the plot
    output_path = 'histogram_rgb_luma.png'
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

    # Show the plot
    plt.show()
