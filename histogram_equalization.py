import cv2
import os

inputs_path = "D:/Tumor Detection Model/dataset/tumor/"
output_path = "D:/Tumor Detection Model/dataset/tumor_equalized/"

os.makedirs(output_path, exist_ok=True)  # Create output folder if it doesn't exist

for filename in os.listdir(inputs_path):
    input_file = os.path.join(inputs_path, filename)

    # Load image in grayscale
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Apply histogram equalization
        equalized = cv2.equalizeHist(img)

        # Save the equalized image
        cv2.imwrite(os.path.join(output_path, filename), equalized)
        print(f"✅ Equalized: {filename}")