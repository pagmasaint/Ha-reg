import cv2
import numpy as np
import os

DATA_DIR = './data_test'

# Load the hand image (your original image)
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img_path = os.path.join(DATA_DIR, dir_, img_path)
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Create a black background of size 800x600
        background_height = 600
        background_width = 800
        black_background = np.zeros((background_height, background_width, 3), dtype=np.uint8)

        # Handle transparency if the image has an alpha channel
        if img.shape[2] == 4:  # Image has an alpha channel
            # Split the image into BGR and Alpha channel
            bgr, alpha = img[:, :, :3], img[:, :, 3]

            # Create a blank black background for the hand image
            hand_img_black_bg = np.zeros_like(bgr)

            # Apply the alpha mask to combine the hand with the black background
            hand_on_black = cv2.bitwise_or(hand_img_black_bg, bgr, mask=alpha)
        else:
            # If no alpha channel, just use the BGR image directly
            hand_on_black = img

        # Get the size of the hand image
        hand_height, hand_width = hand_on_black.shape[:2]

        # Calculate the position to center the hand image within the 800x600 background
        y_offset = (background_height - hand_height) // 2
        x_offset = (background_width - hand_width) // 2

        # Insert the hand image into the center of the black background
        black_background[y_offset:y_offset + hand_height, x_offset:x_offset + hand_width] = hand_on_black

        # Save the resulting image
        output_path = img_path  # Output path
        cv2.imwrite(output_path, black_background)

