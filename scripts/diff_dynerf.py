import os
import cv2
import numpy as np


def process_camera_folder(cam_folder):
    images_folder = os.path.join(cam_folder, "images")
    if not os.path.exists(images_folder):
        print(f"Skipping {cam_folder}, 'images' folder not found.")
        return

    output_folder = cam_folder + "_mark_pixels"
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(f for f in os.listdir(images_folder) if f.endswith((".png", ".jpg", ".jpeg")))
    if not image_files:
        print(f"No images found in {images_folder}")
        return

    prev_image = None
    for i, filename in enumerate(image_files):
        img_path = os.path.join(images_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read {img_path}")
            continue

        if prev_image is None:
            # First image, save as is
            cv2.imwrite(os.path.join(output_folder, filename), img)
        else:
            # Compute pixel difference
            diff_mask = np.any(prev_image != img, axis=-1)
            marked_img = img.copy()
            marked_img[diff_mask] = [0, 255, 0]

            cv2.imwrite(os.path.join(output_folder, filename), marked_img)

        prev_image = img


def main(base_path):
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist.")
        return

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith("cam"):
            print(f"Processing {folder_path}...")
            process_camera_folder(folder_path)


if __name__ == "__main__":
    dir = "data/dynerf/cut_roasted_beef"  # Change this to your target directory
    main(dir)
