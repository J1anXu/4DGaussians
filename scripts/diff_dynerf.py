import os
import cv2
import numpy as np

SUFFIX = "_mark_pixels"


def compute_and_save_diffs(cam_folder):
    images_folder = os.path.join(cam_folder, "images")
    if not os.path.exists(images_folder):
        print(f"Skipping {cam_folder}, 'images' folder not found.")
        return

    output_folder = cam_folder + SUFFIX
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(images_folder) if f.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

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
            mask = np.zeros_like(img)
            cv2.imwrite(os.path.join(output_folder, filename), mask)
        else:
            diff = cv2.absdiff(prev_image, img)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            mask = np.zeros_like(img)
            mask[gray_diff > 25] = (255, 255, 255)

            # result = cv2.addWeighted(img, 1, mask, 1, 0)
            cv2.imwrite(os.path.join(output_folder, filename), mask)

        prev_image = img


def main(base_path):
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist.")
        return

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith("cam"):
            if SUFFIX in folder_path:
                continue
            print(f"Processing {folder_path}...")
            compute_and_save_diffs(folder_path)


if __name__ == "__main__":
    dir = "data/dynerf/cut_roasted_beef"  # Change this to your target directory
    main(dir)
