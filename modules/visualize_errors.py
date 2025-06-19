import os
import matplotlib.pyplot as plt
import numpy as np

def show_misclassified_images(model_name, y_test, y_pred, image_paths, class_names, max_images=None):
    """
    Show misclassified images with filenames.

    Args:
        model_name (str): Name of the model (for plot title)
        y_test (array): True binary labels (0 or 1)
        y_pred (array): Predicted binary labels (0 or 1)
        image_paths (array): Corresponding image paths
        class_names (list): Class names, e.g. ["cats", "dogs"]
        max_images (int or None): Max number of images to show; If None, show all misclassified.
    """
    misclassified_idx = np.where(y_test.ravel() != y_pred.ravel())[0]
    total_misclassified = len(misclassified_idx)
    print(f"[{model_name}] Total misclassified: {total_misclassified}")

    if total_misclassified == 0:
        print("No misclassifications to show.")
        return

    # Determine how many misclassified images to display (all if max_images is None)
    if max_images is None:
        num_images = total_misclassified
    else:
        num_images = min(max_images, total_misclassified)

    
    plt.figure(figsize=(12, (num_images // 4 + 1) * 3))
    plt.suptitle(f"Misclassified Images – {model_name}", fontsize=14)

    for i, idx in enumerate(misclassified_idx[:num_images]):
        true_label = class_names[y_test[idx][0]]
        pred_label = class_names[y_pred[idx][0]]
        image_path = image_paths[idx]
        file_name = os.path.basename(image_path)

        try:
            img = plt.imread(image_path)
        except Exception as e:
            print(f"Could not read {image_path}: {e}")
            continue

        plt.subplot((num_images // 4 + 1), 4, i + 1)
        plt.imshow(img)
        plt.title(f"{file_name}\nTrue: {true_label} | Pred: {pred_label}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()