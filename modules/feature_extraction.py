import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_features(model, feature_path, label_path, path_path, data_dir, class_names, supported_exts, image_size):
    """
    Load features, labels, and paths from disk if available.
    Otherwise extract from images and save them.

    Returns:
        X (ndarray): Feature vectors
        y (ndarray): Labels
        paths (ndarray): Image paths (for inspection/visualization)
    """
    if os.path.exists(feature_path) and os.path.exists(label_path) and os.path.exists(path_path):
        print(f"Loading features from:\n{feature_path}\n{label_path}\n{path_path}")
        X = np.load(feature_path)
        y = np.load(label_path, allow_pickle=True)
        paths = np.load(path_path, allow_pickle=True)
    else:
        print("Extracting features from images...")
        X, y, paths = extract_features(model, data_dir, class_names, supported_exts, image_size)
        np.save(feature_path, X)
        np.save(label_path, y)
        np.save(path_path, paths)
        print(f"Saved to:\n{feature_path}\n{label_path}\n{path_path}")
    return X, y, paths

def extract_features(model, data_dir, class_names, supported_exts, image_size):
    """
    Extract features using a CNN model.

    Returns:
        features: Numpy array of feature vectors
        labels: Numpy array of class labels
        paths: Numpy array of image file paths
    """
    features, labels, paths = [], [], []
    for cls in class_names:
        folder = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in supported_exts]

        for fname in tqdm(files, desc=f"Class: {cls}"):
            img_path = os.path.join(folder, fname)
            try:
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img)
                x = np.expand_dims(img_array, axis=0)
                x = preprocess_input(x)
                feature = model.predict(x, verbose=0)
                features.append(feature.flatten())
                labels.append(cls)
                paths.append(img_path)
            except Exception as e:
                print(f"Skipped '{img_path}': {e}")
    return np.array(features), np.array(labels).reshape(-1, 1), np.array(paths)