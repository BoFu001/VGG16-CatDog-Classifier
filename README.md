# Cat vs Dog Classification Using Transfer Learning (VGG16)

[![View on GitHub](https://img.shields.io/badge/GitHub-Project-blue?logo=github)](https://github.com/BoFu001/vgg16-catdog-classifier)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![View on Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/bofu001/vgg16-catdog-classifier) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Bo%20Fu-blue)](https://www.linkedin.com/in/bofu/)

---

## Table of Contents

- [Project Objective](#-project-objective)
- [Dataset Overview](#-dataset-overview)
- [Dataset Splitting](#-dataset-splitting)
- [Modular Architecture](#-project-structure-modular-design)
- [Why Transfer Learning?](#-why-transfer-learning)
- [Comparative Perspective](#-comparative-perspective)
- [Summary](#-summary)
- [License](#-license)

---

### Project Objective

This notebook explores the application of transfer learning to solve a binary image classification problem: distinguishing between cats and dogs.

By utilizing a pretrained convolutional neural network, high accuracy is achieved using fewer training samples, fewer epochs, and significantly less computational effort than when training CNNs from scratch.

A VGG16 model pretrained on ImageNet is used as a fixed feature extractor. Two fully-connected classifiers—simple and complex—are trained on top of these extracted features to assess the performance trade-offs between model complexity and accuracy.

---

### Dataset Overview

The dataset originates from the [Kaggle Dogs vs. Cats competition](https://www.kaggle.com/competitions/dogs-vs-cats/data), comprising 25,000 labeled images (12,500 per class) in JPEG format.

To reduce computational overhead and emphasize the efficiency of transfer learning, a representative subset of 20,000 images was selected:

- 10,000 cat images: cat.0.jpg to cat.9999.jpg → stored in `data/cats/`
- 10,000 dog images: dog.0.jpg to dog.9999.jpg → stored in `data/dogs/`

---

### Dataset Splitting

After feature extraction with VGG16, the dataset was randomly partitioned into three subsets:

- Training set: 70%  
- Validation set: 15%  
- Test set: 15%

---

### Project Structure: Modular Design

This project adopts a modular architecture for clarity, scalability, and ease of reuse. All key functionalities are encapsulated within independent Python modules located in `./modules/`:

- `feature_extraction.py`: loads or extracts image features using VGG16
- `build_model.py`: defines both simple and complex fully-connected classifiers
- `callbacks.py`: configures training callbacks (e.g., EarlyStopping)
- `train_utils.py`: manages model training, saving, and history persistence
- `evaluate_model.py`: computes evaluation metrics and visualizes classification
- `history_plot.py`: plots training and validation performance curves
- `visualize_errors.py`: displays misclassified images for qualitative analysis

> *Modularization supports cleaner experimentation, easier debugging, and faster iteration—key attributes for any robust machine learning workflow.*

---


### Why Transfer Learning?

Training CNNs from scratch is computationally expensive and typically requires large-scale labeled datasets. Transfer learning addresses these challenges by leveraging representations learned on large datasets (e.g., ImageNet) and adapting them to new tasks.

Benefits include:

- Eliminates the need to relearn low-level features (edges, textures)
- Reduces risk of overfitting, especially on small datasets
- Accelerates training and stabilizes convergence
- Delivers strong results with minimal parameter tuning

See full project notebook: [`VGG16-CatDog-Classifier.ipynb`](./VGG16-CatDog-Classifier.ipynb)

---

### Comparative Perspective

In [a previous project](https://github.com/BoFu001/Catdog-cnn-from-scratch), custom CNNs were trained from scratch using different image resolutions and data augmentation strategies. The best-performing model (Model No. 13) achieved:

- F1-score: 94.9%
- Cost: High training time and data requirement

In this notebook, using VGG16 as a fixed feature extractor, a simple fully-connected model achieves:

- F1-score: 98.29%
- With: 10,000 images per class, fewer epochs, and minimal tuning

---

### Summary

This project demonstrates that even lightweight classifiers can achieve state-of-the-art results when powered by strong pretrained features. Transfer learning not only improves performance but also significantly reduces the cost of model development.

> *Takeaway: Transfer learning is a powerful and efficient paradigm for real-world image classification tasks with limited data and compute resources.*

---

### License

This repository is licensed under the [MIT License](LICENSE).

---

Completed: June 2025  
Author: Bo Fu  
[LinkedIn](https://www.linkedin.com/in/bofu/)