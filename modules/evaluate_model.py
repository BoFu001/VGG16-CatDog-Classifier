import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_classification(y_true, y_pred, y_proba, model_name, class_names):
    """
    Compute and display precision, recall, F1-score, confusion matrix, and ROC curve.

    Args:
        y_true (array-like): True binary labels
        y_pred (array-like): Predicted binary labels (0 or 1)
        y_proba (array-like): Predicted probabilities
        class_names (list): Class names for display (default: ["cat", "dog"])
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)

    # Unpack the confusion matrix into its components
    tn, fp, fn, tp = cm.ravel()

    # Print descriptive metrics from confusion matrix
    print(f"True {class_names[0]} predicted as {class_names[0]}: {tn}")
    print(f"True {class_names[0]} predicted as {class_names[1]} (FP): {fp}")
    print(f"True {class_names[1]} predicted as {class_names[0]} (FN): {fn}")
    print(f"True {class_names[1]} predicted as {class_names[1]}: {tp}")

    # Display confusion matrix visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.grid(False)
    plt.show()

    # Plot ROC curve
    plot_roc_curve(y_true, y_proba, model_name)

def plot_roc_curve(y_true, y_proba, model_name):
    """
    Plot the ROC curve for binary classification.

    Args:
        y_true (array-like): True binary labels
        y_proba (array-like): Predicted probabilities
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()