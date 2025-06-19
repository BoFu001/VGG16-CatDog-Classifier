from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_callbacks(filepath, monitor="val_loss", patience=5):
    """
    Return common callbacks for training.

    Args:
        filepath (str): Path to save best model.
        monitor (str): Metric to monitor (e.g., 'val_loss' or 'val_accuracy').
        patience (int): Early stopping patience.

    Returns:
        list: List of callbacks.
    """
    return [
        ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True
        )
    ]
