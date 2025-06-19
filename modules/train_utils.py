import os
import pickle
from typing import Optional, Dict, Tuple
from tensorflow.keras.models import Model, load_model

def get_model_and_history(
    model_dir: str,
    model_name: str,
    history_name: str,
    train_data,
    val_data,
    input_shape: int,
    create_model_fn,
    epochs: int,
    callbacks=None
) -> Tuple[Model, Optional[Dict]]:
    """
    Load model and training history if they exist; otherwise train a new model,
    save it along with its history, and return both.

    Args:
        model_dir: Directory to save/load models and history.
        model_name: File name for the model.
        history_name: File name for the training history (.pkl).
        train_data: Tuple (X_train, y_train).
        val_data: Tuple (X_val, y_val).
        input_shape: Input shape of the model.
        create_model_fn: Function to create a new model.
        epochs: Number of training epochs.
        callbacks: List of Keras callbacks (optional).

    Returns:
        model: A trained or loaded model.
        history: Training history dictionary (or None if not found).
    """
    model_path = os.path.join(model_dir, model_name)
    history_path = os.path.join(model_dir, history_name)

    if os.path.exists(model_path):
        print(f"Found existing model at: {model_path}.")
        model = load_model(model_path)

        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        else:
            history = None
    else:
        print(f"Training new model: {model_name}")
        model = create_model_fn(input_shape)
        hist_obj = model.fit(
            x=train_data[0], y=train_data[1],
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        history = hist_obj.history

        model.save(model_path)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

    return model, history
