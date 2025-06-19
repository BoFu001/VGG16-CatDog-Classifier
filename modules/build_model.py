from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_fc_model_simple(input_shape, hidden_units=10, lr=1e-4):
    """
    Build a simple fully-connected neural network for baseline comparison.

    Args:
        input_shape (int): Dimension of input features

    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(input_shape,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_fc_model_complex(input_shape, dropout_rate=0.5, hidden_units=256, lr=1e-4):
    """
    Build a more complex fully-connected neural network classifier.

    Args:
        input_shape (int): Dimension of input features
        dropout_rate (float): Dropout rate for regularization
        hidden_units (int): Number of hidden layer units
        lr (float): Learning rate

    Returns:
        A compiled Keras model.
    """
    model = Sequential([
        Dense(hidden_units, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model