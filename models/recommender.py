"""
Neural network model for workout recommendation.
Implements an MLP-based classifier using Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import numpy as np


def create_recommender_model(
    input_dim: int,
    num_classes: int,
    hidden_layers: list = [128, 64, 32],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.01
) -> keras.Model:
    """
    Create an MLP-based workout recommender model.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of workout types to predict
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization strength
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    # First hidden layer
    x = layers.Dense(
        hidden_layers[0],
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name='hidden_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Additional hidden layers
    for i, units in enumerate(hidden_layers[1:], start=2):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f'hidden_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='workout_recommender')
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    metrics: list = ['accuracy']
) -> keras.Model:
    """
    Compile the model with optimizer and loss.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def get_model_weights(model: keras.Model) -> list:
    """
    Get model weights as list of numpy arrays.
    
    Args:
        model: Keras model
        
    Returns:
        List of weight arrays
    """
    return model.get_weights()


def set_model_weights(model: keras.Model, weights: list) -> None:
    """
    Set model weights from list of numpy arrays.
    
    Args:
        model: Keras model
        weights: List of weight arrays
    """
    model.set_weights(weights)


def get_model_parameters_size(model: keras.Model) -> int:
    """
    Get total number of parameters in the model.
    
    Args:
        model: Keras model
        
    Returns:
        Total parameter count
    """
    return model.count_params()


class WorkoutRecommender:
    """Wrapper class for the workout recommendation model."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list = [128, 64, 32],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001
    ):
        """
        Initialize the recommender.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of workout types
            hidden_layers: Hidden layer configuration
            dropout_rate: Dropout rate
            l2_reg: L2 regularization
            learning_rate: Learning rate
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        # Create and compile model
        self.model = create_recommender_model(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )
        
        self.model = compile_model(
            self.model,
            learning_rate=learning_rate
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(X, verbose=0)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 0
    ) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Tuple of (loss, accuracy)
        """
        results = self.model.evaluate(X_test, y_test, verbose=verbose)
        return results[0], results[1]
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_weights(self) -> list:
        """Get model weights."""
        return get_model_weights(self.model)
    
    def set_weights(self, weights: list) -> None:
        """Set model weights."""
        set_model_weights(self.model, weights)
    
    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()


def main():
    """Test the recommender model."""
    print("Testing workout recommender model...")
    
    # Create model
    input_dim = 7  # Number of features
    num_classes = 6  # Number of workout types
    
    recommender = WorkoutRecommender(
        input_dim=input_dim,
        num_classes=num_classes
    )
    
    # Print model summary
    print("\nModel Architecture:")
    recommender.summary()
    
    print(f"\nTotal parameters: {get_model_parameters_size(recommender.model):,}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    X_dummy = np.random.randn(100, input_dim)
    y_dummy = np.random.randint(0, num_classes, 100)
    
    # Train for one epoch
    history = recommender.train(
        X_dummy,
        y_dummy,
        epochs=1,
        batch_size=32,
        verbose=1
    )
    
    # Test prediction
    predictions = recommender.predict(X_dummy[:5])
    print(f"\nSample predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")
    
    # Test class prediction
    classes = recommender.predict_classes(X_dummy[:5])
    print(f"Predicted classes: {classes}")
    
    # Test evaluation
    loss, acc = recommender.evaluate(X_dummy, y_dummy)
    print(f"\nEvaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    print("\nModel test complete!")


if __name__ == '__main__':
    main()