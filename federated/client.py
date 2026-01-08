"""
Federated Learning client implementation using Flower.
Handles local training and model updates with differential privacy.
"""

import flwr as fl
import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np

from models.recommender import create_recommender_model, compile_model
from federated.dp import DifferentialPrivacy


class FitnessClient(fl.client.NumPyClient):
    """
    Flower client for federated workout recommendation.
    Each client represents a mobile device with local fitness data.
    """
    
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        input_dim: int,
        num_classes: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_dp: bool = True,
        dp_noise_multiplier: float = 1.0,
        dp_l2_clip: float = 1.0
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Client identifier
            X_train: Local training features
            y_train: Local training labels
            input_dim: Input feature dimension
            num_classes: Number of output classes
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            use_dp: Whether to use differential privacy
            dp_noise_multiplier: DP noise multiplier
            dp_l2_clip: DP gradient clipping threshold
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_dp = use_dp
        
        # Create model
        self.model = create_recommender_model(
            input_dim=input_dim,
            num_classes=num_classes
        )
        self.model = compile_model(
            self.model,
            learning_rate=learning_rate
        )
        
        # Initialize DP mechanism if enabled
        self.dp = None
        if use_dp:
            self.dp = DifferentialPrivacy(
                noise_multiplier=dp_noise_multiplier,
                l2_norm_clip=dp_l2_clip
            )
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model weight arrays
        """
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: List of model weight arrays
        """
        self.model.set_weights(parameters)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated parameters, number of samples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Store initial weights for DP
        if self.use_dp:
            initial_weights = self.model.get_weights()
        
        # Train locally
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        # Apply differential privacy to weight updates
        if self.use_dp and self.dp is not None:
            updated_weights = self.dp.privatize_weight_updates(
                initial_weights,
                updated_weights
            )
            self.model.set_weights(updated_weights)
        
        # Return updated weights and training metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history.get("accuracy", [0])[-1])
        }
        
        return updated_weights, len(self.X_train), metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local data.
        
        Args:
            parameters: Model parameters
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, number of samples, metrics)
        """
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(
            self.X_train,
            self.y_train,
            verbose=0
        )
        
        metrics = {"accuracy": float(accuracy)}
        
        return float(loss), len(self.X_train), metrics


def create_client_fn(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    num_classes: int,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_dp: bool = True,
    dp_noise_multiplier: float = 1.0,
    dp_l2_clip: float = 1.0
):
    """
    Create a client factory function for Flower.
    
    Args:
        client_data: Dictionary mapping client_id to (X, y) tuples
        input_dim: Input dimension
        num_classes: Number of classes
        local_epochs: Local training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_dp: Use differential privacy
        dp_noise_multiplier: DP noise multiplier
        dp_l2_clip: DP clipping threshold
        
    Returns:
        Client factory function
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Create a client instance."""
        client_id = int(cid)
        X_train, y_train = client_data[client_id]
        
        return FitnessClient(
            client_id=client_id,
            X_train=X_train,
            y_train=y_train,
            input_dim=input_dim,
            num_classes=num_classes,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_dp=use_dp,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_l2_clip=dp_l2_clip
        )
    
    return client_fn


def test_client():
    """Test client implementation."""
    print("Testing Federated Client...")
    
    # Create dummy data
    client_id = 0
    input_dim = 7
    num_classes = 6
    n_samples = 100
    
    X_train = np.random.randn(n_samples, input_dim)
    y_train = np.random.randint(0, num_classes, n_samples)
    
    # Create client
    client = FitnessClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        local_epochs=2,
        batch_size=32,
        use_dp=True,
        dp_noise_multiplier=0.5
    )
    
    # Get initial parameters
    initial_params = client.get_parameters({})
    print(f"Initial parameters: {len(initial_params)} layers")
    
    # Simulate training
    print("\nSimulating local training...")
    updated_params, num_samples, metrics = client.fit(initial_params, {})
    
    print(f"Training completed:")
    print(f"  Samples: {num_samples}")
    print(f"  Metrics: {metrics}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    print(f"Evaluation:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Metrics: {eval_metrics}")
    
    print("\nClient test complete!")


if __name__ == '__main__':
    test_client()