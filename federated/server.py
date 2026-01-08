"""
Federated Learning server implementation using Flower.
Implements FedAvg strategy with configurable aggregation.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from models.recommender import create_recommender_model, compile_model





# --- ADD THIS UTILITY (paste near other imports) ---
from typing import Sequence
import numpy as np
from flwr.common.parameter import parameters_to_ndarrays

def safe_parameters_to_ndarrays(parameters) -> list:
    """
    Convert Flower Parameters-like object to list of ndarrays.
    Accepts either:
      - a Flower Parameters object (with .tensors), or
      - a plain list/sequence of numpy arrays.
    """
    # If the library already passed a list of ndarrays, just return it
    if isinstance(parameters, Sequence) and not hasattr(parameters, "tensors"):
        # Ensure each element is a numpy array
        return [np.asarray(t) for t in parameters]

    # Otherwise try the normal conversion path (Parameters proto)
    try:
        return parameters_to_ndarrays(parameters)
    except Exception:
        # fallback: if it has .tensors attribute but conversion still fails,
        # attempt to convert attribute directly (defensive)
        if hasattr(parameters, "tensors"):
            try:
                return [np.frombuffer(t, dtype=np.float32) for t in parameters.tensors]
            except Exception:
                pass
        raise
# --- END UTILITY ---


class FitnessServerStrategy(FedAvg):
    """
    Custom Flower strategy for fitness recommendation.
    Extends FedAvg with custom evaluation and logging.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
        **kwargs
    ):
        """
        Initialize server strategy.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            X_test: Global test features
            y_test: Global test labels
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
        """
        # Create initial model for server
        self.model = create_recommender_model(
            input_dim=input_dim,
            num_classes=num_classes
        )
        self.model = compile_model(self.model)
        
        # Test set for centralized evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize parent strategy
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=self.get_evaluate_fn(),
            **kwargs
        )
    
    def get_evaluate_fn(self):
        """
        Return evaluation function for centralized evaluation.
        
        Returns:
            Evaluation function
        """
        if self.X_test is None or self.y_test is None:
            return None
        
        def evaluate(
            server_round: int,
            parameters: Parameters,
            config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """
            Evaluate global model on test set.
            
            Args:
                server_round: Current round number
                parameters: Global model parameters
                config: Evaluation configuration
                
            Returns:
                Tuple of (loss, metrics dictionary)
            """
            # Update model with global parameters
            weights = safe_parameters_to_ndarrays(parameters)
            self.model.set_weights(weights)
            
            # Evaluate on test set
            loss, accuracy = self.model.evaluate(
                self.X_test,
                self.y_test,
                verbose=0
            )
            
            print(f"Round {server_round} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
            
            return loss, {"accuracy": accuracy}
        
        return evaluate
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients.
        
        Args:
            server_round: Current round number
            results: Training results from clients
            failures: Failed client updates
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        # Log client participation
        print(f"\nRound {server_round}: {len(results)} clients participated")
        
        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log training metrics
        if results:
            losses = [res.metrics.get("loss", 0) for _, res in results]
            accuracies = [res.metrics.get("accuracy", 0) for _, res in results]
            
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)
            
            print(f"  Avg Client Loss: {avg_loss:.4f}")
            print(f"  Avg Client Accuracy: {avg_acc:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from clients
            failures: Failed evaluations
            
        Returns:
            Aggregated loss and metrics
        """
        if not results:
            return None, {}
        
        # Call parent aggregate_evaluate
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        return loss, metrics


def create_server_strategy(
    input_dim: int,
    num_classes: int,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    fraction_fit: float = 0.1,
    min_fit_clients: int = 5,
    min_available_clients: int = 5
) -> FitnessServerStrategy:
    """
    Create server strategy for federated learning.
    
    Args:
        input_dim: Input dimension
        num_classes: Number of classes
        X_test: Test features
        y_test: Test labels
        fraction_fit: Fraction of clients to sample
        min_fit_clients: Minimum clients per round
        min_available_clients: Minimum available clients
        
    Returns:
        Server strategy
    """
    strategy = FitnessServerStrategy(
        input_dim=input_dim,
        num_classes=num_classes,
        X_test=X_test,
        y_test=y_test,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # No federated evaluation, use centralized
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=min_available_clients
    )
    
    return strategy


def get_initial_parameters(input_dim: int, num_classes: int) -> Parameters:
    """
    Get initial model parameters.
    
    Args:
        input_dim: Input dimension
        num_classes: Number of classes
        
    Returns:
        Initial parameters
    """
    model = create_recommender_model(
        input_dim=input_dim,
        num_classes=num_classes
    )
    model = compile_model(model)
    
    weights = model.get_weights()
    return ndarrays_to_parameters(weights)


def save_global_model(
    parameters: Parameters,
    input_dim: int,
    num_classes: int,
    save_path: str
) -> None:
    """
    Save global model to disk.
    
    Args:
        parameters: Model parameters
        input_dim: Input dimension
        num_classes: Number of classes
        save_path: Path to save model
    """
    # Create model
    model = create_recommender_model(
        input_dim=input_dim,
        num_classes=num_classes
    )
    model = compile_model(model)
    
    # Set weights
    weights = safe_parameters_to_ndarrays(parameters)

    model.set_weights(weights)
    
    # Save
    model.save(save_path)
    print(f"Global model saved to {save_path}")


def main():
    """Test server strategy."""
    print("Testing Federated Server Strategy...")
    
    # Parameters
    input_dim = 7
    num_classes = 6
    n_test = 100
    
    # Create dummy test data
    X_test = np.random.randn(n_test, input_dim)
    y_test = np.random.randint(0, num_classes, n_test)
    
    # Create strategy
    strategy = create_server_strategy(
        input_dim=input_dim,
        num_classes=num_classes,
        X_test=X_test,
        y_test=y_test,
        fraction_fit=0.1,
        min_fit_clients=2,
        min_available_clients=2
    )
    
    print("Server strategy created successfully!")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Test samples: {n_test}")
    
    # Get initial parameters
    initial_params = get_initial_parameters(input_dim, num_classes)
    print(f"\nInitial parameters obtained")
    
    # Test centralized evaluation
    if strategy.evaluate_fn:
        print("\nTesting centralized evaluation...")
        loss, metrics = strategy.evaluate_fn(0, initial_params, {})
        print(f"  Initial Loss: {loss:.4f}")
        print(f"  Initial Metrics: {metrics}")
    
    print("\nServer test complete!")


if __name__ == '__main__':
    main()