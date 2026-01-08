"""
Secure aggregation utilities for federated learning.
Provides weighted averaging and aggregation configuration.
"""

import numpy as np
from typing import List, Tuple, Dict


def weighted_average(
    weights_list: List[List[np.ndarray]],
    num_samples_list: List[int]
) -> List[np.ndarray]:
    """
    Compute weighted average of model weights.
    
    Args:
        weights_list: List of weight arrays from each client
        num_samples_list: Number of samples for each client
        
    Returns:
        Averaged weights
    """
    # Calculate total samples
    total_samples = sum(num_samples_list)
    
    # Initialize averaged weights with zeros
    num_layers = len(weights_list[0])
    averaged_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    # Weighted sum
    for client_weights, num_samples in zip(weights_list, num_samples_list):
        weight_factor = num_samples / total_samples
        
        for i, layer_weights in enumerate(client_weights):
            averaged_weights[i] += layer_weights * weight_factor
    
    return averaged_weights


def federated_averaging(
    client_updates: Dict[int, Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Perform FedAvg aggregation.
    
    Args:
        client_updates: Dictionary mapping client_id to (weights, num_samples)
        
    Returns:
        Aggregated global weights
    """
    weights_list = [update[0] for update in client_updates.values()]
    num_samples_list = [update[1] for update in client_updates.values()]
    
    return weighted_average(weights_list, num_samples_list)


def compute_weight_difference(
    weights_before: List[np.ndarray],
    weights_after: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Compute difference between weight sets.
    
    Args:
        weights_before: Initial weights
        weights_after: Updated weights
        
    Returns:
        Weight differences
    """
    return [after - before for after, before in zip(weights_after, weights_before)]


def apply_weight_difference(
    base_weights: List[np.ndarray],
    weight_diff: List[np.ndarray],
    scale: float = 1.0
) -> List[np.ndarray]:
    """
    Apply scaled weight difference to base weights.
    
    Args:
        base_weights: Base weights
        weight_diff: Weight difference
        scale: Scaling factor
        
    Returns:
        Updated weights
    """
    return [base + scale * diff for base, diff in zip(base_weights, weight_diff)]


def clip_weights_norm(
    weights: List[np.ndarray],
    max_norm: float = 10.0
) -> List[np.ndarray]:
    """
    Clip weights to maximum norm (for stability).
    
    Args:
        weights: Model weights
        max_norm: Maximum allowed norm
        
    Returns:
        Clipped weights
    """
    # Compute global norm
    global_norm = np.sqrt(sum(np.sum(w ** 2) for w in weights))
    
    if global_norm > max_norm:
        clip_factor = max_norm / global_norm
        clipped = [w * clip_factor for w in weights]
        return clipped
    
    return weights


def add_gaussian_noise_to_weights(
    weights: List[np.ndarray],
    noise_scale: float = 0.01
) -> List[np.ndarray]:
    """
    Add Gaussian noise to weights (for privacy/security).
    
    Args:
        weights: Model weights
        noise_scale: Standard deviation of noise
        
    Returns:
        Noisy weights
    """
    noisy_weights = []
    for w in weights:
        noise = np.random.normal(0, noise_scale, w.shape)
        noisy_weights.append(w + noise)
    
    return noisy_weights


class SecureAggregator:
    """
    Aggregator with security features.
    Implements weighted averaging with optional security enhancements.
    """
    
    def __init__(
        self,
        clip_norm: float = None,
        add_noise: bool = False,
        noise_scale: float = 0.01
    ):
        """
        Initialize secure aggregator.
        
        Args:
            clip_norm: Maximum norm for weight clipping (None = no clipping)
            add_noise: Whether to add noise to aggregated weights
            noise_scale: Scale of noise to add
        """
        self.clip_norm = clip_norm
        self.add_noise = add_noise
        self.noise_scale = noise_scale
    
    def aggregate(
        self,
        client_updates: Dict[int, Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """
        Aggregate client updates with security features.
        
        Args:
            client_updates: Dictionary of client updates
            
        Returns:
            Aggregated weights
        """
        # Clip individual client weights if specified
        if self.clip_norm is not None:
            clipped_updates = {}
            for client_id, (weights, num_samples) in client_updates.items():
                clipped_weights = clip_weights_norm(weights, self.clip_norm)
                clipped_updates[client_id] = (clipped_weights, num_samples)
            client_updates = clipped_updates
        
        # Perform federated averaging
        aggregated = federated_averaging(client_updates)
        
        # Add noise if specified
        if self.add_noise:
            aggregated = add_gaussian_noise_to_weights(
                aggregated,
                self.noise_scale
            )
        
        return aggregated


def compute_aggregation_stats(
    weights_list: List[List[np.ndarray]]
) -> Dict[str, float]:
    """
    Compute statistics about weight distributions across clients.
    
    Args:
        weights_list: List of weight arrays from different clients
        
    Returns:
        Dictionary of statistics
    """
    # Flatten all weights
    all_weights = []
    for client_weights in weights_list:
        for layer_weights in client_weights:
            all_weights.extend(layer_weights.flatten())
    
    all_weights = np.array(all_weights)
    
    stats = {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'median': float(np.median(all_weights))
    }
    
    return stats


def verify_aggregation_correctness(
    client_weights: List[List[np.ndarray]],
    client_samples: List[int],
    aggregated: List[np.ndarray],
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that aggregation was computed correctly.
    
    Args:
        client_weights: List of client weight arrays
        client_samples: Number of samples per client
        aggregated: Aggregated weights
        tolerance: Numerical tolerance
        
    Returns:
        True if aggregation is correct
    """
    # Recompute aggregation
    expected = weighted_average(client_weights, client_samples)
    
    # Check each layer
    for agg_layer, exp_layer in zip(aggregated, expected):
        if not np.allclose(agg_layer, exp_layer, atol=tolerance):
            return False
    
    return True


def main():
    """Test secure aggregation utilities."""
    print("Testing Secure Aggregation...")
    
    # Create dummy client updates
    num_clients = 3
    layer_shapes = [(10, 5), (5, 3), (3,)]
    
    client_updates = {}
    for i in range(num_clients):
        weights = [np.random.randn(*shape) for shape in layer_shapes]
        num_samples = np.random.randint(50, 200)
        client_updates[i] = (weights, num_samples)
    
    print(f"Created {num_clients} client updates")
    for client_id, (weights, n) in client_updates.items():
        print(f"  Client {client_id}: {n} samples, {len(weights)} layers")
    
    # Test basic aggregation
    print("\nTesting weighted average...")
    weights_list = [u[0] for u in client_updates.values()]
    samples_list = [u[1] for u in client_updates.values()]
    aggregated = weighted_average(weights_list, samples_list)
    print(f"Aggregated {len(aggregated)} layers")
    
    # Verify correctness
    is_correct = verify_aggregation_correctness(
        weights_list,
        samples_list,
        aggregated
    )
    print(f"Aggregation correctness: {is_correct}")
    
    # Test secure aggregator
    print("\nTesting SecureAggregator...")
    aggregator = SecureAggregator(
        clip_norm=10.0,
        add_noise=False
    )
    secure_agg = aggregator.aggregate(client_updates)
    print(f"Secure aggregation complete")
    
    # Test with noise
    print("\nTesting with noise...")
    noisy_aggregator = SecureAggregator(
        clip_norm=10.0,
        add_noise=True,
        noise_scale=0.01
    )
    noisy_agg = noisy_aggregator.aggregate(client_updates)
    
    # Compare with and without noise
    diff = sum(np.sum((s - n) ** 2) for s, n in zip(secure_agg, noisy_agg))
    print(f"Difference with noise: {diff:.6f}")
    
    # Compute statistics
    print("\nWeight statistics:")
    stats = compute_aggregation_stats(weights_list)
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nSecure aggregation test complete!")


if __name__ == '__main__':
    main()