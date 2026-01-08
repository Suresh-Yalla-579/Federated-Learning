"""
Differential Privacy mechanisms for federated learning.
Implements gradient clipping and noise addition for privacy preservation.
"""

import numpy as np
from typing import List, Tuple
import tensorflow as tf


class DifferentialPrivacy:
    """
    Implements Differential Privacy for model updates.
    Uses gradient clipping and Gaussian noise mechanism.
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0,
        epsilon: float = None,
        delta: float = 1e-5
    ):
        """
        Initialize DP mechanism.
        
        Args:
            noise_multiplier: Multiplier for Gaussian noise (sigma)
            l2_norm_clip: Clipping threshold for gradients
            epsilon: Privacy budget (if None, computed from noise_multiplier)
            delta: Privacy parameter (probability of privacy breach)
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.delta = delta
        
        if epsilon is None:
            # Approximate epsilon using moments accountant
            # This is a simplified approximation
            self.epsilon = self._compute_epsilon()
        else:
            self.epsilon = epsilon
        
        print(f"DP initialized: ε={self.epsilon:.2f}, δ={self.delta}, clip={self.l2_norm_clip}, noise={self.noise_multiplier}")
    
    def _compute_epsilon(self, num_steps: int = 100) -> float:
        """
        Compute epsilon using simplified privacy accounting.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Epsilon value
        """
        # Simplified epsilon computation
        # In practice, use TensorFlow Privacy's privacy accountant
        if self.noise_multiplier == 0:
            return float('inf')
        
        # Approximate epsilon using strong composition
        # epsilon ≈ sqrt(2 * num_steps * log(1/delta)) / noise_multiplier
        epsilon = np.sqrt(2 * num_steps * np.log(1 / self.delta)) / self.noise_multiplier
        
        return epsilon
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Clipped gradients
        """
        # Compute global L2 norm of all gradients
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        
        # Clip if necessary
        if global_norm > self.l2_norm_clip:
            clip_factor = self.l2_norm_clip / global_norm
            clipped = [g * clip_factor for g in gradients]
        else:
            clipped = gradients
        
        return clipped
    
    def add_noise(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise to gradients.
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            Noisy gradients
        """
        noise_stddev = self.l2_norm_clip * self.noise_multiplier
        
        noisy_gradients = []
        for grad in gradients:
            noise = np.random.normal(0, noise_stddev, grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def privatize_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply DP to model weights (clip + add noise).
        
        Args:
            weights: List of weight arrays
            
        Returns:
            Privatized weights
        """
        # Treat weights as gradients for DP
        clipped = self.clip_gradients(weights)
        noisy = self.add_noise(clipped)
        return noisy
    
    def privatize_weight_updates(
        self,
        old_weights: List[np.ndarray],
        new_weights: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Apply DP to weight updates (delta weights).
        
        Args:
            old_weights: Weights before training
            new_weights: Weights after training
            
        Returns:
            Privatized new weights
        """
        # Compute weight updates
        updates = [new - old for new, old in zip(new_weights, old_weights)]
        
        # Apply DP to updates
        clipped_updates = self.clip_gradients(updates)
        noisy_updates = self.add_noise(clipped_updates)
        
        # Add noisy updates back to old weights
        privatized_weights = [
            old + noisy_update
            for old, noisy_update in zip(old_weights, noisy_updates)
        ]
        
        return privatized_weights
    
    def get_privacy_spent(self, num_steps: int) -> Tuple[float, float]:
        """
        Get privacy budget spent.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Tuple of (epsilon, delta)
        """
        epsilon = self._compute_epsilon(num_steps)
        return epsilon, self.delta


class DPOptimizer:
    """
    Wrapper for TensorFlow optimizer with DP.
    Uses TensorFlow Privacy's DPKerasSGDOptimizer.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0,
        num_microbatches: int = 1
    ):
        """
        Initialize DP optimizer.
        
        Args:
            learning_rate: Learning rate
            noise_multiplier: Noise multiplier for DP
            l2_norm_clip: Gradient clipping threshold
            num_microbatches: Number of microbatches for DP-SGD
        """
        try:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
            
            self.optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate
            )
            
            self.noise_multiplier = noise_multiplier
            self.l2_norm_clip = l2_norm_clip
            
            print(f"DP-SGD Optimizer initialized with clip={l2_norm_clip}, noise={noise_multiplier}")
            
        except ImportError:
            print("Warning: tensorflow_privacy not available. Using standard SGD.")
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            self.noise_multiplier = 0
            self.l2_norm_clip = 0


def compute_privacy_loss(
    noise_multiplier: float,
    batch_size: int,
    num_examples: int,
    epochs: int,
    delta: float = 1e-5
) -> float:

    # ✅ SAFETY CASTS
    batch_size = int(batch_size)
    num_examples = int(num_examples)
    epochs = int(epochs)
    delta = float(delta)
    noise_multiplier = float(noise_multiplier)

    """
    Compute privacy loss (epsilon) for given training parameters.
    
    Args:
        noise_multiplier: Noise multiplier
        batch_size: Batch size
        num_examples: Number of training examples
        epochs: Number of epochs
        delta: Delta parameter
        
    Returns:
        Epsilon value
    """
    try:
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
        
        steps_per_epoch = num_examples // batch_size
        total_steps = steps_per_epoch * epochs
        
        epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta,
        )

        
        return epsilon
        
    except ImportError:
        # Simplified approximation if TF Privacy not available
        steps_per_epoch = num_examples // batch_size
        total_steps = steps_per_epoch * epochs
        
        if noise_multiplier == 0:
            return float('inf')
        
        epsilon = np.sqrt(2 * total_steps * np.log(1 / delta)) / noise_multiplier
        return epsilon


def print_privacy_guarantees(
    noise_multiplier: float,
    l2_norm_clip: float,
    batch_size: int,
    num_examples: int,
    epochs: int,
    delta: float = 1e-5
) -> None:
    # ✅ CAST FIRST (CRITICAL FIX)
    batch_size = int(batch_size)
    num_examples = int(num_examples)
    epochs = int(epochs)
    delta = float(delta)
    noise_multiplier = float(noise_multiplier)

    epsilon = compute_privacy_loss(
        noise_multiplier,
        batch_size,
        num_examples,
        epochs,
        delta
    )

    print("\n" + "=" * 60)
    print("DIFFERENTIAL PRIVACY GUARANTEES")
    print("=" * 60)
    print(f"Privacy Budget (ε): {epsilon:.2f}")
    print(f"Delta (δ): {delta:.1e}")
    print(f"Noise Multiplier: {noise_multiplier}")
    print(f"L2 Norm Clip: {l2_norm_clip}")
    print(f"Training Details:")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Examples: {num_examples}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Total Steps: {(num_examples // batch_size) * epochs}")
    print("\nInterpretation:")
    if epsilon < 1:
        print("  ✓ Strong privacy (ε < 1)")
    elif epsilon < 10:
        print("  ⚠ Moderate privacy (1 ≤ ε < 10)")
    else:
        print("  ✗ Weak privacy (ε ≥ 10)")
    print("=" * 60 + "\n")



def main():
    """Test DP mechanisms."""
    print("Testing Differential Privacy mechanisms...")
    
    # Create DP instance
    dp = DifferentialPrivacy(
        noise_multiplier=1.0,
        l2_norm_clip=1.0,
        delta=1e-5
    )
    
    # Test with dummy gradients
    dummy_gradients = [
        np.random.randn(10, 5),
        np.random.randn(5, 3),
        np.random.randn(3,)
    ]
    
    print("\nOriginal gradient norms:")
    for i, grad in enumerate(dummy_gradients):
        norm = np.linalg.norm(grad)
        print(f"  Layer {i}: {norm:.4f}")
    
    # Clip gradients
    clipped = dp.clip_gradients(dummy_gradients)
    print("\nClipped gradient norms:")
    for i, grad in enumerate(clipped):
        norm = np.linalg.norm(grad)
        print(f"  Layer {i}: {norm:.4f}")
    
    # Add noise
    noisy = dp.add_noise(clipped)
    print("\nNoisy gradient norms:")
    for i, grad in enumerate(noisy):
        norm = np.linalg.norm(grad)
        print(f"  Layer {i}: {norm:.4f}")
    
    # Test privacy accounting
    print("\n" + "=" * 60)
    print("Testing Privacy Accounting")
    print("=" * 60)
    
    print_privacy_guarantees(
        noise_multiplier=0.3,
        l2_norm_clip=2.0,
        batch_size=32,
        num_examples=1000,
        epochs=10,
        delta=1e-5
    )
    
    print("DP test complete!")


if __name__ == '__main__':
    main()