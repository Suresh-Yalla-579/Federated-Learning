"""
Client data partitioning utilities for federated learning.
Handles client selection and data loading strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os


class ClientDataPartitioner:
    """Manages client data partitions for federated learning."""
    
    def __init__(
        self,
        num_clients: int,
        data_dir: str = 'data/raw',
        seed: int = 42
    ):
        """
        Initialize the partitioner.
        
        Args:
            num_clients: Total number of clients
            data_dir: Directory containing client data files
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.data_dir = data_dir
        self.seed = seed
        np.random.seed(seed)
        
        # Verify all client files exist
        self._verify_client_files()
    
    def _verify_client_files(self) -> None:
        """Verify that all client data files exist."""
        for client_id in range(self.num_clients):
            filepath = os.path.join(self.data_dir, f'client_{client_id}.csv')
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Client data file not found: {filepath}")
    
    def get_client_ids(self) -> List[int]:
        """
        Get list of all client IDs.
        
        Returns:
            List of client IDs
        """
        return list(range(self.num_clients))
    
    def sample_clients(
        self,
        fraction: float = 0.1,
        min_clients: int = 5
    ) -> List[int]:
        """
        Randomly sample a subset of clients for a training round.
        
        Args:
            fraction: Fraction of clients to sample
            min_clients: Minimum number of clients to sample
            
        Returns:
            List of sampled client IDs
        """
        num_sampled = max(int(self.num_clients * fraction), min_clients)
        num_sampled = min(num_sampled, self.num_clients)
        
        sampled_ids = np.random.choice(
            self.num_clients,
            size=num_sampled,
            replace=False
        )
        
        return sorted(sampled_ids.tolist())
    
    def load_client_data(self, client_id: int) -> pd.DataFrame:
        """
        Load data for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client's DataFrame
        """
        filepath = os.path.join(self.data_dir, f'client_{client_id}.csv')
        return pd.read_csv(filepath)
    
    def get_client_data_sizes(self) -> Dict[int, int]:
        """
        Get the number of samples for each client.
        
        Returns:
            Dictionary mapping client_id to number of samples
        """
        sizes = {}
        for client_id in range(self.num_clients):
            df = self.load_client_data(client_id)
            sizes[client_id] = len(df)
        return sizes
    
    def get_client_class_distribution(
        self,
        client_id: int
    ) -> pd.Series:
        """
        Get class distribution for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Series with class counts
        """
        df = self.load_client_data(client_id)
        return df['workout_type'].value_counts()
    
    def analyze_data_heterogeneity(self) -> Dict:
        """
        Analyze the heterogeneity (non-IID-ness) of client data.
        
        Returns:
            Dictionary with heterogeneity metrics
        """
        all_distributions = []
        sizes = []
        
        for client_id in range(self.num_clients):
            df = self.load_client_data(client_id)
            sizes.append(len(df))
            
            # Get normalized class distribution
            dist = df['workout_type'].value_counts(normalize=True).sort_index()
            all_distributions.append(dist.values)
        
        all_distributions = np.array(all_distributions)
        
        # Calculate metrics
        mean_dist = np.mean(all_distributions, axis=0)
        std_dist = np.std(all_distributions, axis=0)
        
        # KL divergence from uniform distribution
        num_classes = all_distributions.shape[1]
        uniform_dist = np.ones(num_classes) / num_classes
        
        kl_divergences = []
        for dist in all_distributions:
            kl = np.sum(dist * np.log(dist / uniform_dist + 1e-10))
            kl_divergences.append(kl)
        
        return {
            'num_clients': self.num_clients,
            'total_samples': sum(sizes),
            'mean_samples_per_client': np.mean(sizes),
            'std_samples_per_client': np.std(sizes),
            'min_samples': min(sizes),
            'max_samples': max(sizes),
            'mean_class_distribution': mean_dist,
            'std_class_distribution': std_dist,
            'mean_kl_divergence': np.mean(kl_divergences),
            'std_kl_divergence': np.std(kl_divergences)
        }
    
    def create_centralized_dataset(
        self,
        sample_fraction: float = 1.0
    ) -> pd.DataFrame:
        """
        Create a centralized dataset from all clients (for baseline).
        
        Args:
            sample_fraction: Fraction of data to include
            
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for client_id in range(self.num_clients):
            df = self.load_client_data(client_id)
            
            if sample_fraction < 1.0:
                n_samples = int(len(df) * sample_fraction)
                df = df.sample(n=n_samples, random_state=self.seed + client_id)
            
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return combined


def print_partition_stats(partitioner: ClientDataPartitioner) -> None:
    """
    Print statistics about client data partitions.
    
    Args:
        partitioner: ClientDataPartitioner instance
    """
    print("Client Data Partition Statistics")
    print("=" * 50)
    
    stats = partitioner.analyze_data_heterogeneity()
    
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples per client: {stats['mean_samples_per_client']:.1f} ± {stats['std_samples_per_client']:.1f}")
    print(f"Sample range: [{stats['min_samples']}, {stats['max_samples']}]")
    print(f"\nData Heterogeneity (KL Divergence):")
    print(f"  Mean: {stats['mean_kl_divergence']:.4f}")
    print(f"  Std: {stats['std_kl_divergence']:.4f}")
    
    print(f"\nMean class distribution across clients:")
    for i, prob in enumerate(stats['mean_class_distribution']):
        print(f"  Class {i}: {prob:.3f}")
    
    # Sample a few clients
    print("\nSample of 3 clients:")
    sampled = partitioner.sample_clients(fraction=0.03, min_clients=3)
    for client_id in sampled:
        dist = partitioner.get_client_class_distribution(client_id)
        print(f"\nClient {client_id} (n={dist.sum()}):")
        for workout, count in dist.items():
            print(f"  {workout}: {count}")


def main():
    """Demonstrate partitioning functionality."""
    print("Testing client data partitioning...")
    
    partitioner = ClientDataPartitioner(
        num_clients=100,
        data_dir='data/raw'
    )
    
    # Print statistics
    print_partition_stats(partitioner)
    
    # Test client sampling
    print("\n" + "=" * 50)
    print("Testing client sampling...")
    sampled = partitioner.sample_clients(fraction=0.1)
    print(f"Sampled {len(sampled)} clients: {sampled[:10]}...")
    
    # Test centralized dataset creation
    print("\nCreating centralized dataset...")
    centralized = partitioner.create_centralized_dataset()
    print(f"Centralized dataset size: {len(centralized)}")
    print(f"Class distribution:")
    print(centralized['workout_type'].value_counts())
    
    print("\nPartitioning test complete!")


if __name__ == '__main__':
    main()