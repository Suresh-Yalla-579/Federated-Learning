"""
Generates synthetic fitness data for federated learning.
Creates non-IID client distributions to simulate real-world scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


class FitnessDataGenerator:
    """Generates synthetic fitness data with non-IID distributions."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Workout types
        self.workout_types = ['cardio', 'strength', 'hiit', 'yoga', 'pilates', 'cycling']
        
        # Feature ranges
        self.age_range = (18, 70)
        self.weight_range = (45, 120)  # kg
        self.rhr_range = (50, 90)  # resting heart rate
        self.steps_range = (2000, 15000)
        self.calories_range = (1500, 3500)
        
    def generate_client_data(
        self,
        num_clients: int = 100,
        samples_per_client: Tuple[int, int] = (50, 200),
        non_iid_degree: float = 0.7
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate data for multiple clients with non-IID distributions.
        
        Args:
            num_clients: Number of simulated clients
            samples_per_client: Min and max samples per client
            non_iid_degree: Degree of non-IID-ness (0=IID, 1=highly non-IID)
            
        Returns:
            Dictionary mapping client_id to their DataFrame
        """
        clients_data = {}
        
        for client_id in range(num_clients):
            # Random number of samples for this client
            n_samples = np.random.randint(samples_per_client[0], samples_per_client[1])
            
            # Create client-specific bias for non-IID data
            # Each client has preferences for certain workout types
            client_workout_probs = self._generate_client_workout_distribution(
                client_id, non_iid_degree
            )
            
            # Generate demographic bias (e.g., age groups prefer different workouts)
            age_bias = self._get_age_bias(client_id, num_clients)
            
            # Generate data
            data = self._generate_samples(
                n_samples, 
                client_workout_probs,
                age_bias
            )
            
            clients_data[client_id] = data
            
        return clients_data
    
    def _generate_client_workout_distribution(
        self, 
        client_id: int, 
        non_iid_degree: float
    ) -> np.ndarray:
        """
        Create client-specific workout preferences.
        
        Args:
            client_id: Client identifier
            non_iid_degree: How different clients should be from each other
            
        Returns:
            Probability distribution over workout types
        """
        # Base uniform distribution
        uniform_probs = np.ones(len(self.workout_types)) / len(self.workout_types)
        
        # Client-specific random distribution
        np.random.seed(client_id)
        random_probs = np.random.dirichlet(np.ones(len(self.workout_types)))
        
        # Blend uniform and random based on non_iid_degree
        probs = (1 - non_iid_degree) * uniform_probs + non_iid_degree * random_probs
        
        # Reset seed
        np.random.seed(None)
        
        return probs
    
    def _get_age_bias(self, client_id: int, num_clients: int) -> float:
        """
        Assign age bias to client (some clients are younger/older).
        
        Args:
            client_id: Client identifier
            num_clients: Total number of clients
            
        Returns:
            Age bias factor
        """
        # Divide clients into age groups
        age_group = client_id % 5
        age_biases = [0.0, 0.25, 0.5, 0.75, 1.0]  # Young to old
        return age_biases[age_group]
    
    def _generate_samples(
        self,
        n_samples: int,
        workout_probs: np.ndarray,
        age_bias: float
    ) -> pd.DataFrame:
        """
        Generate individual samples with feature correlations.
        
        Args:
            n_samples: Number of samples to generate
            workout_probs: Distribution over workout types
            age_bias: Age bias for this client
            
        Returns:
            DataFrame with generated samples
        """
        data = []
        
        for _ in range(n_samples):
            # Sample workout type
            workout_type = np.random.choice(self.workout_types, p=workout_probs)
            
            # Generate features with correlations
            age = self._sample_age(age_bias, workout_type)
            gender = np.random.choice(['M', 'F'])
            weight = self._sample_weight(age, gender)
            rhr = self._sample_rhr(age, workout_type)
            steps = self._sample_steps(workout_type)
            calories = self._sample_calories(weight, steps, workout_type)
            workout_history = self._sample_workout_history(workout_type)
            
            data.append({
                'age': age,
                'gender': gender,
                'weight': weight,
                'resting_heart_rate': rhr,
                'avg_steps_per_day': steps,
                'calories_burned': calories,
                'workout_history': workout_history,
                'workout_type': workout_type
            })
        
        return pd.DataFrame(data)
    
    def _sample_age(self, age_bias: float, workout_type: str) -> int:
        """Sample age with bias and workout correlation."""
        base_age = self.age_range[0] + age_bias * (self.age_range[1] - self.age_range[0])
        
        # Workout-specific age adjustments
        age_adjustments = {
            'hiit': -10,
            'cardio': -5,
            'strength': 0,
            'cycling': -5,
            'yoga': 5,
            'pilates': 5
        }
        
        age = int(np.random.normal(
            base_age + age_adjustments.get(workout_type, 0),
            8
        ))
        return np.clip(age, self.age_range[0], self.age_range[1])
    
    def _sample_weight(self, age: int, gender: str) -> float:
        """Sample weight based on age and gender."""
        base_weight = 70 if gender == 'M' else 60
        age_factor = (age - 35) * 0.2  # Weight tends to increase with age
        
        weight = np.random.normal(base_weight + age_factor, 12)
        return round(np.clip(weight, self.weight_range[0], self.weight_range[1]), 1)
    
    def _sample_rhr(self, age: int, workout_type: str) -> int:
        """Sample resting heart rate."""
        # Fitter people have lower RHR
        fitness_levels = {
            'hiit': -10,
            'cardio': -8,
            'cycling': -8,
            'strength': -5,
            'yoga': -3,
            'pilates': -3
        }
        
        base_rhr = 70 + (age - 40) * 0.2
        rhr = int(np.random.normal(
            base_rhr + fitness_levels.get(workout_type, 0),
            6
        ))
        return np.clip(rhr, self.rhr_range[0], self.rhr_range[1])
    
    def _sample_steps(self, workout_type: str) -> int:
        """Sample average steps per day."""
        step_levels = {
            'cardio': 12000,
            'hiit': 10000,
            'cycling': 8000,
            'strength': 7000,
            'yoga': 6000,
            'pilates': 6500
        }
        
        base_steps = step_levels.get(workout_type, 8000)
        steps = int(np.random.normal(base_steps, 2000))
        return np.clip(steps, self.steps_range[0], self.steps_range[1])
    
    def _sample_calories(
        self, 
        weight: float, 
        steps: int, 
        workout_type: str
    ) -> int:
        """Sample calories burned with correlations."""
        # Base metabolic rate
        bmr = 1500 + weight * 10
        
        # Activity calories
        activity_cal = steps * 0.04
        
        # Workout intensity
        workout_intensity = {
            'hiit': 500,
            'cardio': 400,
            'strength': 350,
            'cycling': 450,
            'yoga': 200,
            'pilates': 250
        }
        
        total = bmr + activity_cal + workout_intensity.get(workout_type, 300)
        calories = int(np.random.normal(total, 200))
        return np.clip(calories, self.calories_range[0], self.calories_range[1])
    
    def _sample_workout_history(self, workout_type: str) -> str:
        """Sample workout history (simplified categorical)."""
        # Users tend to have history in similar workout types
        similar_workouts = {
            'cardio': ['cardio', 'hiit', 'cycling'],
            'strength': ['strength', 'hiit'],
            'hiit': ['hiit', 'cardio', 'strength'],
            'yoga': ['yoga', 'pilates'],
            'pilates': ['pilates', 'yoga'],
            'cycling': ['cycling', 'cardio']
        }
        
        history_options = similar_workouts.get(workout_type, [workout_type])
        # 70% chance current workout is in history
        if np.random.random() < 0.7:
            return workout_type
        else:
            return np.random.choice(history_options)
    
    def save_client_data(
        self,
        clients_data: Dict[int, pd.DataFrame],
        output_dir: str = 'data/raw'
    ) -> None:
        """
        Save client data to disk.
        
        Args:
            clients_data: Dictionary of client DataFrames
            output_dir: Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for client_id, df in clients_data.items():
            filepath = os.path.join(output_dir, f'client_{client_id}.csv')
            df.to_csv(filepath, index=False)
        
        print(f"Saved data for {len(clients_data)} clients to {output_dir}")
    
    def generate_global_test_set(
        self,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generate a global test set with balanced distribution.
        
        Args:
            n_samples: Number of test samples
            
        Returns:
            Test DataFrame
        """
        # Use uniform distribution for test set
        uniform_probs = np.ones(len(self.workout_types)) / len(self.workout_types)
        
        test_data = self._generate_samples(
            n_samples,
            uniform_probs,
            age_bias=0.5  # Middle-aged
        )
        
        return test_data


def main():
    """Generate and save synthetic fitness data."""
    print("Generating synthetic fitness data...")
    
    generator = FitnessDataGenerator(seed=42)
    
    # Generate client data
    clients_data = generator.generate_client_data(
        num_clients=100,
        samples_per_client=(50, 200),
        non_iid_degree=0.7
    )
    
    # Save client data
    generator.save_client_data(clients_data, 'data/raw')
    
    # Generate and save test set
    test_data = generator.generate_global_test_set(n_samples=1000)
    os.makedirs('data/processed', exist_ok=True)
    test_data.to_csv('data/processed/test_set.csv', index=False)
    
    # Print statistics
    total_samples = sum(len(df) for df in clients_data.values())
    print(f"\nDataset Statistics:")
    print(f"  Total clients: {len(clients_data)}")
    print(f"  Total training samples: {total_samples}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Avg samples per client: {total_samples / len(clients_data):.1f}")
    
    # Show class distribution for first client
    first_client_data = clients_data[0]
    print(f"\nSample client (0) class distribution:")
    print(first_client_data['workout_type'].value_counts())
    
    print("\nData generation complete!")


if __name__ == '__main__':
    main()