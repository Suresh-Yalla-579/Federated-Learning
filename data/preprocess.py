"""
Preprocessing utilities for fitness data.
Handles normalization, encoding, and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import pickle
import os


class FitnessDataPreprocessor:
    """Preprocesses fitness data for model training."""
    
    def __init__(self):
        """Initialize preprocessor with encoders and scalers."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.workout_history_encoder = LabelEncoder()
        
        # Feature columns
        self.numerical_features = [
            'age',
            'weight',
            'resting_heart_rate',
            'avg_steps_per_day',
            'calories_burned'
        ]
        
        self.categorical_features = [
            'gender',
            'workout_history'
        ]
        
        self.target_column = 'workout_type'
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FitnessDataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for chaining
        """
        # Fit scaler on numerical features
        self.scaler.fit(df[self.numerical_features])
        
        # Fit label encoders
        self.gender_encoder.fit(df['gender'])
        self.workout_history_encoder.fit(df['workout_history'])
        self.label_encoder.fit(df[self.target_column])
        
        self.is_fitted = True
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        include_labels: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to model-ready format.
        
        Args:
            df: DataFrame to transform
            include_labels: Whether to return labels
            
        Returns:
            Tuple of (features, labels) or just features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Normalize numerical features
        numerical = self.scaler.transform(df[self.numerical_features])
        
        # Encode categorical features
        gender_encoded = self.gender_encoder.transform(df['gender']).reshape(-1, 1)
        workout_hist_encoded = self.workout_history_encoder.transform(
            df['workout_history']
        ).reshape(-1, 1)
        
        # Concatenate all features
        features = np.concatenate([
            numerical,
            gender_encoded,
            workout_hist_encoded
        ], axis=1)
        
        if include_labels:
            labels = self.label_encoder.transform(df[self.target_column])
            return features, labels
        else:
            return features
    
    def fit_transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Tuple of (features, labels)
        """
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original workout types.
        
        Args:
            encoded_labels: Encoded integer labels
            
        Returns:
            Original workout type strings
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_num_classes(self) -> int:
        """Get number of workout classes."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return len(self.label_encoder.classes_)
    
    def get_num_features(self) -> int:
        """Get number of input features."""
        return len(self.numerical_features) + len(self.categorical_features)
    
    def get_feature_names(self) -> list:
        """Get ordered list of feature names."""
        return (
            self.numerical_features + 
            ['gender_encoded', 'workout_history_encoded']
        )
    
    def save(self, filepath: str = 'data/processed/preprocessor.pkl') -> None:
        """
        Save preprocessor to disk.
        
        Args:
            filepath: Path to save preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str = 'data/processed/preprocessor.pkl') -> 'FitnessDataPreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to load preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_client_data(
    client_id: int,
    preprocessor: FitnessDataPreprocessor,
    data_dir: str = 'data/raw'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data for a single client.
    
    Args:
        client_id: Client identifier
        preprocessor: Fitted preprocessor
        data_dir: Directory containing client data
        
    Returns:
        Tuple of (features, labels)
    """
    filepath = os.path.join(data_dir, f'client_{client_id}.csv')
    df = pd.read_csv(filepath)
    return preprocessor.transform(df)


def load_and_preprocess_test_set(
    preprocessor: FitnessDataPreprocessor,
    test_path: str = 'data/processed/test_set.csv'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the test set.
    
    Args:
        preprocessor: Fitted preprocessor
        test_path: Path to test set
        
    Returns:
        Tuple of (features, labels)
    """
    df = pd.read_csv(test_path)
    return preprocessor.transform(df)


def main():
    """Demonstrate preprocessing pipeline."""
    print("Testing preprocessing pipeline...")
    
    # Load sample data
    sample_df = pd.read_csv('data/raw/client_0.csv')
    print(f"Loaded sample data: {sample_df.shape}")
    
    # Create and fit preprocessor
    preprocessor = FitnessDataPreprocessor()
    X, y = preprocessor.fit_transform(sample_df)
    
    print(f"\nPreprocessed data shape:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Num classes: {preprocessor.get_num_classes()}")
    print(f"  Num features: {preprocessor.get_num_features()}")
    
    print(f"\nFeature names: {preprocessor.get_feature_names()}")
    print(f"Workout classes: {preprocessor.label_encoder.classes_}")
    
    # Save preprocessor
    preprocessor.save()
    
    # Test loading
    loaded = FitnessDataPreprocessor.load()
    X_loaded, y_loaded = loaded.transform(sample_df)
    
    # Verify consistency
    assert np.allclose(X, X_loaded), "Loaded preprocessor produces different results"
    assert np.array_equal(y, y_loaded), "Loaded labels don't match"
    
    print("\nPreprocessing pipeline test passed!")


if __name__ == '__main__':
    main()