"""
SHAP-based explainability for workout recommendations.
Provides feature importance and individual prediction explanations.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import tensorflow as tf
from tensorflow import keras


class WorkoutExplainer:
    """
    SHAP explainer for workout recommendation model.
    Explains which features influence workout recommendations.
    """
    
    def __init__(
        self,
        model: keras.Model,
        X_background: np.ndarray,
        feature_names: List[str],
        class_names: List[str]
    ):
        """
        Initialize explainer.
        
        Args:
            model: Trained Keras model
            X_background: Background dataset for SHAP
            feature_names: List of feature names
            class_names: List of class (workout type) names
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Create SHAP explainer
        # Use a subset of background data for efficiency
        n_background = min(100, len(X_background))
        self.X_background = X_background[:n_background]
        
        print(f"Initializing SHAP explainer with {n_background} background samples...")
        
        # Use DeepExplainer for neural networks
        self.explainer = shap.DeepExplainer(
            model,
            self.X_background
        )
        
        print("SHAP explainer initialized!")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        class_index: int = None
    ) -> np.ndarray:
        """
        Get SHAP values for predictions.
        
        Args:
            X: Input samples to explain
            class_index: Specific class to explain (None = all classes)
            
        Returns:
            SHAP values array
        """
        shap_values = self.explainer.shap_values(X)
        
        # shap_values is a list of arrays, one per output class
        if class_index is not None:
            return shap_values[class_index]
        
        return shap_values
    
    def plot_feature_importance(
        self,
        X: np.ndarray,
        class_index: int = 0,
        max_display: int = None,
        save_path: str = None
    ) -> None:
        """
        Plot feature importance for a specific class.
        
        Args:
            X: Input samples
            class_index: Class to explain
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        shap_values = self.explain_prediction(X, class_index)
        
        if max_display is None:
            max_display = len(self.feature_names)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f'Feature Importance for {self.class_names[class_index]}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        class_index: int = None,
        save_path: str = None
    ) -> None:
        """
        Plot waterfall explanation for a single prediction.
        
        Args:
            X: Input samples
            sample_index: Which sample to explain
            class_index: Which class to explain
            save_path: Path to save figure
        """
        shap_values = self.explain_prediction(X)
        
        if class_index is None:
            # Determine predicted class
            predictions = self.model.predict(X[sample_index:sample_index+1], verbose=0)
            class_index = np.argmax(predictions[0])
        
        # Create SHAP Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values[class_index][sample_index],
            base_values=self.explainer.expected_value[class_index],
            data=X[sample_index],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'Explanation for {self.class_names[class_index]} Recommendation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def explain_individual_prediction(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        top_k: int = 5
    ) -> Dict:
        """
        Get detailed explanation for a single prediction.
        
        Args:
            X: Input samples
            sample_index: Which sample to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        # Get prediction
        sample = X[sample_index:sample_index+1]
        prediction = self.model.predict(sample, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        
        # Get SHAP values for predicted class
        shap_values = self.explain_prediction(sample, predicted_class)
        shap_values = shap_values[0]  # Get values for the single sample
        
        # Get top contributing features
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': self.feature_names[idx],
                'value': float(X[sample_index, idx]),
                'shap_value': float(shap_values[idx]),
                'importance': float(abs_shap[idx])
            })
        
        explanation = {
            'sample_index': sample_index,
            'predicted_class': self.class_names[predicted_class],
            'prediction_confidence': float(prediction[predicted_class]),
            'all_class_probabilities': {
                self.class_names[i]: float(prediction[i])
                for i in range(len(prediction))
            },
            'top_contributing_features': top_features
        }
        
        return explanation
    
    def print_explanation(self, explanation: Dict) -> None:
        """
        Print explanation in readable format.
        
        Args:
            explanation: Explanation dictionary from explain_individual_prediction
        """
        print("\n" + "=" * 60)
        print("WORKOUT RECOMMENDATION EXPLANATION")
        print("=" * 60)
        
        print(f"\nPredicted Workout: {explanation['predicted_class']}")
        print(f"Confidence: {explanation['prediction_confidence']:.2%}")
        
        print("\nAll Class Probabilities:")
        for workout, prob in explanation['all_class_probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {workout:12s}: {prob:.2%} {bar}")
        
        print("\nTop Contributing Features:")
        print(f"{'Feature':<25} {'Value':>10} {'SHAP':>10} {'Impact':>10}")
        print("-" * 60)
        
        for feat in explanation['top_contributing_features']:
            impact = "+" if feat['shap_value'] > 0 else "-"
            print(
                f"{feat['feature']:<25} "
                f"{feat['value']:>10.2f} "
                f"{feat['shap_value']:>10.4f} "
                f"{impact:>10s}"
            )
        
        print("=" * 60 + "\n")


def create_explainer(
    model: keras.Model,
    X_background: np.ndarray,
    feature_names: List[str],
    class_names: List[str]
) -> WorkoutExplainer:
    """
    Factory function to create explainer.
    
    Args:
        model: Trained model
        X_background: Background data
        feature_names: Feature names
        class_names: Class names
        
    Returns:
        WorkoutExplainer instance
    """
    return WorkoutExplainer(
        model=model,
        X_background=X_background,
        feature_names=feature_names,
        class_names=class_names
    )


def main():
    """Test SHAP explainer."""
    print("Testing SHAP Explainer...")
    
    # Create dummy model and data
    from models.recommender import create_recommender_model, compile_model
    
    input_dim = 7
    num_classes = 6
    n_samples = 200
    
    # Create and train a simple model
    model = create_recommender_model(input_dim, num_classes)
    model = compile_model(model)
    
    # Generate dummy data
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randint(0, num_classes, n_samples)
    
    # Train briefly
    print("Training dummy model...")
    model.fit(X, y, epochs=5, verbose=0)
    
    # Feature and class names
    feature_names = [
        'age', 'weight', 'resting_heart_rate',
        'avg_steps_per_day', 'calories_burned',
        'gender_encoded', 'workout_history_encoded'
    ]
    class_names = ['cardio', 'strength', 'hiit', 'yoga', 'pilates', 'cycling']
    
    # Create explainer
    explainer = WorkoutExplainer(
        model=model,
        X_background=X[:100],
        feature_names=feature_names,
        class_names=class_names
    )
    
    # Test individual explanation
    print("\nGenerating explanation for sample...")
    test_sample = X[:10]
    explanation = explainer.explain_individual_prediction(test_sample, sample_index=0)
    explainer.print_explanation(explanation)
    
    print("SHAP explainer test complete!")


if __name__ == '__main__':
    main()