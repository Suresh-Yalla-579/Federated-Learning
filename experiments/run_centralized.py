"""
Run centralized training baseline.
Trains model on combined data from all clients.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.partitions import ClientDataPartitioner
from data.preprocess import FitnessDataPreprocessor, load_and_preprocess_test_set
from models.recommender import WorkoutRecommender
from models.metrics import (
    calculate_all_metrics,
    print_metrics,
    print_classification_report,
    plot_confusion_matrix
)


def load_config(config_path: str = 'experiments/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_centralized_training(config: dict) -> dict:
    """
    Run centralized training baseline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("CENTRALIZED TRAINING BASELINE")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    
    # Load and combine all client data
    print("\nLoading client data...")
    partitioner = ClientDataPartitioner(
        num_clients=config['data']['num_clients'],
        data_dir='data/raw'
    )
    
    # Create centralized dataset
    print("Creating centralized dataset...")
    train_df = partitioner.create_centralized_dataset()
    print(f"  Total training samples: {len(train_df)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = FitnessDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    
    # Save preprocessor
    preprocessor.save('data/processed/preprocessor.pkl')
    
    # Load test set
    print("Loading test set...")
    X_test, y_test = load_and_preprocess_test_set(preprocessor)
    print(f"  Test samples: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = WorkoutRecommender(
        input_dim=preprocessor.get_num_features(),
        num_classes=preprocessor.get_num_classes(),
        hidden_layers=config['model']['hidden_layers'],
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model']['l2_regularization'],
        learning_rate=config['training']['centralized']['learning_rate']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    print(f"  Epochs: {config['training']['centralized']['epochs']}")
    print(f"  Batch size: {config['training']['centralized']['batch_size']}")
    
    history = model.train(
        X_train,
        y_train,
        epochs=config['training']['centralized']['epochs'],
        batch_size=config['training']['centralized']['batch_size'],
        validation_split=config['training']['centralized']['validation_split'],
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(config['output']['plots_dir'], 'centralized_training_history.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = calculate_all_metrics(
        y_test,
        y_pred,
        y_pred_proba,
        k_values=config['evaluation']['recall_at_k']
    )
    
    print_metrics(metrics)
    
    # Classification report
    class_names = preprocessor.label_encoder.classes_.tolist()
    print_classification_report(y_test, y_pred, class_names)
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=class_names,
        save_path=os.path.join(
            config['output']['plots_dir'],
            'centralized_confusion_matrix.png'
        )
    )
    
    # Save model
    if config['output']['save_models']:
        model_path = os.path.join(
            config['output']['models_dir'],
            'centralized_model.h5'
        )
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
    
    # Prepare results
    results = {
        'metrics': metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_epochs': config['training']['centralized']['epochs'],
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1])
    }
    
    return results


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    
    # Run centralized training
    results = run_centralized_training(config)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CENTRALIZED TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"Recall@3: {results['metrics']['recall@3']:.4f}")
    
    print("\nResults saved to:")
    print(f"  - Plots: {config['output']['plots_dir']}")
    print(f"  - Models: {config['output']['models_dir']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()