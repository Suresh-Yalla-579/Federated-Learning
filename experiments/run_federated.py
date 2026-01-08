"""
Run federated learning experiment.
Trains model using federated learning with differential privacy.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import flwr as fl
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.partitions import ClientDataPartitioner
from data.preprocess import (
    FitnessDataPreprocessor,
    preprocess_client_data,
    load_and_preprocess_test_set
)
from models.recommender import create_recommender_model, compile_model
from models.metrics import (
    calculate_all_metrics,
    print_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_metric_comparison
)
from federated.client import create_client_fn
from federated.server import create_server_strategy, save_global_model
from federated.dp import print_privacy_guarantees
from explainability.shap_explainer import WorkoutExplainer


def load_config(config_path: str = 'experiments/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_client_data(
    config: dict,
    preprocessor: FitnessDataPreprocessor
) -> Dict[int, tuple]:
    """
    Prepare client data for federated learning.
    
    Args:
        config: Configuration dictionary
        preprocessor: Fitted preprocessor
        
    Returns:
        Dictionary mapping client_id to (X, y) tuples
    """
    print("Preparing client data...")
    client_data = {}
    
    for client_id in range(config['data']['num_clients']):
        X, y = preprocess_client_data(client_id, preprocessor, 'data/raw')
        client_data[client_id] = (X, y)
    
    print(f"  Prepared data for {len(client_data)} clients")
    
    return client_data


def run_federated_training(config: dict) -> dict:
    """
    Run federated learning experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("FEDERATED LEARNING TRAINING")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    
    # Load preprocessor
    print("\nLoading preprocessor...")
    try:
        preprocessor = FitnessDataPreprocessor.load('data/processed/preprocessor.pkl')
    except FileNotFoundError:
        print("Preprocessor not found. Creating new one...")
        from data.partitions import ClientDataPartitioner
        partitioner = ClientDataPartitioner(
            num_clients=config['data']['num_clients'],
            data_dir='data/raw'
        )
        train_df = partitioner.create_centralized_dataset(sample_fraction=0.1)
        preprocessor = FitnessDataPreprocessor()
        preprocessor.fit(train_df)
        preprocessor.save('data/processed/preprocessor.pkl')
    
    input_dim = preprocessor.get_num_features()
    num_classes = preprocessor.get_num_classes()
    
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    
    # Prepare client data
    client_data = prepare_client_data(config, preprocessor)
    
    # Load test set
    print("\nLoading test set...")
    X_test, y_test = load_and_preprocess_test_set(preprocessor)
    print(f"  Test samples: {len(X_test)}")
    
    # Print privacy guarantees
    if config['privacy']['use_dp']:
        print("\n" + "=" * 70)
        print("PRIVACY SETTINGS")
        print("=" * 70)
        avg_samples = np.mean([len(X) for X, _ in client_data.values()])
        print_privacy_guarantees(
            noise_multiplier=config['privacy']['noise_multiplier'],
            l2_norm_clip=config['privacy']['l2_norm_clip'],
            batch_size=config['training']['federated']['batch_size'],
            num_examples=int(avg_samples),
            epochs=config['training']['federated']['local_epochs'],
            delta=config['privacy']['delta']
        )
    
    # Create client factory
    print("\nCreating federated learning setup...")
    client_fn = create_client_fn(
        client_data=client_data,
        input_dim=input_dim,
        num_classes=num_classes,
        local_epochs=config['training']['federated']['local_epochs'],
        batch_size=config['training']['federated']['batch_size'],
        learning_rate=config['training']['federated']['learning_rate'],
        use_dp=config['privacy']['use_dp'],
        dp_noise_multiplier=config['privacy']['noise_multiplier'],
        dp_l2_clip=config['privacy']['l2_norm_clip']
    )
    
    # Create server strategy
    strategy = create_server_strategy(
        input_dim=input_dim,
        num_classes=num_classes,
        X_test=X_test,
        y_test=y_test,
        fraction_fit=config['training']['federated']['fraction_fit'],
        min_fit_clients=config['training']['federated']['min_fit_clients'],
        min_available_clients=config['training']['federated']['min_available_clients']
    )
    
    # Run federated learning
    print("\nStarting federated learning...")
    print(f"  Rounds: {config['training']['federated']['num_rounds']}")
    print(f"  Clients per round: ~{int(config['data']['num_clients'] * config['training']['federated']['fraction_fit'])}")
    print(f"  Local epochs: {config['training']['federated']['local_epochs']}")
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['data']['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['training']['federated']['num_rounds']),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    
    # Extract metrics history
    rounds = []
    accuracies = []
    losses = []
    
    for round_num, (loss, metrics) in enumerate(history.metrics_centralized['accuracy'], start=1):
        rounds.append(round_num)
        accuracies.append(metrics)
    
    for round_num, loss in enumerate(history.losses_centralized, start=1):
        losses.append(loss)
    
    # Plot training progress
    print("\nPlotting training progress...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(rounds, accuracies, marker='o', linewidth=2)
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Federated Learning - Test Accuracy')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(rounds, losses, marker='o', linewidth=2, color='orange')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Federated Learning - Test Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(config['output']['plots_dir'], 'federated_training_progress.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Load final model and evaluate
    print("\nFinal evaluation...")
    final_model = create_recommender_model(input_dim, num_classes)
    final_model = compile_model(final_model)
    final_model.set_weights(strategy.model.get_weights())
    
    # Make predictions
    y_pred_proba = final_model.predict(X_test, verbose=0)
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
            'federated_confusion_matrix.png'
        )
    )
    
    # Save final model
    if config['output']['save_models']:
        model_path = os.path.join(
            config['output']['models_dir'],
            'federated_model.h5'
        )
        final_model.save(model_path)
        print(f"\nFederated model saved to {model_path}")
    
    # SHAP explainability
    if config['explainability']['enabled']:
        print("\n" + "=" * 70)
        print("GENERATING SHAP EXPLANATIONS")
        print("=" * 70)
        
        try:
            explainer = WorkoutExplainer(
                model=final_model,
                X_background=X_test[:config['explainability']['background_samples']],
                feature_names=preprocessor.get_feature_names(),
                class_names=class_names
            )
            
            # Explain a few samples
            n_explain = min(config['explainability']['num_samples_to_explain'], len(X_test))
            
            print(f"\nExplaining {n_explain} sample predictions...")
            for i in range(n_explain):
                explanation = explainer.explain_individual_prediction(X_test, sample_index=i)
                explainer.print_explanation(explanation)
            
            # Generate feature importance plot for first class
            print("Generating feature importance plots...")
            for class_idx in range(min(3, num_classes)):
                explainer.plot_feature_importance(
                    X_test[:100],
                    class_index=class_idx,
                    save_path=os.path.join(
                        config['output']['plots_dir'],
                        f'shap_importance_class_{class_idx}_{class_names[class_idx]}.png'
                    )
                )
            
            print("SHAP explanations complete!")
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            print("Continuing without explainability...")
    
    # Prepare results
    results = {
        'metrics': metrics,
        'num_rounds': config['training']['federated']['num_rounds'],
        'final_accuracy': accuracies[-1],
        'final_loss': losses[-1],
        'history': {
            'rounds': rounds,
            'accuracies': accuracies,
            'losses': losses
        }
    }
    
    return results


def compare_with_centralized(
    federated_results: dict,
    centralized_results: dict,
    config: dict
) -> None:
    """
    Compare federated and centralized results.
    
    Args:
        federated_results: Federated training results
        centralized_results: Centralized training results
        config: Configuration
    """
    print("\n" + "=" * 70)
    print("COMPARING FEDERATED VS CENTRALIZED")
    print("=" * 70)
    
    # Plot comparison
    plot_metric_comparison(
        centralized_results['metrics'],
        federated_results['metrics'],
        label1="Centralized",
        label2="Federated (DP)",
        save_path=os.path.join(
            config['output']['plots_dir'],
            'federated_vs_centralized.png'
        )
    )
    
    # Print comparison
    print("\nAccuracy Comparison:")
    cent_acc = centralized_results['metrics']['accuracy']
    fed_acc = federated_results['metrics']['accuracy']
    diff = fed_acc - cent_acc
    
    print(f"  Centralized: {cent_acc:.4f}")
    print(f"  Federated:   {fed_acc:.4f}")
    print(f"  Difference:  {diff:+.4f} ({diff/cent_acc*100:+.2f}%)")
    
    print("\nF1 Score Comparison:")
    cent_f1 = centralized_results['metrics']['f1_score']
    fed_f1 = federated_results['metrics']['f1_score']
    diff = fed_f1 - cent_f1
    
    print(f"  Centralized: {cent_f1:.4f}")
    print(f"  Federated:   {fed_f1:.4f}")
    print(f"  Difference:  {diff:+.4f} ({diff/cent_f1*100:+.2f}%)")


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    
    # Run federated training
    fed_results = run_federated_training(config)
    
    # Try to load centralized results for comparison
    try:
        print("\nLoading centralized results for comparison...")
        # This assumes run_centralized.py was run first
        from experiments.run_centralized import run_centralized_training
        # For now, just print federated results
        print("\n" + "=" * 70)
        print("FEDERATED TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nFinal Test Accuracy: {fed_results['metrics']['accuracy']:.4f}")
        print(f"Final Test F1 Score: {fed_results['metrics']['f1_score']:.4f}")
        print(f"Recall@3: {fed_results['metrics']['recall@3']:.4f}")
        
    except Exception as e:
        print(f"\nCould not load centralized results: {e}")
        print("\n" + "=" * 70)
        print("FEDERATED TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nFinal Test Accuracy: {fed_results['metrics']['accuracy']:.4f}")
        print(f"Final Test F1 Score: {fed_results['metrics']['f1_score']:.4f}")
        print(f"Recall@3: {fed_results['metrics']['recall@3']:.4f}")
    
    print("\nResults saved to:")
    print(f"  - Plots: {config['output']['plots_dir']}")
    print(f"  - Models: {config['output']['models_dir']}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()