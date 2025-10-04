#!/usr/bin/env python3
"""
Script to run wandb sweeps for hyperparameter optimization.
"""

import sys
import os
sys.path.append('/home/abhishek/Desktop/Projects/finances/nn')

import wandb
from train import train_model
from config import WANDB_SWEEP_CONFIG, MULTI_MODEL_SWEEP_CONFIG, DEFAULT_MODEL_CONFIG, SWEEP_CONFIGS

def run_single_training(config_name='medium'):
    """Run a single training with predefined config."""
    if config_name == 'small':
        config = SWEEP_CONFIGS['small']
    elif config_name == 'large':
        config = SWEEP_CONFIGS['large']
    else:
        config = DEFAULT_MODEL_CONFIG
    
    print(f"Running single training with config: {config_name}")
    print(f"Config: {config}")
    
    best_val = train_model(config)
    print(f"\n🎯 Training completed! Best validation loss: {best_val:.6f}")
    return best_val

def run_sweep(count=50, sweep_type='hyperparameter'):
    """Run a wandb sweep."""
    if sweep_type == 'multi_model':
        sweep_config = MULTI_MODEL_SWEEP_CONFIG
        project_name = "sp500-multi-model-comparison-oct4"
        print("Starting multi-model comparison sweep...")
        print("Model configurations:")
        for model_config in sweep_config['parameters']['model_config']['values']:
            model_type = model_config['model_type']
            quantile_levels = model_config['quantile_levels']
            loss_weights = model_config['loss_weights']
            print(f"  - {model_type}: quantiles={quantile_levels}, weights={loss_weights}")
    else:
        sweep_config = WANDB_SWEEP_CONFIG
        project_name = "sp500-transformer-sweep"
        print("Starting hyperparameter sweep...")
    
    print("Sweep parameters:")
    print(f"  Batch sizes: {sweep_config['parameters']['batch_size']['values']}")
    print(f"  Window sizes: {sweep_config['parameters']['window_size']['values']}")
    print(f"  Pred horizons: {sweep_config['parameters']['pred_horizon']['values']}")
    print(f"  Optimizers: {sweep_config['parameters']['optimizer']['values']}")
    print(f"  Schedulers: {sweep_config['parameters']['scheduler']['values']}")
    print(f"  Epochs: {sweep_config['parameters']['epochs']['value']}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"\nCreated sweep with ID: {sweep_id}")
    
    # Run sweep agent
    print(f"Running {count} experiments...")
    wandb.agent(sweep_id, train_model, count=count)
    print("\n🎯 Sweep completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'sweep':
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            run_sweep(count, 'hyperparameter')
        elif mode == 'multi_model':
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            run_sweep(count, 'multi_model')
        elif mode in ['small', 'medium', 'large']:
            run_single_training(mode)
        else:
            print("Usage:")
            print("  python run_training.py sweep [count]      # Run hyperparameter sweep")
            print("  python run_training.py multi_model [count] # Run multi-model comparison")
            print("  python run_training.py small             # Run single training")
            print("  python run_training.py medium            # Run single training")
            print("  python run_training.py large             # Run single training")
    else:
        # Default to sweep
        print("No arguments provided. Running multi-model comparison...")
        print("Usage:")
        print("  python run_training.py sweep [count]      # Run hyperparameter sweep")
        print("  python run_training.py multi_model [count] # Run multi-model comparison")
        print("  python run_training.py small             # Run single training")
        print("  python run_training.py medium            # Run single training")
        print("  python run_training.py large             # Run single training")
        run_sweep(10, 'multi_model')  # Default to multi-model comparison
