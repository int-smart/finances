#!/usr/bin/env python3
"""
Configuration file for the stock prediction model.
"""

# Data Configuration
PICKLE_PATH = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
WINDOW_SIZE = 64  # timesteps per sample
PRED_HORIZON = 7  # predict next 7 days
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
MIN_DATE = "2005-01-01"  # Only use data from 2005 onwards

# Model Configuration
DEFAULT_MODEL_CONFIG = {
    'lr': 1e-3,
    'wd': 1e-4,
    'embed_dim': 32,
    'num_heads': 4,
    'num_layers': 2,
    'epochs': 5,
    'optimizer': 'adam',
    'scheduler': None,
    'use_wandb': True
}

# Training Configuration
BATCH_SIZE = 128
TRAIN_SPLIT = 0.8
NUM_WORKERS = 0  # Fix for multiprocessing issues in Jupyter

# Progress Display Configuration
PRINT_FREQUENCY = 10  # Print progress every N batches
SHOW_PROGRESS_BAR = True  # Show progress bar during training
SAVE_CHECKPOINTS = True  # Save model checkpoints
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints

# Hyperparameter Search Configs
SWEEP_CONFIGS = {
    'small': {
        'lr': 1e-3,
        'wd': 1e-4,
        'embed_dim': 16,
        'num_heads': 2,
        'num_layers': 1,
        'epochs': 3,
        'optimizer': 'adam',
        'scheduler': None,
        'use_wandb': False
    },
    'medium': {
        'lr': 1e-3,
        'wd': 1e-4,
        'embed_dim': 32,
        'num_heads': 4,
        'num_layers': 2,
        'epochs': 5,
        'optimizer': 'adam',
        'scheduler': None,
        'use_wandb': True
    },
    'large': {
        'lr': 1e-3,
        'wd': 1e-4,
        'embed_dim': 64,
        'num_heads': 8,
        'num_layers': 3,
        'epochs': 10,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_wandb': True
    }
}

# Wandb Sweep Configuration
WANDB_SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Model hyperparameters
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'wd': {
            'distribution': 'log_uniform_values', 
            'min': 1e-6,
            'max': 1e-2
        },
        'embed_dim': {
            'values': [16, 32, 64, 128]
        },
        'num_heads': {
            'values': [2, 4, 8]
        },
        'num_layers': {
            'values': [1, 2, 3, 4]
        },
        'epochs': {
            'value': 1
        },
        
        # Data hyperparameters
        'batch_size': {
            'values': [32, 64, 128, 256, 512]
        },
        'train_split': {
            'distribution': 'uniform',
            'min': 0.8,
            'max': 0.9
        },
        'window_size': {
            'values': [32, 64, 128, 256]
        },
        'pred_horizon': {
            'values': [7]
        },
        
        # Training hyperparameters
        'optimizer': {
            'values': ['adamw']
        },
        'scheduler': {
            'values': ['cosine']
        },
        'scheduler_eta_min_ratio': {
            'values': [0.001, 0.01, 0.1]  # Minimum LR as ratio of initial LR
        }
    }
}

# Multi-Model Systematic Comparison Sweep
MULTI_MODEL_SWEEP_CONFIG = {
    'method': 'random',  # Use random search for better exploration
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Model configurations - each entry defines model_type, quantile_levels, and loss_weights together
        'model_config': {
            'values': [
                {
                    'model_type': 'regression',
                    'quantile_levels': [0.1, 0.5, 0.9],  # Not used but needed for consistency
                    'loss_weights': [1.0, 0.0, 0.0]      # Regression only
                },
                {
                    'model_type': 'quantile',
                    'quantile_levels': [0.1, 0.5, 0.9],
                    'loss_weights': [0.0, 1.0, 0.0]      # Quantile only
                },
                {
                    'model_type': 'quantile',
                    'quantile_levels': [0.05, 0.25, 0.5, 0.75, 0.95],
                    'loss_weights': [0.0, 1.0, 0.0]      # Quantile only (5 levels)
                },
                {
                    'model_type': 'probabilistic',
                    'quantile_levels': [0.1, 0.5, 0.9],  # Not used but needed for consistency
                    'loss_weights': [0.0, 0.0, 1.0]      # Probabilistic only
                },
                {
                    'model_type': 'regression_quantile',
                    'quantile_levels': [0.1, 0.5, 0.9],
                    'loss_weights': [0.5, 0.5, 0.0]      # Reg + Quantile
                },
                {
                    'model_type': 'regression_prob',
                    'quantile_levels': [0.1, 0.5, 0.9],  # Not used but needed for consistency
                    'loss_weights': [0.5, 0.0, 0.5]      # Reg + Prob
                },
                {
                    'model_type': 'quantile_prob',
                    'quantile_levels': [0.1, 0.5, 0.9],
                    'loss_weights': [0.0, 0.5, 0.5]      # Quantile + Prob
                }
                # {
                #     'model_type': 'all_combined',
                #     'quantile_levels': [0.1, 0.5, 0.9],
                #     'loss_weights': [0.33, 0.33, 0.34]   # All equal
                # }
            ]
        },
        
        # Core hyperparameters (minimal for quick comparison)
        'lr': {
            'values': [1e-3]  # Single value
        },
        'embed_dim': {
            'values': [32]  # Single value
        },
        'num_heads': {
            'values': [4]
        },
        'num_layers': {
            'values': [2]  # Single value
        },
        'epochs': {
            'value': 1
        },
        'batch_size': {
            'values': [64]  # Single value
        },
        'window_size': {
            'values': [16, 32, 64, 128]  # Single value
        },
        'pred_horizon': {
            'values': [7]  # Single value
        },
        'optimizer': {
            'values': ['adamw']
        },
        'scheduler': {
            'values': ['cosine']
        },
        'scheduler_eta_min_ratio': {
            'values': [0.01]
        }
    }
}
