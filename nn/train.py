#!/usr/bin/env python3
"""
Training script for stock prediction model with dataset debugging.
"""

import os
import pickle
import random
import math
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Import dataset utilities
from dataset import load_stock_data, create_data_loaders, MultiStockWindowDataset
from model import SimpleTimeSeriesTransformer, create_model, count_parameters, print_model_summary, UnifiedPredictionModel, create_unified_model
from config import PRINT_FREQUENCY
from utils import TrainingLogger, validate_model, print_training_progress, create_optimizer, create_scheduler, MultiObjectiveLoss, evaluate_predictions, ActivationMonitor
from config import *
import torch.nn.functional as F

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Utility: ensure reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model is now imported from model.py

def validate_unified_model(model, val_loader, criterion, device, config):
    """Validate unified model on validation set."""
    model.eval()
    total_loss = 0.0
    all_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss, loss_components = criterion(outputs, y)
            
            # Evaluate predictions
            batch_metrics = evaluate_predictions(
                outputs, y, 
                config.get('model_type', 'regression'),
                config.get('quantile_levels', [0.1, 0.5, 0.9])
            )
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = 0.0
                all_metrics[key] += value
            
            total_loss += loss.item()
            num_batches += 1
    
    # Average metrics
    for key in all_metrics:
        all_metrics[key] /= num_batches
    
    avg_loss = total_loss / num_batches
    all_metrics['val_loss'] = avg_loss
    
    return avg_loss, all_metrics

# Training functions
def train_one_epoch(model, train_loader, val_loader, optimizer, criterion, scheduler, logger, epoch, total_epochs, config, activation_monitor=None):
    """Train one epoch with validation within epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_val_loss = 0.0  # Initialize validation loss
    import pdb; pdb.set_trace()
    # Progress tracking
    total_batches = len(train_loader)
    if SHOW_PROGRESS_BAR:
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    else:
        pbar = train_loader
    
    for batch_idx, (x, y, _) in enumerate(pbar):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Handle both unified and simple models
        if isinstance(model, UnifiedPredictionModel):
            outputs = model(x)
            loss, loss_components = criterion(outputs, y)
        else:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss_components = {'total_loss': loss.item()}
        
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update learning rate scheduler on every batch
        if scheduler is not None:
            scheduler.step()
        
        # Update progress bar
        if SHOW_PROGRESS_BAR:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Grad Norm': f'{grad_norm.item():.3f}'
            })
        
        # Validate and log at specified frequency
        # if (batch_idx + 1) % 5000 == 0:
        #     # Run validation - handle both model types
        #     if isinstance(model, UnifiedPredictionModel):
        #         epoch_val_loss, _ = validate_unified_model(model, val_loader, criterion, DEVICE, config)
        #     else:
        #         epoch_val_loss = validate_model(model, val_loader, criterion, DEVICE)
            
        #     # Log activation statistics
        #     if activation_monitor is not None:
        #         activation_monitor.log_activations()
        
        # Save checkpoint every 5000 batches
        if (batch_idx + 1) % 5000 == 0:
            # Calculate global batch number
            global_batch = epoch * total_batches + batch_idx + 1
            
            # Run validation for checkpoint
            if isinstance(model, UnifiedPredictionModel):
                checkpoint_val_loss, _ = validate_unified_model(model, val_loader, criterion, DEVICE, config)
            else:
                checkpoint_val_loss = validate_model(model, val_loader, criterion, DEVICE)
            
            # Save intermediate checkpoint
            is_best = logger.save_checkpoint(
                model, optimizer, epoch + 1, loss.item(), checkpoint_val_loss, 
                is_best=(checkpoint_val_loss < logger.best_val_loss),
                suffix=f"batch_{global_batch}"
            )
            
            print(f"\n📁 Checkpoint saved at batch {global_batch} (epoch {epoch+1}, batch {batch_idx+1})")
            print(f"   Val loss: {checkpoint_val_loss:.4f} {'(NEW BEST!)' if is_best else ''}")
        
        if (batch_idx + 1) % PRINT_FREQUENCY == 0:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log batch metrics including gradient norm
            logger.log_batch(batch_idx, total_batches, loss.item(), epoch_val_loss, epoch, current_lr)
            if logger.use_wandb:
                import wandb
                wandb.log({'grad_norm': grad_norm.item()})
            
            # Print progress with gradient norm
            print_training_progress(epoch + 1, batch_idx, total_batches, loss.item(), epoch_val_loss, current_lr)
            print(f"   Gradient Norm: {grad_norm.item():.4f}")
    
    return total_loss / num_batches

def train_model(config=None):
    """Train the transformer model with enhanced logging and checkpointing."""
    print("=== Training Transformer Model ===")
    
    # Handle config from wandb sweep or direct call
    if config is None:
        # Check if we're in a wandb sweep
        try:
            import wandb
            if wandb.run is None:
                # Initialize wandb run for sweep
                wandb.init()
            config = dict(wandb.config)
            print("Using config from wandb sweep")
            print(f"Config: {config}")
        except:
            config = {}
            print("No config provided, using defaults")
    else:
        print(f"Using provided config: {config}")
    
    # Load data
    print("Loading data...")
    data = load_stock_data(PICKLE_PATH, min_date=MIN_DATE, features=FEATURES)
    
    # Use sweep parameters if provided, otherwise use defaults
    train_split = config.get('train_split', TRAIN_SPLIT)
    batch_size = config.get('batch_size', BATCH_SIZE)
    window_size = config.get('window_size', WINDOW_SIZE)
    pred_horizon = config.get('pred_horizon', PRED_HORIZON)
    
    train_loader, val_loader, train_tickers, val_tickers = create_data_loaders(
        data, 
        train_split=train_split, 
        batch_size=batch_size, 
        window_size=window_size, 
        pred_horizon=pred_horizon,
        num_workers=NUM_WORKERS
    )
    
    print(f"Data Configuration:")
    print(f"  Train split: {train_split}")
    print(f"  Batch size: {batch_size}")
    print(f"  Window size: {window_size}")
    print(f"  Pred horizon: {pred_horizon}")
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model - handle nested model_config or flat config
    print("Creating model...")
    
    # Extract model configuration
    if 'model_config' in config:
        # Nested config from sweep
        model_config_dict = config['model_config']
        model_type = model_config_dict['model_type']
        quantile_levels = model_config_dict['quantile_levels']
        loss_weights = model_config_dict['loss_weights']
    else:
        # Flat config (backward compatibility)
        model_type = config.get('model_type', 'regression')
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        loss_weights = config.get('loss_weights', [1.0, 0.0, 0.0])
    
    # Update config with extracted values for downstream use
    config['model_type'] = model_type
    config['quantile_levels'] = quantile_levels
    config['loss_weights'] = loss_weights
    
    print(f"Model type: {model_type}")
    print(f"Quantile levels: {quantile_levels}")
    print(f"Loss weights: {loss_weights}")
    
    if model_type != 'regression' or 'quantile' in model_type or 'prob' in model_type:
        # Use unified model for multi-objective training
        model = create_unified_model(config, DEVICE)
        print(f"Created unified model: {model_type}")
        
        # Create multi-objective loss
        criterion = MultiObjectiveLoss(
            model_type=model_type,
            quantile_levels=quantile_levels,
            loss_weights=loss_weights
        )
    else:
        # Use simple model for regression only
        model_config_simple = {
            'seq_len': window_size,
            'pred_horizon': pred_horizon,
            'embed_dim': config.get('embed_dim', 32),
            'num_heads': config.get('num_heads', 4),
            'num_layers': config.get('num_layers', 2)
        }
        model = create_model(model_config_simple, DEVICE)
        criterion = nn.MSELoss()
    
    # model = torch.compile(model)
    # Show model summary
    print_model_summary(model)
    model.print_init_summary()
    
    # Setup training components
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    # Initialize logger
    logger = TrainingLogger(
        config=config,
        checkpoint_dir=CHECKPOINT_DIR,
        use_wandb=config.get('use_wandb', True)
    )
    
    # Initialize activation monitor
    activation_monitor = ActivationMonitor(use_wandb=config.get('use_wandb', True))
    activation_monitor.register_hooks(model)
    
    # Training loop
    epochs = config.get('epochs', 5)
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training with validation within epoch
        train_loss = train_one_epoch(
            model, train_loader, val_loader, optimizer, criterion, 
            scheduler, logger, epoch, epochs, config, activation_monitor
        )
        
        # Final validation at end of epoch
        if isinstance(model, UnifiedPredictionModel):
            val_loss, val_metrics = validate_unified_model(model, val_loader, criterion, DEVICE, config)
            # Log additional metrics
            if logger.use_wandb:
                import wandb
                wandb.log(val_metrics)
        else:
            val_loss = validate_model(model, val_loader, criterion, DEVICE)
        
        # Calculate improvement
        improvement = None
        if val_loss < logger.best_val_loss:
            improvement = ((logger.best_val_loss - val_loss) / logger.best_val_loss) * 100
        
        # Log epoch metrics
        logger.log_epoch(epoch + 1, train_loss, val_loss, improvement)
        
        # Save checkpoint
        is_best = logger.save_checkpoint(
            model, optimizer, epoch + 1, train_loss, val_loss, 
            is_best=(val_loss < logger.best_val_loss)
        )
        
        print("-" * 60)
    
    # Clean up activation monitor
    activation_monitor.remove_hooks()
    
    # Finish logging
    logger.finish()
    return logger.best_val_loss

if __name__ == "__main__":
    print("Starting training...")
    
    # Train the model with default config
    best_val = train_model(DEFAULT_MODEL_CONFIG)
    
    print(f"\nTraining completed! Best validation loss: {best_val:.4f}")
