#!/usr/bin/env python3
"""
Utility functions for training, logging, and checkpointing.
"""

import os
import torch
import wandb
from pathlib import Path
from datetime import datetime

import torch.nn.functional as F
import torch.nn as nn

class TrainingLogger:
    """Handles logging, checkpointing, and wandb integration."""
    
    def __init__(self, config, checkpoint_dir="checkpoints", use_wandb=True):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_wandb = use_wandb
        self.best_val_loss = float('inf')
        self.epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project="sp500-transformer",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def log_batch(self, batch_idx, total_batches, train_loss, val_loss=None, epoch=None, lr=None):
        """Log metrics for a single batch."""
        metrics = {
            'batch_idx': batch_idx,
            'total_batches': total_batches,
            'train_loss': train_loss,
            'epoch': epoch or self.epoch
        }
        
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        
        if lr is not None:
            metrics['learning_rate'] = lr
        
        if self.use_wandb:
            wandb.log(metrics)
    
    def log_epoch(self, epoch, train_loss, val_loss, improvement=None):
        """Log metrics for a complete epoch."""
        self.epoch = epoch
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        if improvement is not None:
            metrics['improvement'] = improvement
        
        if self.use_wandb:
            wandb.log(metrics)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        if improvement is not None:
            print(f"  Improvement: {improvement:.2f}%")
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, is_best=False, suffix=None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint with optional suffix
        if suffix:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        wandb.save(checkpoint_path)
        
        # Save best model if improved
        if is_best:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            if self.use_wandb:
                wandb.save(str(best_path))
            
            print(f"  🎉 New best model saved: {best_path}")
            return True
        
        return False
    
    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def finish(self):
        """Finish logging session."""
        if self.use_wandb:
            wandb.finish()
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")


def validate_model(model, val_loader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def print_training_progress(epoch, batch_idx, total_batches, train_loss, val_loss=None, lr=None):
    """Print training progress."""
    # progress = (batch_idx + 1) / total_batches * 100
    # print(f"Epoch {epoch}, Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) - "
    #       f"Train Loss: {train_loss:.6f}", end="")
    
    if val_loss is not None:
        print(f", Val Loss: {val_loss:.6f}", end="")
    
    if lr is not None:
        print(f", LR: {lr:.2e}", end="")
    
    print()


def create_optimizer(model, config):
    """Create optimizer with given configuration."""
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('wd', 1e-4)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config, train_loader=None):
    """Create learning rate scheduler."""
    scheduler_type = config.get('scheduler', None)
    
    if scheduler_type is None:
        return None
    elif scheduler_type == 'step':
        step_size = config.get('scheduler_step_size', 10)
        gamma = config.get('scheduler_gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler_type == 'cosine':
        # Use CosineAnnealingLR with T_max = total batches per epoch
        # This will go from max LR (batch 0) to min LR (last batch) in one cycle
        if train_loader is not None:
            T_max = len(train_loader)  # Total batches in epoch
        else:
            T_max = config.get('scheduler_T_max', 1000)  # Fallback
        
        # Make eta_min relative to initial LR to ensure consistent behavior
        initial_lr = config.get('lr', 1e-3)
        eta_min_ratio = config.get('scheduler_eta_min_ratio', 0.01)  # Default: 1% of initial LR
        eta_min = initial_lr * eta_min_ratio
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss for unified prediction model."""
    
    def __init__(self, model_type, quantile_levels=[0.1, 0.5, 0.9], loss_weights=[1.0, 0.0, 0.0]):
        super().__init__()
        self.model_type = model_type
        self.quantile_levels = torch.tensor(quantile_levels)
        self.loss_weights = loss_weights
        
        self.mse_loss = nn.MSELoss()
    
    def quantile_loss(self, pred, target, quantile):
        """Quantile (pinball) loss."""
        # pred: [batch, pred_horizon, num_quantiles]
        # target: [batch, pred_horizon]
        # quantile: scalar
        target_expanded = target.unsqueeze(-1)  # [batch, pred_horizon, 1]
        error = target_expanded - pred  # [batch, pred_horizon, num_quantiles]
        
        loss = torch.max((quantile - 1) * error, quantile * error)
        return loss.mean()
    
    def probabilistic_loss(self, mean, var, target):
        """Negative log-likelihood for Gaussian distribution."""
        # mean, var, target: [batch, pred_horizon]
        
        # Ensure numerical stability
        var = torch.clamp(var, min=1e-6)
        
        # NLL = 0.5 * (log(var) + (target - mean)^2 / var)
        nll = 0.5 * (torch.log(var) + (target - mean)**2 / var)
        return nll.mean()
    
    def forward(self, outputs, targets):
        """Compute multi-objective loss."""
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        loss_components = {}
        
        # Regression loss
        if 'regression' in outputs and self.loss_weights[0] > 0:
            reg_loss = self.mse_loss(outputs['regression'], targets)
            total_loss = total_loss + self.loss_weights[0] * reg_loss
            loss_components['regression_loss'] = reg_loss.item()
        
        # Quantile loss
        if 'quantiles' in outputs and self.loss_weights[1] > 0:
            quantile_losses = []
            quantiles_pred = outputs['quantiles']  # [batch, pred_horizon, num_quantiles]
            
            for i, q in enumerate(self.quantile_levels):
                q_pred = quantiles_pred[:, :, i]  # [batch, pred_horizon]
                q_loss = self.quantile_loss(q_pred.unsqueeze(-1), targets, q.item())
                quantile_losses.append(q_loss)
            
            avg_quantile_loss = torch.stack(quantile_losses).mean()
            total_loss = total_loss + self.loss_weights[1] * avg_quantile_loss
            loss_components['quantile_loss'] = avg_quantile_loss.item()
        # Probabilistic loss
        if 'prob_mean' in outputs and 'prob_var' in outputs and self.loss_weights[2] > 0:
            prob_loss = self.probabilistic_loss(
                outputs['prob_mean'], outputs['prob_var'], targets
            )
            total_loss = total_loss + self.loss_weights[2] * prob_loss
            loss_components['probabilistic_loss'] = prob_loss.item()
        
        # Add total loss to components
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components


def evaluate_predictions(outputs, targets, model_type, quantile_levels):
    """Comprehensive evaluation of all prediction types."""
    metrics = {}
    
    # Regression metrics
    if 'regression' in outputs:
        reg_pred = outputs['regression']
        metrics['mse'] = F.mse_loss(reg_pred, targets).item()
        metrics['mae'] = F.l1_loss(reg_pred, targets).item()
        metrics['rmse'] = torch.sqrt(F.mse_loss(reg_pred, targets)).item()
        
        # R-squared
        ss_res = torch.sum((targets - reg_pred) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = r2.item()
    
    # Quantile metrics
    if 'quantiles' in outputs:
        quantiles = outputs['quantiles']  # [batch, pred_horizon, num_quantiles]
        
        # Quantile coverage (how often true value falls within quantiles)
        for i, q in enumerate(quantile_levels):
            q_pred = quantiles[:, :, i]  # [batch, pred_horizon]
            
            if q < 0.5:
                # Lower quantile - should be below true value
                coverage = (targets >= q_pred).float().mean()
                metrics[f'quantile_{q:.2f}_coverage'] = coverage.item()
            elif q > 0.5:
                # Upper quantile - should be above true value  
                coverage = (targets <= q_pred).float().mean()
                metrics[f'quantile_{q:.2f}_coverage'] = coverage.item()
            else:
                # Median - evaluate MAE
                mae = F.l1_loss(q_pred, targets)
                metrics[f'quantile_{q:.2f}_mae'] = mae.item()
        
        # Quantile loss for each level
        total_q_loss = 0
        for i, q in enumerate(quantile_levels):
            q_pred = quantiles[:, :, i]
            error = targets - q_pred
            q_loss = torch.max((q - 1) * error, q * error).mean()
            metrics[f'quantile_{q:.2f}_loss'] = q_loss.item()
            total_q_loss += q_loss.item()
        
        metrics['avg_quantile_loss'] = total_q_loss / len(quantile_levels)
        
        # Interval coverage (between quantiles)
        if len(quantile_levels) >= 3:
            # Assume symmetric quantiles around median
            lower_idx = 0
            upper_idx = -1
            if quantile_levels[0] < 0.5 and quantile_levels[-1] > 0.5:
                lower_bound = quantiles[:, :, lower_idx]
                upper_bound = quantiles[:, :, upper_idx]
                
                interval_coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float().mean()
                expected_coverage = quantile_levels[-1] - quantile_levels[0]
                metrics[f'interval_coverage'] = interval_coverage.item()
                metrics[f'expected_coverage'] = expected_coverage
    
    # Probabilistic metrics
    if 'prob_mean' in outputs and 'prob_var' in outputs:
        mean = outputs['prob_mean']
        var = outputs['prob_var']
        std = torch.sqrt(var)
        
        # NLL
        nll = 0.5 * (torch.log(var) + (targets - mean)**2 / var)
        metrics['nll'] = nll.mean().item()
        
        # Calibration (prediction intervals)
        for confidence in [0.68, 0.95]:  # 1σ, 2σ
            z_score = 1.0 if confidence == 0.68 else 1.96
            lower = mean - z_score * std
            upper = mean + z_score * std
            
            coverage = ((targets >= lower) & (targets <= upper)).float().mean()
            metrics[f'prob_{confidence:.0%}_coverage'] = coverage.item()
        
        # Mean absolute error for probabilistic mean
        metrics['prob_mae'] = F.l1_loss(mean, targets).item()
        
        # Average prediction uncertainty
        metrics['avg_uncertainty'] = std.mean().item()
    
    return metrics


class ActivationMonitor:
    """Simple activation monitoring for training diagnostics."""
    
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks on Linear, LayerNorm, and MultiheadAttention layers."""
        def hook_fn(name):
            def hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    # Linear, LayerNorm outputs
                    self.activations[name] = output.detach().float()
                elif isinstance(output, tuple) and len(output) >= 1:
                    # MultiheadAttention returns (output, attention_weights)
                    self.activations[name] = output[0].detach().float()
                    # Store attention weights separately if available
                    if len(output) > 1 and output[1] is not None:
                        self.activations[f"{name}_attn_weights"] = output[1].detach().float()
            return hook
        
        # Register hooks for Linear, LayerNorm, and MultiheadAttention layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
        
        print(f"  Registered activation hooks on {len(self.hooks)} layers")
    
    def log_activations(self):
        """Log activation statistics to wandb."""
        if not self.use_wandb or not self.activations:
            return
        
        metrics = {}
        
        for layer_name, act in self.activations.items():
            if '_attn_weights' in layer_name:
                # Attention-specific metrics
                metrics[f'attention/{layer_name}/mean'] = act.mean().item()
                metrics[f'attention/{layer_name}/std'] = act.std().item()
                
                # Max attention weight (measures attention sparsity/focus)
                max_attn = act.max(dim=-1)[0].mean().item()
                metrics[f'attention/{layer_name}/max_weight'] = max_attn
                
                # Attention entropy (higher = more uniform, lower = more focused)
                attn_entropy = -(act * torch.log(act + 1e-10)).sum(dim=-1).mean().item()
                metrics[f'attention/{layer_name}/entropy'] = attn_entropy
                
                # Histogram
                metrics[f'attention/{layer_name}/histogram'] = wandb.Histogram(act.cpu().numpy().flatten())
            else:
                # Regular activation statistics
                metrics[f'activations/{layer_name}/mean'] = act.mean().item()
                metrics[f'activations/{layer_name}/std'] = act.std().item()
                
                # Dead neurons ratio (activations near zero)
                dead_ratio = (act.abs() < 1e-6).float().mean().item()
                metrics[f'activations/{layer_name}/dead_ratio'] = dead_ratio
                
                # Histogram for distribution
                metrics[f'activations/{layer_name}/histogram'] = wandb.Histogram(act.cpu().numpy().flatten())
        
        # Don't specify step - let wandb use its automatic step counter
        wandb.log(metrics)
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
