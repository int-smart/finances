#!/usr/bin/env python3
"""
Transformer model for time series prediction.
"""

import torch
from torch import nn
from torch.nn import functional as F

class SimpleTimeSeriesTransformer(nn.Module):
    """Simple Transformer model for time series prediction."""
    
    def __init__(self, in_channels=5, seq_len=64, pred_horizon=7, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_horizon = pred_horizon
        
        # Input projection: (B, C, T) -> (B, T, embed_dim)
        self.input_proj = nn.Linear(in_channels, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        # Transformer encoder (minimal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # Small feedforward
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: predict multiple future values
        self.output_head = nn.Linear(embed_dim, pred_horizon)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights according to best practices."""
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize output head
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        
        # Initialize transformer layers
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                # Initialize linear layers with xavier uniform
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize layer norm with standard values
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
    
    def print_init_summary(self):
        """Print initialization summary."""
        print("Model Initialization Summary:")
        print(f"  Input projection: Xavier uniform")
        print(f"  Positional embedding: Normal (std=0.02)")
        print(f"  Output head: Xavier uniform")
        print(f"  Transformer layers: Xavier uniform + LayerNorm standard")
        print(f"  Attention weights: Xavier uniform")
        
    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        
        # Reshape and project: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)  # (B, T, C)
        
        # Input projection
        x = self.input_proj(x)  # (B, T, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer encoding
        x = self.transformer(x)  # (B, T, embed_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Predict future values
        output = self.output_head(x)  # (B, pred_horizon)
        
        return output


def create_model(config, device):
    """Create and initialize a model with the given configuration."""
    model = SimpleTimeSeriesTransformer(
        in_channels=5,
        seq_len=config.get('seq_len', 64),
        pred_horizon=config.get('pred_horizon', 7),
        embed_dim=config.get('embed_dim', 32),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2)
    ).to(device)
    
    return model


def count_parameters(model):
    """Count and return model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }


def print_model_summary(model):
    """Print a summary of the model."""
    params = count_parameters(model)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Model size: {params['size_mb']:.2f} MB")
    
    # Print layer information
    print(f"\nModel Architecture:")
    for name, module in model.named_children():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module.weight.shape}")
        else:
            print(f"  {name}: {type(module).__name__}")


class UnifiedPredictionModel(nn.Module):
    """Unified model supporting regression, quantile, and probabilistic prediction."""
    
    def __init__(self, model_type, quantile_levels=[0.1, 0.5, 0.9], 
                 in_channels=5, seq_len=64, pred_horizon=7, 
                 embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.model_type = model_type
        self.quantile_levels = quantile_levels
        self.pred_horizon = pred_horizon
        self.embed_dim = embed_dim
        
        # Shared transformer backbone
        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Different prediction heads based on model type
        self.regression_head = nn.Linear(embed_dim, pred_horizon)
        self.quantile_head = nn.Linear(embed_dim, pred_horizon * len(quantile_levels))
        self.prob_mean_head = nn.Linear(embed_dim, pred_horizon)
        self.prob_var_head = nn.Linear(embed_dim, pred_horizon)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Prediction heads
        for head in [self.regression_head, self.quantile_head, 
                     self.prob_mean_head, self.prob_var_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        
        # Transformer layers
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def get_features(self, x):
        """Extract features from transformer backbone."""
        # Input projection: (B, C, T) -> (B, T, embed_dim)
        B, C, T = x.shape
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.input_proj(x)  # (B, T, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer encoding
        x = self.transformer(x)  # (B, T, embed_dim)
        
        # Global average pooling
        features = x.mean(dim=1)  # (B, embed_dim)
        
        return features
    
    def forward(self, x):
        """Forward pass with multiple prediction types."""
        # Get shared features
        features = self.get_features(x)
        
        outputs = {}
        
        # Regression prediction
        if 'regression' in self.model_type:
            outputs['regression'] = self.regression_head(features)
        
        # Quantile prediction
        if 'quantile' in self.model_type:
            quantile_preds = self.quantile_head(features)
            # Reshape: [batch, pred_horizon * num_quantiles] -> [batch, pred_horizon, num_quantiles]
            batch_size = quantile_preds.shape[0]
            outputs['quantiles'] = quantile_preds.view(
                batch_size, self.pred_horizon, len(self.quantile_levels)
            )
        
        # Probabilistic prediction
        if 'prob' in self.model_type or 'probabilistic' in self.model_type:
            outputs['prob_mean'] = self.prob_mean_head(features)
            outputs['prob_var'] = F.softplus(self.prob_var_head(features))  # Ensure positive
        
        return outputs
    
    def print_init_summary(self):
        """Print initialization summary."""
        print("Unified Model Initialization Summary:")
        print(f"  Model type: {self.model_type}")
        print(f"  Quantile levels: {self.quantile_levels}")
        print(f"  Input projection: Xavier uniform")
        print(f"  Positional embedding: Normal (std=0.02)")
        print(f"  Prediction heads: Xavier uniform")
        print(f"  Transformer layers: Xavier uniform + LayerNorm standard")


def create_unified_model(config, device):
    """Create unified model based on configuration."""
    model_type = config.get('model_type', 'regression')
    quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
    
    model_config = {
        'seq_len': config.get('window_size', 64),
        'pred_horizon': config.get('pred_horizon', 7),
        'embed_dim': config.get('embed_dim', 32),
        'num_heads': config.get('num_heads', 4),
        'num_layers': config.get('num_layers', 2)
    }
    
    model = UnifiedPredictionModel(
        model_type=model_type,
        quantile_levels=quantile_levels,
        **model_config
    ).to(device)
    
    return model
