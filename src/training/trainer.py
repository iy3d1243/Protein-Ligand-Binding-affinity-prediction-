"""
Trainer class for Drug-Target Affinity prediction model.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Trainer:
    """
    Trainer class with multi-GPU support and mixed precision training.
    
    Features:
    - DataParallel for multi-GPU training
    - Mixed precision (FP16) training
    - Gradient accumulation
    - Early stopping
    - Model checkpointing
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, model, train_loader, val_loader, config, pki_mean=0, pki_std=1):
        """
        Initialize trainer.
        
        Args:
            model: DTA prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            pki_mean (float): Mean pKi for denormalization
            pki_std (float): Std pKi for denormalization
        """
        self.config = config
        self.pki_mean = pki_mean
        self.pki_std = pki_std
        
        # Multi-GPU setup
        if config.USE_MULTI_GPU:
            print(f"✓ Using {torch.cuda.device_count()} GPUs: DataParallel")
            model = nn.DataParallel(model)
        
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            # Move data to device
            graph_batch = batch['graph_batch'].to(self.config.DEVICE)
            protein_input_ids = batch['protein_input_ids'].to(self.config.DEVICE)
            protein_attention_mask = batch['protein_attention_mask'].to(self.config.DEVICE)
            pki_true = batch['pki'].to(self.config.DEVICE)
            
            # Mixed precision forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pki_pred, _ = self.model(graph_batch, protein_input_ids, protein_attention_mask)
                    loss = self.criterion(pki_pred, pki_true)
                    loss = loss / self.config.GRADIENT_ACCUMULATION
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard forward pass
                pki_pred, _ = self.model(graph_batch, protein_input_ids, protein_attention_mask)
                loss = self.criterion(pki_pred, pki_true)
                loss = loss / self.config.GRADIENT_ACCUMULATION
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_NORM)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        """
        Evaluate model on a dataset.
        
        Args:
            loader: Data loader for evaluation
            
        Returns:
            tuple: (loss, r2, rmse, mae) - evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc='Evaluating'):
                # Move data to device
                graph_batch = batch['graph_batch'].to(self.config.DEVICE)
                protein_input_ids = batch['protein_input_ids'].to(self.config.DEVICE)
                protein_attention_mask = batch['protein_attention_mask'].to(self.config.DEVICE)
                pki_true = batch['pki'].to(self.config.DEVICE)
                
                # Forward pass
                pki_pred, _ = self.model(graph_batch, protein_input_ids, protein_attention_mask)
                
                # Handle NaN/Inf values
                if torch.isnan(pki_pred).any() or torch.isinf(pki_pred).any():
                    pki_pred = torch.nan_to_num(pki_pred, nan=0.0, posinf=10.0, neginf=0.0)
                
                # Compute loss
                loss = self.criterion(pki_pred, pki_true)
                
                total_loss += loss.item()
                all_preds.extend(pki_pred.cpu().numpy())
                all_trues.extend(pki_true.cpu().numpy())
        
        # Denormalize predictions and targets
        all_preds = np.array(all_preds) * self.pki_std + self.pki_mean
        all_trues = np.array(all_trues) * self.pki_std + self.pki_mean
        
        # Handle NaN/Inf values in final predictions
        all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=100.0, neginf=0.0)
        all_trues = np.nan_to_num(all_trues, nan=0.0, posinf=100.0, neginf=0.0)
        
        # Compute metrics
        avg_loss = total_loss / len(loader)
        r2 = r2_score(all_trues, all_preds)
        rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
        mae = mean_absolute_error(all_trues, all_preds)
        
        return avg_loss, r2, rmse, mae
    
    def train(self):
        """
        Train the model with early stopping and checkpointing.
        """
        print(f"\nTraining on {torch.cuda.device_count()} GPU(s)")
        print("=" * 80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_r2, val_rmse, val_mae = self.evaluate(self.val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | R²: {val_r2:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
            
            # Model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model (unwrap DataParallel if needed)
                model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'pki_mean': self.pki_mean,
                    'pki_std': self.pki_std
                }
                
                # Save to outputs directory
                torch.save(checkpoint, self.config.CHECKPOINTS_DIR / 'best_model_checkpoint.pt')
                torch.save(model_to_save.state_dict(), self.config.CHECKPOINTS_DIR / 'best_model_weights.pt')
                print("✓ Model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        # Load best model
        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_load.load_state_dict(torch.load(self.config.CHECKPOINTS_DIR / 'best_model_weights.pt'))
        print("\n" + "=" * 80)
        print("Training completed!")
