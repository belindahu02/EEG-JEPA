 # =============================================
# FIXED trainers_cosine.py with Class-Balanced Loss
# =============================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbones import SpectrogramResNet, LightweightSpectrogramResNet
from data_loader_musicid import create_session_based_dataloaders
import os
import json
from datetime import datetime
import time
import logging
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import cohen_kappa_score, confusion_matrix

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# =============================================
# Class-Balanced Loss (CRITICAL FIX)
# =============================================
class ClassBalancedLoss(nn.Module):
    """Handles severe class imbalance"""
    def __init__(self, num_classes, samples_per_class, beta=0.9999, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * num_classes
        
        self.register_buffer('weights', torch.FloatTensor(weights))
        
        print(f"\n{'='*80}")
        print("CLASS-BALANCED LOSS INITIALIZED")
        print(f"{'='*80}")
        print(f"Samples per class: {samples_per_class}")
        print(f"Class weights: {[f'{w:.3f}' for w in weights]}")
        print(f"Weight ratio (max/min): {weights.max()/weights.min():.2f}x")
        print(f"{'='*80}\n")
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            confidence = 1.0 - self.smoothing
            true_dist.scatter_(1, targets.unsqueeze(1), confidence)
        
        weights = self.weights[targets]
        loss = -torch.sum(true_dist * log_probs, dim=1)
        loss = loss * weights
        
        return loss.mean()


# =============================================
# Cosine Classifier
# =============================================
class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(x_norm, w_norm)
        return logits * self.scale


# =============================================
# Warmup Scheduler
# =============================================
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None, metrics=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        elif self.base_scheduler is not None:
            if isinstance(self.base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def setup_logging(save_dir):
    log_file = os.path.join(save_dir, f"training_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def spectrogram_trainer_2d(data_path, user_ids,
                           model_path, normalization_method='none',
                           model_type='lightweight', batch_size=8, epochs=100, lr=0.001, device=None,
                           use_augmentation=False, save_model_checkpoints=True, checkpoint_every=10,
                           max_cache_size=100,
                           use_cosine_classifier=True, cosine_scale=20.0,
                           label_smoothing=0.1, warmup_epochs=5,
                           lr_scheduler_type='cosine', random_seed=None,
                           dropout_rate=0.2, classifier_dropout=0.3, weight_decay=1e-4,
                           test_sessions_per_user=1, val_sessions_per_user=1):
    """
    FIXED trainer with class-balanced loss for imbalanced data
    """

    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Create checkpoint directory
    if save_model_checkpoints:
        os.makedirs(model_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier_type = "cosine" if use_cosine_classifier else "linear"
        run_id = f"{normalization_method}_{model_type}_{classifier_type}_{len(user_ids)}users_{timestamp}"
        run_checkpoint_dir = os.path.join(model_path, run_id)
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        logger = setup_logging(run_checkpoint_dir)
        logger.info(f"Starting FIXED 2D training run: {run_id}")
        logger.info(f"Using Class-Balanced Loss to handle imbalance")
    else:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    # --------------------------
    # Create data loaders
    # --------------------------
    logger.info("Creating session-based data loaders...")
    start_time = time.time()

    try:
        train_loader, val_loader, test_loader, num_classes = create_session_based_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            normalization=normalization_method,
            batch_size=batch_size,
            augment_train=use_augmentation,
            cache_size=max_cache_size,
            test_sessions_per_user=test_sessions_per_user,
            val_sessions_per_user=val_sessions_per_user
        )
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        raise

    data_load_time = time.time() - start_time
    logger.info(f"Data loader creation completed in {data_load_time:.2f}s")

    # --------------------------
    # CRITICAL: Analyze class distribution
    # --------------------------
    logger.info("\n" + "="*80)
    logger.info("ANALYZING CLASS DISTRIBUTION")
    logger.info("="*80)
    
    train_labels = []
    for _, batch_labels in train_loader:
        train_labels.extend(batch_labels.tolist())
    
    from collections import Counter
    train_counter = Counter(train_labels)
    samples_per_class = [train_counter.get(i, 0) for i in range(len(user_ids))]
    
    logger.info(f"Samples per class: {samples_per_class}")
    logger.info(f"Min: {min(samples_per_class)}, Max: {max(samples_per_class)}")
    logger.info(f"Imbalance ratio: {max(samples_per_class)/min([s for s in samples_per_class if s>0]):.2f}x")
    logger.info("="*80 + "\n")

    # Get dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    logger.info(f"Session-based dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Get input shape
    sample_batch = next(iter(train_loader))
    X_sample, y_sample = sample_batch
    input_shape = X_sample[0].shape
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes (users): {num_classes}")

    # --------------------------
    # Create model
    # --------------------------
    if model_type == 'lightweight':
        model = LightweightSpectrogramResNet(
            input_channels=1,
            num_classes=num_classes,
            channels=[32, 64, 128],
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout
        )
    else:
        model = SpectrogramResNet(
            input_channels=1,
            num_classes=num_classes,
            channels=[64, 128, 256, 512],
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout
        )

    # Replace classifier if using cosine
    if use_cosine_classifier:
        if model_type == 'lightweight':
            in_features = 128
        else:
            in_features = 512
        model.classifier = CosineClassifier(in_features, num_classes, scale=cosine_scale)
        logger.info(f"Using Cosine Classifier with scale={cosine_scale}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {model_type}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --------------------------
    # CRITICAL: Use Class-Balanced Loss
    # --------------------------
    criterion = ClassBalancedLoss(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        beta=0.9999,
        smoothing=label_smoothing
    )
    criterion = criterion.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if lr_scheduler_type == 'cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
        )
    else:
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

    scheduler = WarmupScheduler(optimizer, warmup_epochs, base_scheduler)

    logger.info(f"Using warmup for {warmup_epochs} epochs")
    logger.info("Starting training...")

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    learning_rates = []

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Learning rate step
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler_type == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step(epoch, metrics=val_acc)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        learning_rates.append(current_lr)

        # Check improvement
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            improved = "NEW BEST"
            if save_model_checkpoints:
                torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Best Val: {best_val_acc:.4f} {improved}"
        )

        # Per-class accuracy check every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            per_class_correct = [0] * num_classes
            per_class_total = [0] * num_classes
            
            for true, pred in zip(val_true, val_preds):
                per_class_total[true] += 1
                if true == pred:
                    per_class_correct[true] += 1
            
            per_class_acc = [
                per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
                for i in range(num_classes)
            ]
            
            num_zero = sum(1 for acc in per_class_acc if acc == 0)
            logger.info(f"Per-class accuracies: {[f'{a:.3f}' for a in per_class_acc]}")
            
            if num_zero > num_classes * 0.5:
                logger.warning(f"⚠️  {num_zero}/{num_classes} classes have 0% accuracy!")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

        # Periodic checkpoint
        if save_model_checkpoints and (epoch + 1) % checkpoint_every == 0:
            torch.save(model.state_dict(), 
                      os.path.join(run_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    total_training_time = time.time() - training_start
    logger.info(f"Training completed in {total_training_time:.2f}s")

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(run_checkpoint_dir, 'best_model.pth')))
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_preds = []
    test_true = []
    test_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            test_preds.extend(predicted.cpu().tolist())
            test_true.extend(labels.cpu().tolist())
            test_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    kappa = cohen_kappa_score(test_true, test_preds)

    # Per-class test accuracy
    per_class_test_correct = [0] * num_classes
    per_class_test_total = [0] * num_classes
    
    for true, pred in zip(test_true, test_preds):
        per_class_test_total[true] += 1
        if true == pred:
            per_class_test_correct[true] += 1
    
    per_class_test_acc = [
        per_class_test_correct[i] / per_class_test_total[i] if per_class_test_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    logger.info("\n" + "="*80)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Cohen's Kappa: {kappa:.4f}")
    logger.info(f"Per-class test accuracies: {[f'{a:.3f}' for a in per_class_test_acc]}")
    logger.info("="*80 + "\n")

    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'kappa_score': kappa,
        'best_val_acc': best_val_acc,
        'per_class_test_acc': per_class_test_acc,
        'training_completed': True,
        'total_epochs': epoch + 1,
        'total_training_time': total_training_time,
        'num_users': len(user_ids),
        'user_ids': user_ids,
        'samples_per_class': samples_per_class,
        'imbalance_ratio': max(samples_per_class) / min([s for s in samples_per_class if s > 0])
    }

    if save_model_checkpoints:
        with open(os.path.join(run_checkpoint_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_checkpoint_dir, f'training_curves_final.png'), dpi=150)
        plt.close()

    return test_acc, kappa
