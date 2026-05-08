"""
CNN Model Training Script for Emotion Detection (PyTorch) - FAST VERSION
=========================================================================
Preloads all images into RAM for maximum training speed on CPU.

Architecture follows the PRD specification:
    Conv2D(32) -> BN -> Pool -> Conv2D(64) -> BN -> Pool ->
    Conv2D(128) -> BN -> Pool -> Dropout(0.25) -> Flatten ->
    Dense(256) -> Dropout(0.5) -> Dense(7, softmax)

Usage:
    python model/train_model.py --epochs 30

The trained model is saved to model/emotion_model.pth
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.emotion_labels import EMOTION_LABELS, NUM_CLASSES


# ---- Constants ----
IMG_SIZE = 48
BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model.pth")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageFolder sorts alphabetically:
# angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---- CNN Model ----
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout1 = nn.Dropout(0.25)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---- Fast Data Loading (preload into RAM) ----
def load_images_from_directory(root_dir, class_names):
    """Load ALL images into numpy arrays at once for fast training."""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"   WARNING: {class_dir} not found, skipping")
            continue
        
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert('L')  # Grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_idx)
            except Exception as e:
                pass  # Skip corrupted images silently
        
        print(f"     {class_name:>10}: {len([l for l in labels if l == class_idx]):>5} images loaded")
    
    X = np.array(images, dtype=np.float32).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    y = np.array(labels, dtype=np.int64)
    return X, y


def prepare_data(data_path):
    """Load all data into memory, create train/val/test splits."""
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    
    print("Loading training images into memory...")
    X_train_full, y_train_full = load_images_from_directory(train_dir, CLASS_NAMES)
    print(f"\n   Total training images: {len(X_train_full)}")
    
    print("\nLoading test images into memory...")
    X_test, y_test = load_images_from_directory(test_dir, CLASS_NAMES)
    print(f"\n   Total test images: {len(X_test)}")
    
    # Split training into train (90%) and val (10%)
    np.random.seed(42)
    n = len(X_train_full)
    indices = np.random.permutation(n)
    val_size = int(0.1 * n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]
    
    print(f"\n   Final splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Class distribution
    print("\n   Class distribution (train):")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == idx))
        print(f"     {name:>10}: {count:>5} ({count/len(y_train)*100:.1f}%)")
    
    # Create DataLoaders from tensors (FAST - data already in memory)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, y_train


# ---- Data Augmentation (applied on-the-fly to tensors) ----
def augment_batch(images):
    """Apply simple augmentation to a batch of tensors."""
    batch = images.clone()
    # Random horizontal flip (50% chance per image)
    flip_mask = torch.rand(batch.size(0)) > 0.5
    batch[flip_mask] = batch[flip_mask].flip(-1)
    return batch


# ---- Training Functions ----
def compute_class_weights(y_train, num_classes):
    """Compute balanced class weights."""
    class_counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    weights = total / (num_classes * class_counts.astype(float))
    print("\n   Class weights:")
    for idx in range(num_classes):
        label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else f"Class {idx}"
        print(f"     {label:>10}: {weights[idx]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = augment_batch(inputs)  # Apply augmentation
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ---- Plotting ----
def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"\nTraining curves saved to: {save_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.colorbar(im)
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_dir}/confusion_matrix.png")
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification Report:\n{report}")


# ---- Main Training Pipeline ----
def train_model(data_path, epochs=DEFAULT_EPOCHS):
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}\n")
    
    # 1. Load all data into memory
    start_load = time.time()
    train_loader, val_loader, test_loader, y_train = prepare_data(data_path)
    load_time = time.time() - start_load
    print(f"\n   Data loaded in {load_time:.1f}s")
    
    # 2. Build model
    print("\nBuilding CNN model...")
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # 3. Class weights + optimizer
    class_weights = compute_class_weights(y_train, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 4. Training loop
    print(f"\nStarting training for {epochs} epochs...\n")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch+1:>2}/{epochs} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"LR: {lr:.6f} | {epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, MODEL_SAVE_PATH)
            print(f"         -> Best model saved (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # 5. Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    if test_acc >= 0.60:
        print("   TARGET ACCURACY (>=60%) ACHIEVED!")
    else:
        print("   Below target (60%). Consider more epochs.")
    
    # 6. Plots
    plot_training_history(history, PLOT_DIR)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, PLOT_DIR)
    
    file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Model file size: {file_size:.1f} MB")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Emotion Detection (FER-2013)")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "data", "FER 2013"),
                        help="Path to FER-2013 image directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    args = parser.parse_args()
    
    train_dir = os.path.join(args.data_path, "train")
    test_dir = os.path.join(args.data_path, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"ERROR: Dataset not found at '{args.data_path}'")
        print(f"Expected: {args.data_path}/train/ and {args.data_path}/test/")
        sys.exit(1)
    
    train_model(args.data_path, args.epochs)
