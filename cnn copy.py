"""
We will use this notation:
    - B: size of batch
    - C: number of classes, i.e. NUM_CLASSES
    - D: size of inputs, i.e. INPUT_DIM
    - N: number of training examples
    - N_dev: number of dev examples
"""
import argparse
import copy
import sys
import time
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.use_deterministic_algorithms(True)

OPTS = None

IMAGE_SHAPE = (224, 224)  # Size of fashion images
INPUT_DIM = 50176  # = 224 * 224, total size of vector
NUM_CLASSES = 10  # Number of classes we are classifying over

# Add checkpoint directory
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# CHANGE/EXPERIMENT num_channels, kernel_size, hidden_dim, maybe dropout_prob
class ConvNet(nn.Module):
    def __init__(self, num_channels=5, kernel_size=3, hidden_dim=200, dropout_prob=0.0):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, kernel_size)
        self.maxPool = nn.MaxPool2d(2)
        
        # Calculate output size after convolution and pooling
        # For a 224x224 image with kernel_size=3 and padding=0:
        # After conv: (224-3+1) = 222
        # After first maxpool: 222//2 = 111
        # After second maxpool: 111//2 = 55
        # So final size is 55x55
        final_size = ((IMAGE_SHAPE[0] - kernel_size + 1) // 2) // 2
        flattened_size = final_size * final_size * num_channels
        
        
        self.linear1 = nn.Linear(flattened_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x):
        """Output the predicted scores for each class."""
        B, D = x.shape
        x = x.reshape(B, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])  # Reshape to (B, 1, X, Y)
        
        
        conv = self.conv(x)
        
        conv = F.relu(conv)
        maxPool1 = self.maxPool(conv)
        
        maxPool2 = self.maxPool(maxPool1)
        
        flat = maxPool2.reshape(B, -1)
        
        hidden = self.linear1(flat)
        
        hidden = F.relu(hidden)
        dropout = self.dropout(hidden)
        output = self.linear2(dropout)
        
        return output

def load_data_from_folders(data_path):
    """Load data from multiple folders, each representing a class."""
    # Find all .npy files
    npy_files = glob(os.path.join(data_path, '**/*.npy'), recursive=True)
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_path}")
    
    
    # Create label mapping
    label_map = {}
    current_label = 0
    
    # Load and process images
    all_images = []
    all_labels = []
    
    print("\nProcessing images...")
    for npy_file in tqdm(npy_files, desc="Loading images"):
        # Get label from folder name
        folder_name = os.path.basename(os.path.dirname(npy_file))
        if folder_name not in label_map:
            label_map[folder_name] = current_label
            current_label += 1
        
        # Load and process image
        try:
            img = np.load(npy_file)
            # Resize if needed
            if img.shape != IMAGE_SHAPE:
                img = np.resize(img, IMAGE_SHAPE)
            all_images.append(img.flatten())  # Flatten to 1D vector
            all_labels.append(label_map[folder_name])
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    
    # Split into train, dev, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_dev = torch.tensor(X_dev, dtype=torch.float)
    y_dev = torch.tensor(y_dev, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def save_checkpoint(model, optimizer, epoch, best_dev_acc, checkpoint_path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dev_acc': best_dev_acc,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_dev_acc = checkpoint['best_dev_acc']
        print(f"\nLoaded checkpoint from epoch {epoch} with dev accuracy {best_dev_acc:.5f}")
        return epoch, best_dev_acc
    return 0, -1

def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30):
    """Run the training loop for the model."""
    
    start_time = time.time()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Try to load latest checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    start_epoch, best_dev_acc = load_checkpoint(model, optimizer, checkpoint_path)
    
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch}")
    else:
        best_dev_acc = -1
        start_epoch = 0
    
    best_checkpoint = None
    best_epoch = -1
    
    # Prepare the training dataset
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))]

    for t in range(start_epoch, num_epochs): 
        print(f"\nEpoch {t+1}/{num_epochs}")
        train_num_correct = 0
        total_loss = 0

        # Training loop
        model.train()
        for batch in tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True), desc="Training batches"):
            x_batch, y_batch = batch
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_func(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            train_num_correct += torch.sum(preds == y_batch).item()
            total_loss += loss.item()

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(y_train)
        avg_loss = total_loss / len(train_dataset)
        
        model.eval()
        with torch.no_grad():
            dev_logits = model(X_dev)
            dev_preds = torch.argmax(dev_logits, dim=1)
            dev_acc = torch.mean((dev_preds == y_dev).float()).item()
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc 
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = t
                # Save best model checkpoint
                best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{t+1}_acc_{dev_acc:.5f}.pt")
                save_checkpoint(model, optimizer, t, best_dev_acc, best_checkpoint_path)
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, t, best_dev_acc, checkpoint_path)
        
        print(f"Epoch {t+1: <2}:")
        print(f"  Train Accuracy: {train_acc:.5f}")
        print(f"  Dev Accuracy: {dev_acc:.5f}")
        print(f"  Average Loss: {avg_loss:.5f}")

    # Set the model parameters to the best checkpoint across all epochs
    model.load_state_dict(best_checkpoint) 
    end_time = time.time() 
    print(f'\nTraining completed in {end_time - start_time:.2f} seconds')
    print(f'Best epoch was {best_epoch+1}, dev_acc={best_dev_acc:.5f}')

def evaluate(model, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
    with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
        logits = model(X)  # tensor of size (N, 10)
        y_preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
        acc = torch.mean((y_preds == y).float()).item()
    print(f'    {name} Accuracy: {acc:.5f}')
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['linear', 'mlp2', 'mlp3', 'cnn'])
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-1)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--num-epochs', '-T', type=int, default=30)
    parser.add_argument('--hidden-dim', '-i', type=int, default=200)
    parser.add_argument('--dropout-prob', '-p', type=float, default=0.0)
    parser.add_argument('--cnn-num-channels', '-c', type=int, default=5)
    parser.add_argument('--cnn-kernel-size', '-k', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

def main():
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    # Load the data from folders
    data_path = "/Users/maiankonabang/Desktop/CSCI467/outfit-aesthetic-classifier/outfit-aesthetic-classifier-dataset-web-scraping/dataset"
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data_from_folders(data_path)

    # Train model
    if OPTS.model == 'cnn':
        model = ConvNet(num_channels=OPTS.cnn_num_channels,
                        kernel_size=OPTS.cnn_kernel_size,
                        hidden_dim=OPTS.hidden_dim,
                        dropout_prob=OPTS.dropout_prob)
    train(model, X_train, y_train, X_dev, y_dev, lr=OPTS.learning_rate,
          batch_size=OPTS.batch_size, num_epochs=OPTS.num_epochs)

    # Evaluate the model
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(model, X_test, y_test, 'Test')

if __name__ == '__main__':
    OPTS = parse_args()
    main()

