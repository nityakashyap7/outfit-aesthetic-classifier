# imports
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import shutil
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from glob import glob
from tqdm import tqdm

# with zipfile.ZipFile('dataset.zip', 'r') as zf:
#   zf.extractall('dataset_images')

def print_dataset_structure(root_dir):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = '  ' * level
        npy_count = len([f for f in files if f.endswith('.npy')])
        folder_name = os.path.basename(root)
        print(f"{indent}{folder_name}/ ({npy_count} .npy files)")

# print_dataset_structure('dataset_images')

# shutil.rmtree('dataset_images/__MACOSX', ignore_errors=True)
dataset_root = "dataset_images/dataset"
# print_dataset_structure(dataset_root)

IMAGE_SHAPE = (224, 224)  # Size of fashion images
INPUT_DIM = 150528  # = 3 * 224 * 224, total size of vector
NUM_CLASSES = 10  # Number of classes we are classifying over

def load_data_from_folders(data_path):
    """Load data from multiple folders, each representing a class."""
    print("\n=== Starting Data Loading ===")
    # Find all .npy files
    npy_files = glob(os.path.join(data_path, '**/*.npy'), recursive=True)
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_path}")
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Create label mapping
    label_map = {}
    current_label = 0
    
    # Load and process images
    all_images = []
    all_labels = []
    
    numIters = 0
    print("\nProcessing images...")
    for npy_file in tqdm(npy_files, desc="Loading images"):
      # Get label from folder name
      folder_name = os.path.basename(os.path.dirname(npy_file))
      if folder_name not in label_map:
          label_map[folder_name] = current_label
          current_label += 1
          print(f"\nNew class found: {folder_name} -> {label_map[folder_name]}")
      
      # Load and process image
      try:
          img = np.load(npy_file)
          # Resize if needed
          if img.shape != (3, 224, 224):
              # print(f"Resizing image from {img.shape} to {IMAGE_SHAPE}")
              img = np.resize(img, (3, 224, 224))
          all_images.append(img.flatten())  # Flatten to 1D vector
          all_labels.append(label_map[folder_name])
          numIters += 1
      except Exception as e:
          print(f"Error processing {npy_file}: {str(e)}")
          continue
    
    print("\nConverting to numpy arrays...")
    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    print("\nSplitting into train/dev/test sets...")
    # Split into train, dev, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print("\nConverting to tensors...")
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_dev = torch.tensor(X_dev, dtype=torch.float)
    y_dev = torch.tensor(y_dev, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    print("\n=== Data Loading Complete ===")
    print(f"\nDataset sizes:")
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Development set: {X_dev.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    print("\nLabel mapping:")
    for folder, label in label_map.items():
        print(f"{folder}: {label}")
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

torch.use_deterministic_algorithms(True)

# CHANGE/EXPERIMENT num_channels, kernel_size, hidden_dim, maybe dropout_prob
class ConvNet(nn.Module):
    def __init__(self, num_channels=5, kernel_size=3, hidden_dim=200, dropout_prob=0.0):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size), 
            nn.ReLU(), 
            nn.MaxPool2d(2),

            nn.Conv2d(num_channels, num_channels, kernel_size), 
            nn.ReLU(),  
            nn.MaxPool2d(2),

            nn.Conv2d(num_channels, num_channels, kernel_size), 
            nn.ReLU(), 
            nn.MaxPool2d(2),       
        )

        self.flattened_size = 3380 # 5 * 26 * 26

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

        # self.conv = nn.Conv2d(1, num_channels, kernel_size)
        # self.maxPool = nn.MaxPool2d(2)
        
        # Calculate output size after convolution and pooling
        # For a 224x224 image with kernel_size=3 and padding=0:
        # After conv: (224-3+1) = 222
        # After first maxpool: 222//2 = 111
        # After second maxpool: 111//2 = 55
        # So final size is 55x55
        # final_size = ((IMAGE_SHAPE[0] - kernel_size + 1) // 2) // 2
        # flattened_size = final_size * final_size * num_channels
        
        # print(f"\nCNN Architecture:")
        # print(f"Input shape: {IMAGE_SHAPE}")
        # print(f"After conv: {num_channels} channels")
        # print(f"After maxpool: {(final_size, final_size)}")
        # print(f"Flattened size: {flattened_size}")
        # print(f"Hidden dim: {hidden_dim}")
        # print(f"Output classes: {NUM_CLASSES}")
        
        # self.linear1 = nn.Linear(flattened_size, hidden_dim)
        # self.dropout = nn.Dropout(dropout_prob)


        # self.linear2 = nn.Linear(hidden_dim, NUM_CLASSES)



    def forward(self, x):
        """Output the predicted scores for each class."""
        B, D = x.shape
        x = x.reshape(B, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])  # Reshape to (B, 3, X, Y)
        
        # Print shapes for debugging
        # print(f"\nForward pass shapes:")
        # print(f"Input: {x.shape}")
        
        # conv = self.conv(x)
        # # print(f"After conv: {conv.shape}")
        
        # conv = F.relu(conv)
        # maxPool1 = self.maxPool(conv)
        # # print(f"After first maxpool: {maxPool1.shape}")
        
        # maxPool2 = self.maxPool(maxPool1)
        # # print(f"After second maxpool: {maxPool2.shape}")
        
        # flat = maxPool2.reshape(B, -1)
        # # print(f"After flatten: {flat.shape}")
        
        # hidden = self.linear1(flat)
        # # print(f"After linear1: {hidden.shape}")
        
        # hidden = F.relu(hidden)
        # dropout = self.dropout(hidden)
        # output = self.linear2(dropout)
        # # print(f"After linear2: {output.shape}")

        x = self.features(x)            # (B, C, H, W)
        x = x.view(B, -1)               # flatten to (B, D)
        x = self.classifier(x)          # final FC layers
        return x
        

def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30):
    """Run the training loop for the model."""
    print("\n=== Starting Training ===")
    print(f"Training parameters:")
    print(f"- Learning rate: {lr}")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of epochs: {num_epochs}")
    
    start_time = time.time()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare the training dataset
    print("\nPreparing training dataset...")
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))]
    print(f"Training dataset size: {len(train_dataset)}")

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1 
    best_checkpoint = None
    best_epoch = -1 

    print("\nStarting training loop...")
    for t in range(num_epochs): 
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
            
            # Compute running count of number of training examples correct
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
            dev_f1 = f1_score(y_dev.cpu(), dev_preds.cpu(), average='macro')
            print(f'  Dev F1 Score: {dev_f1:.5f}')
            
            # if dev_acc > best_dev_acc:
            #     best_dev_acc = dev_acc 
            #     best_checkpoint = copy.deepcopy(model.state_dict())
            #     best_epoch = t 

            if dev_acc > best_dev_acc:
              best_dev_acc = dev_acc 
              best_epoch = t 
              torch.save({
                  'epoch': t,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'dev_accuracy': dev_acc,
              }, 'best_model_checkpoint.pt')
            print(f"  🔒 Saved new best model checkpoint (dev_acc={dev_acc:.5f})")
            print(f"  🔒 Saved new best model checkpoint (dev_f1={dev_f1:.5f})")
        
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

def main(args):
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    # Load the data from folders
    data_path = args['dataset_root']
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data_from_folders(data_path)

    # Train CNN model
    model = ConvNet(num_channels=args['cnn_num_channels'],
                    kernel_size=args['cnn_kernel_size'],
                    hidden_dim=args['hidden_dim'],
                    dropout_prob=args['dropout_prob'])
    train(model, X_train, y_train, X_dev, y_dev, lr=args['learning_rate'],
          batch_size=args['batch_size'], num_epochs=args['num_epochs'])

    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    if args['test']:
        test_acc = evaluate(model, X_test, y_test, 'Test')

if __name__ == '__main__':
    args = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_epochs': 10,
        'hidden_dim': 200,
        'dropout_prob': 0.01,
        'cnn_num_channels': 5,
        'cnn_kernel_size': 3,
        'test': True,
        'dataset_root': dataset_root
    }

    main(args)