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
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.use_deterministic_algorithms(True)

OPTS = None

#IMAGE_SHAPE = (28, 28)  # Size of fashion images
#INPUT_DIM = 784  # = 28 * 28, total size of vector
NUM_CLASSES = 10  # Number of classes we are classifying over

# CHANGE/EXPERIMENT num_channels, kernel_size, hidden_dim, maybe dropout_prob
class ConvNet(nn.Module):
    def __init__(self, num_channels=5, kernel_size=3, hidden_dim=200, dropout_prob=0.0):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, kernel_size)
        self.maxPool = nn.MaxPool2d(2)
        # CHANGE 845
        self.linear1 = nn.Linear(845, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, NUM_CLASSES)


    def forward(self, x):
        """Output the predicted scores for each class.

        The outputs are the scores *before* the softmax function.

        Inputs:
            x: Torch tensor of size (B, D)
        Outputs:
            Matrix of size (B, C).
        """
        B, D = x.shape
        x = x.reshape(B, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])  # Reshape to (B, 1, X, Y)

        conv = self.conv(x)
        conv = F.relu(conv)
        maxPool = self.maxPool(conv)
        flat = maxPool.reshape(B, -1)
        hidden = self.linear1(flat)
        hidden = F.relu(hidden)
        dropout = self.dropout(hidden)
        output = self.linear2(dropout)
        return output

# CHANGE BATCH_SIZE
def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30):
    """Run the training loop for the model.

    All of this code is highly generic and works for any model that does multi-class classification.

    Args:
        model: A nn.Module model, must take in inputs of size (B, D)
               and output predictions of size (B, C)
        X_train: Tensor of size (N, D)
        y_train: Tensor of size (N,)
        X_dev: Tensor of size (N_dev, D). Used for early stopping.
        y_dev: Tensor of size (N_dev,). Used for early stopping.
        lr: Learning rate for SGD
        batch_size: Desired batch size.
        num_epochs: Number of epochs of SGD to run
    """
    start_time = time.time()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Prepare the training dataset
    # Pytorch DataLoader expects a dataset to be a list of (x, y) pairs
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))]

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1 
    best_checkpoint = None
    best_epoch = -1 

    for t in range(num_epochs): 
        train_num_correct = 0

        # Training loop
        model.train()
                    # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                    # DataLoader automatically groups the data into batches of roughly batch_size
                    # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch
                    # unpack batch, which is a tuple (x_batch, y_batch)
                    # x_batch is tensor of size (B, D)
                    # y_batch is tensor of size (B,)
            optimizer.zero_grad()
                    # Reset the gradients to zero
                    # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                    # So we need to reset to zero before running on a new batch!
            logits = model(x_batch)
                    # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
                    # For MNIST, C=10
            loss = loss_func(logits, y_batch)
                    # Compute the loss of the model output compared to true labels
            loss.backward()
                    # Run backpropagation to compute gradients
            optimizer.step()
                    # Take a SGD step
                    # Note that when we created the optimizer, we passed in model.parameters()
                    # This is a list of all parameters of all layers of the model
                    # optimizer.step() iterates over this list and does an SGD update to each parameter

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1)
                    # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item()

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(y_train) 
        model.eval()
                    # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad():
                    # Don't allocate memory for storing gradients, more efficient when not training
            dev_logits = model(X_dev)
            dev_preds = torch.argmax(dev_logits, dim=1)
            dev_acc = torch.mean((dev_preds == y_dev).float()).item()
            if dev_acc > best_dev_acc:
                # Save this checkpoint if it has best dev accuracy so far
                best_dev_acc = dev_acc 
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = t 
        print(f'Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')

    # Set the model parameters to the best checkpoint across all epochs
    model.load_state_dict(best_checkpoint) 
    end_time = time.time() 
    print(f'Training took {end_time - start_time:.2f} seconds')
    print(f'\nBest epoch was {best_epoch}, dev_acc={best_dev_acc:.5f}')



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

    # Read the data
    # CHANGE DIRECTORY
    all_data = np.load('q4_data.npy', allow_pickle=True).item()
    X_train = torch.tensor(all_data['X_train'], dtype=torch.float)
    y_train = torch.tensor(all_data['y_train'], dtype=torch.long)
    X_dev = torch.tensor(all_data['X_dev'], dtype=torch.float)
    y_dev = torch.tensor(all_data['y_dev'], dtype=torch.long)
    X_test = torch.tensor(all_data['X_test'], dtype=torch.float)
    y_test = torch.tensor(all_data['y_test'], dtype=torch.long)

    # Train model
    if OPTS.model == 'cnn':
        model = ConvNet(num_channels=OPTS.cnn_num_channels,
                        kernel_size=OPTS.cnn_kernel_size,
                        hidden_dim=OPTS.hidden_dim,
                        dropout_prob=OPTS.dropout_prob)
    train(model, X_train, y_train, X_dev, y_dev, lr=OPTS.learning_rate,
          batch_size=OPTS.batch_size, num_epochs=OPTS.num_epochs)

    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    if OPTS.test:
        test_acc = evaluate(model, X_test, y_test, 'Test')

if __name__ == '__main__':
    OPTS = parse_args()
    main()

