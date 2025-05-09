import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import time
import copy
import os
import numpy as np
from sklearn.metrics import f1_score


torch.use_deterministic_algorithms(True)

class MLP_Head(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super().__init__()  # call init for nn.Module
        
        self.MLP_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 10)  # 10 classes for the final output
        )

    def forward(self, x):
        """Output the predicted scores for each class.

        The outputs are the scores *before* the softmax function.
        """
        z = self.MLP_head(x)
        return z

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
    loss_func = nn.CrossEntropyLoss()  # (QUESTION 4a: line 1)
                    # Cross-entropy loss is just softmax regression loss
    optimizer = optim.SGD(model.MLP_head.parameters(), lr=lr)  # (QUESTION 4a: line 2)
                    # Stochastic gradient descent optimizer

    # Prepare the training dataset
    # Pytorch DataLoader expects a dataset to be a list of (x, y) pairs
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))] # (QUESTION 4a: line 3)

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_f1 = -1 # (QUESTION 4a: line 4)
    best_checkpoint = None # (QUESTION 4a: line 5)
    best_epoch = -1 # (QUESTION 4a: line 6)

    for t in range(num_epochs): # (QUESTION 4a: line 7)
        train_num_correct = 0 # (QUESTION 4a: line 8)

        # Training loop
        model.train()  # (QUESTION 4a: line 9)
                    # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True): # (QUESTION 4a: line 10)
                    # DataLoader automatically groups the data into batchse of roughly batch_size
                    # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch  # (QUESTION 4a: line 11)
                    # unpack batch, which is a tuple (x_batch, y_batch)
                    # x_batch is tensor of size (B, D)
                    # y_batch is tensor of size (B,)
            optimizer.zero_grad()  #(QUESTION 4a: line 12)
                    # Reset the gradients to zero
                    # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                    # So we need to reset to zero before running on a new batch!
            logits = model(x_batch) #(QUESTION 4a: line 13)
                    # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
                    # For MNIST, C=10
            loss = loss_func(logits, y_batch)  #(QUESTION 4a: line 14)
                    # Compute the loss of the model output compared to true labels
            loss.backward()  # (QUESTION 4a: line 15)
                    # Run backpropagation to compute gradients
            optimizer.step() # (QUESTION 4a: line 16)
                    # Take a SGD step
                    # Note that when we created the optimizer, we passed in model.parameters()
                    # This is a list of all parameters of all layers of the model
                    # optimizer.step() iterates over this list and does an SGD update to each parameter

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1) # (QUESTION 4a: line 17)
                    # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item() # (QUESTION 4a: line 18)

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(y_train) # (QUESTION 4a: line 19)
        model.eval()  # (QUESTION 4a: line 20)
                    # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad():  # (QUESTION 4a: line 21)
                    # Don't allocate memory for storing gradients, more efficient when not training
            dev_logits = model(X_dev) # (QUESTION 4a: line 22)
            dev_preds = torch.argmax(dev_logits, dim=1) # (QUESTION 4a: line 23)
            dev_acc = torch.mean((dev_preds == y_dev).float()).item() # (QUESTION 4a: line 24)
            dev_f1 = f1_score(y_dev.cpu(), dev_preds.cpu(), average='macro')
            if dev_f1 > best_dev_f1:  # (QUESTION 4a: line 25)
                # Save this checkpoint if it has best dev accuracy so far
                best_dev_f1 = dev_f1 # (QUESTION 4a: line 26)
                best_checkpoint = copy.deepcopy(model.state_dict()) # (QUESTION 4a: line 27)
                best_epoch = t # (QUESTION 4a: line 28)
        print(f'Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}, dev_f1={dev_f1:.5f}') # (QUESTION 4a: line 29)

    # Set the model parameters to the best checkpoint across all epochs
    model.load_state_dict(best_checkpoint) # (QUESTION 4a: line 30)
    end_time = time.time()  # (QUESTION 4a: line 31)
    print(f'Training took {end_time - start_time:.2f} seconds') # (QUESTION 4a: line 32)
    print(f'\nBest epoch was {best_epoch}, dev_f1={best_dev_f1:.5f}') # (QUESTION 4a: line 33)

def evaluate(model, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
    with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
        logits = model(X)  # tensor of size (N, 10)
        y_preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
        acc = torch.mean((y_preds == y).float()).item()
        f1 = f1_score(y.cpu(), y_preds.cpu(), average='macro')
    print(f'    {name} Accuracy: {acc:.5f}, F1: {f1:.5f}')
    return acc


def main(args):
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    all_data = torch.load(args['dataset_root'])
    X_train = all_data['X_train']
    y_train = all_data['y_train']
    X_dev   = all_data['X_dev']
    y_dev   = all_data['y_dev']
    X_test  = all_data['X_test']
    y_test  = all_data['y_test']

    # # Normalize the data to match imagenet distribution
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # X_train = (X_train - mean) / std
    # X_dev = (X_dev - mean) / std
    # X_test = (X_test - mean) / std

    # Train CNN model
    model = MLP_Head(dropout_prob=args['dropout_prob'])
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
        'learning_rate': 1e-2,
        'batch_size': 32,
        'num_epochs': 50,
        'dropout_prob': 0.2,
        'test': True,
        'dataset_root': '../normalized_data.pt'
    }

    main(args)