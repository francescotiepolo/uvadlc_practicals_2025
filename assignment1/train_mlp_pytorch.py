################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      llabels: 1D int array of size [batch_size]. Ground truth labels for
               each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with torch.no_grad():
      preds = torch.argmax(predictions, dim=1)
      accuracy = (preds == targets).float().mean().item()
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    device = next(model.parameters()).device
    tot_correct = 0
    tot_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            predictions = torch.argmax(model(inputs), dim=1)
            tot_correct += (predictions == targets).sum().item()
            tot_samples += targets.size(0)

    avg_accuracy = tot_correct / tot_samples
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    x_sample, _ = next(iter(cifar10_loader['train']))
    n_inputs = np.prod(x_sample.shape[1:])
    # TODO: Initialize model and loss module
    model = MLP(n_inputs=n_inputs, n_hidden=hidden_dims, n_classes=10,
                use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # TODO: Training loop including validation
    val_accuracies = []
    train_losses = []
    best_val_accuracy = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in cifar10_loader['train']:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0), -1)
    # TODO: Do optimization with the simple SGD optimizer
            optimizer.zero_grad()
            out = model(x_batch)

            loss = loss_module(out, y_batch)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            num_batches += 1

        avg_loss = tot_loss / num_batches
        train_losses.append(avg_loss)
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model = deepcopy(model)
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'train_losses': train_losses
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    import matplotlib.pyplot as plt

    losses = logging_dict['train_losses']
    plt.plot(losses, label='Training Losses')
    plt.plot(val_accuracies, label='Validation Accuracies')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training Loss and Validation Accuracy over Epochs')
    plt.grid(True)
    plt.savefig('training_plot_pytorch.png')

    print(f'Test accuracy of best model: {test_accuracy*100:.2f}%')

    # Now unisng batch norm
    kwargs['use_batch_norm'] = True
    model_bn, val_accuracies_bn, test_accuracy_bn, logging_dict_bn = train(**kwargs)

    losses = logging_dict_bn['train_losses']
    plt.plot(losses, label='Training Losses (BN)')
    plt.plot(val_accuracies_bn, label='Validation Accuracies (BN)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.title('Training Loss and Validation Accuracy over Epochs (with Batch Norm)')
    plt.grid(True)
    plt.savefig('training_plot_pytorch_bn.png')

    print(f'Test accuracy of best model: {test_accuracy_bn*100:.2f}%')