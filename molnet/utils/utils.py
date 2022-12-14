import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class TorchStandardScaler(nn.Module):
    """
    StandardScaler class to z-score data.

    Args:
        eps: tolerance to avoid dividing by 0.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def fit(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, unbiased=False, keepdim=True)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def transform(self, x):
        x = x - self.mean
        x = x / (self.std + self.eps)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.std + self.eps)
        x = x + self.mean
        return x


def create_logger(name, log_dir):
    """
    Creates a logger with a stream handler and file handler.
    
    Args:
        name (str): The name of the logger.
        log_dir (str): The directory in which to save the logs.
    
    Returns:
        logger: the logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def plot_train_val_loss(log_file):
    """
    Plots the training and validation loss by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    train_loss = []
    val_loss = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Overall Training RMSE' in line:
                train_loss.append(float(line.split(' ')[-1].split('/')[0].rstrip()))
            elif 'Overall Validation RMSE' in line and 'Best' not in line:
                val_loss.append(float(line.split(' ')[-1].split('/')[0].rstrip()))

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(train_loss))[::-1], train_loss, label='Train RMSE')
    ax.plot(np.arange(len(val_loss))[::-1], val_loss, label='Val RMSE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(log_file), 'train_val_loss.pdf'), bbox_inches='tight')


def plot_lr(log_file):
    """
    Plots the learning rate by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    lr = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'lr_0' in line:
                lr.append(float(line.split(' ')[-1].rstrip()))
            if 'Steps per epoch:' in line:
                steps_per_epoch = line.split(' ')[-1].rstrip()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(lr))[::-1], lr)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'learning_rate.pdf'), bbox_inches='tight')


def plot_gnorm(log_file):
    """
    Plots the gradient norm by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    gnorm = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'PNorm' in line:
                # split gives ['Training', 'RMSE:', '0.00327,', 'PNorm:', '127.8966,', 'GNorm:', '2.5143']
                gnorm.append(float(line.split()[6].rstrip(',')))
            if 'Steps per epoch:' in line:
                steps_per_epoch = line.split()[-1].rstrip()
                break

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(gnorm))[::-1], gnorm)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'gnorm.pdf'), bbox_inches='tight')


def plot_pnorm(log_file):
    """
    Plots the parameter norm by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    pnorm = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'PNorm' in line:
                # split gives ['Training', 'RMSE:', '0.00327,', 'PNorm:', '127.8966,', 'GNorm:', '2.5143']
                pnorm.append(float(line.split()[4].rstrip(',')))
            if 'Steps per epoch:' in line:
                steps_per_epoch = line.split()[-1].rstrip()
                break

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(pnorm))[::-1], pnorm)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Parameter Norm')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'pnorm.pdf'), bbox_inches='tight')
