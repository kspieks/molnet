import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


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

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_loss))[::-1], train_loss, label='Train RMSE')
    ax.plot(np.arange(len(val_loss))[::-1], val_loss, label='Val RMSE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RMSE')
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(log_file), 'train_val_loss.pdf'), bbox_inches='tight')
