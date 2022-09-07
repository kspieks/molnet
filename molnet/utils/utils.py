from typing import Any, List, Optional
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


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
