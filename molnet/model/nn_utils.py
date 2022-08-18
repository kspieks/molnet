import math
import random
from argparse import Namespace
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).

    Args:
        optimizer: A PyTorch optimizer.
        warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        total_epochs: The total number of epochs.
        steps_per_epoch: The number of steps (batches) per epoch.
        init_lr: The initial learning rate.
        max_lr: The maximum learning rate (achieved after warmup_epochs).
        final_lr: The final learning rate (achieved after total_epochs).
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        Args:
            current_step: Optionally specify what step to set the learning rate to.
                          If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_noam_lr_scheduler(optimizer: Optimizer, args: Namespace, train_data_size: int) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    Args:
        optimizer: The Optimizer whose learning rate will be scheduled.
        args: Arguments.
        train_data_size: The size of the training dataset.

    Returns:
        An initialized learning rate scheduler.
    """
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.n_epochs],
        steps_per_epoch=train_data_size // args.batch_size,
        init_lr=[args.lr / 10],
        max_lr=[args.lr],
        final_lr=[args.lr / 100]
    )


def get_optimizer_and_scheduler(args, model, train_data_size):
    """
    Get optimizer and learning rate scheduler

    Args:
        args: Arguments.
        model: An nn.Module.
        train_data_size: The size of the training dataset.

    Returns:
        optimizer: torch optimizer.
        scheduler: learning rate scheduler.

    """
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implementer.")

    if args.lr_scheduler == 'noam':
        scheduler = build_noam_lr_scheduler(optimizer=optimizer, args=args, train_data_size=train_data_size)
    else:
        scheduler = None

    return optimizer, scheduler


def set_seed(seed):
    """
    Sets the seed for generating random numbers in for pseudo-random number 
    generators in Python.random, numpy, and PyTorch.

    Args:
        seed (int): The desired seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_weights(model):
    """
    Initializes the weights of a model in place.

    Args:
        model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    Args:
        model: An nn.Module.

    Returns:
        The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    Args:
        activation: The name of the activation function.

    Returns:
        The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'SiLU':
        return nn.SiLU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
