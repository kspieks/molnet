import math

import torch
from tqdm import tqdm

from .nn_utils import NoamLR, compute_gnorm, compute_pnorm


def train(model, loader, optimizer, loss, scaler, device, max_grad_norm, scheduler, logger=None):
    model.train()
    rmse_total, mae_total = 0, 0

    scaler.to(device)
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, scaler.transform(data.y))
        result.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)

        if logger:
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            lrs = scheduler.get_lr()
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            logger.info(f'Training RMSE: {math.sqrt(result.item()):.5f}, PNorm: {pnorm:.4f}, GNorm: {gnorm:.4f}, {lrs_str}')

        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()

        preds = scaler.inverse_transform(out)
        rmse_total += (preds - data.y).square().sum(dim=0).detach().cpu()
        mae_total += (preds - data.y).abs().sum(dim=0).detach().cpu()

    # divide by number of molecules
    train_rmse = torch.sqrt(rmse_total / len(loader.dataset))   # rmse with units
    train_mae = mae_total / len(loader.dataset)                 # mae with units

    return train_rmse, train_mae


def eval(model, loader, scaler, device):
    model.eval()
    rmse_total, mae_total = 0, 0
    preds_all = []

    scaler.to(device)
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data)
            preds = scaler.inverse_transform(out)
            preds_all.extend(preds.cpu().detach().tolist())

            rmse_total += (preds - data.y).square().sum(dim=0).detach().cpu()
            mae_total += (preds - data.y).abs().sum(dim=0).detach().cpu()

    # divide by number of molecules
    val_rmse = torch.sqrt(rmse_total / len(loader.dataset)) # rmse with units
    val_mae = mae_total / len(loader.dataset)               # mae with units

    return val_rmse, val_mae, preds_all
