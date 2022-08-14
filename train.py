import json
import math
import os

import pandas as pd
import torch

from molnet.features.data import construct_loader
from molnet.model.model import GNN
from molnet.model.nn_utils import (NoamLR,
                                   get_optimizer_and_scheduler,
                                   param_count,
                                   set_seed,
                                   )
from molnet.model.training import test, train
from molnet.utils.parsing import parse_command_line_arguments
from molnet.utils.utils import (TorchStandardScaler,
                                create_logger,
                                plot_train_val_loss,
                                plot_lr,
                                plot_gnorm,
                                plot_pnorm,
                                )


args, config_dict = parse_command_line_arguments()

# set seed
set_seed(args.seed)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = create_logger('train', args.log_dir)
logger.info('Using arguments...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

# construct dataloaders and scaler
train_loader, val_loader, test_loader = construct_loader(args, modes=('train', 'val', 'test'))
print(f'\nTraining mean +- 1 std: {train_loader.dataset.mean} +- {train_loader.dataset.std}')
print(f'Validation mean +- 1 std: {val_loader.dataset.mean} +- {val_loader.dataset.std}')
print(f'Testing mean +- 1 std: {test_loader.dataset.mean} +- {test_loader.dataset.std}\n')
scaler = TorchStandardScaler()
targets = torch.tensor(train_loader.dataset.targets, requires_grad=False)
scaler.fit(targets)

# save the model arguments
model_config = config_dict['model_config']
model_config['num_node_features'] = train_loader.dataset.node_dim
model_config['num_edge_features'] = train_loader.dataset.edge_dim
with open('model_config.json', 'w') as f:
    json.dump(model_config, f)
model = GNN(**model_config).to(device)

# get optimizer and scheduler and define loss
optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')

# record parameters
logger.info(f'Total number of parameters: {param_count(model):,}')
logger.info(f'Model architecture is:\n{model}\n')
logger.info(f'Optimizer parameters are:\n{optimizer}\n')
logger.info(f'Scheduler state dict is:')
if scheduler:
    for key, value in scheduler.state_dict().items():
        logger.info(f'{key}: {value}')
    logger.info('')
logger.info(f'Batch size: {args.batch_size}')
logger.info(f'Steps per epoch: {len(train_loader)}')

best_val_rmse = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs+1):
    train_rmse, train_mae = train(model, train_loader, optimizer, loss, scaler, device, args.max_grad_norm, scheduler, logger)
    logger.info(f'Epoch {epoch}: Overall Training RMSE/MAE {train_rmse.mean():.5f}/{train_mae.mean():.5f}')
    for target, rmse, mae in zip(args.targets, train_rmse, train_mae):
        logger.info(f'Epoch {epoch}: {target} Training RMSE/MAE {rmse:.5f}/{mae:.5f}')

    val_rmse, val_mae, _ = test(model, val_loader, scaler, device)
    logger.info(f'Epoch {epoch}: Overall Validation RMSE/MAE {val_rmse.mean():.5f}/{val_mae.mean():.5f}')
    for target, rmse, mae in zip(args.targets, val_rmse, val_mae):
        logger.info(f'Epoch {epoch}: {target} Validation RMSE/MAE {rmse:.5f}/{mae:.5f}')
    
    if scheduler and not isinstance(scheduler, NoamLR):
        scheduler.step(val_rmse)

    if val_rmse.mean() <= best_val_rmse:
        best_val_rmse = val_rmse.mean()
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pt'))

logger.info(f'\nCompleted {args.n_epochs} epochs. Done with training.\n')
logger.info(f'Best Overall Validation RMSE {best_val_rmse:.5f} on Epoch {best_epoch}')

# load best model
model = GNN(**model_config).to(device)
state_dict = torch.load(os.path.join(args.log_dir, 'best_model.pt'), map_location=device)
model.load_state_dict(state_dict)

# predict test data
test_rmse, test_mae, preds = test(model, test_loader, scaler, device)
logger.info(f'Overall Testing RMSE/MAE {test_rmse.mean():.5f}/{test_mae.mean():.5f}')
for target, rmse, mae in zip(args.targets, test_rmse, test_mae):
    logger.info(f'{target} Testing RMSE/MAE {rmse:.5f}/{mae:.5f}')

# save predictions
smiles = test_loader.dataset.smiles
labels = test_loader.dataset.targets

df_smi = pd.DataFrame(smiles, columns=['smiles'])
df_true = pd.DataFrame(labels, columns=[f'{target}_true' for target in args.targets])
df_preds = pd.DataFrame(preds, columns=[f'{target}_pred' for target in args.targets])
df = pd.concat([df_smi, df_true, df_preds], axis=1)
preds_path = os.path.join(args.log_dir, 'test_predictions.csv')
df.to_csv(preds_path, index=False)

# make plots
log_file = os.path.join(args.log_dir, 'train.log')
plot_train_val_loss(log_file)
plot_lr(log_file)
plot_gnorm(log_file)
plot_pnorm(log_file)
