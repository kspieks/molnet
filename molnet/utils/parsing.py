from argparse import ArgumentParser


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """
    parser = ArgumentParser()

    # general arguments
    parser.add_argument('--log_dir', type=str,
                        help='Directory to store the log file.')

    parser.add_argument('--data_path', type=str,
                        help='Path to the csv file containing SMILES and targets.')

    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file with train, val, and test indices.')

    parser.add_argument('--model_config', type=str,
                        help='Path to json file containing model parameters. Used when fine-tuning.')

    parser.add_argument('--state_dict', type=str,
                        help='Path to model checkpoint (.pt file). Used when fine-tuning.')

    parser.add_argument('--targets', nargs='+',
                        help='Name of columns to use as target labels.')

    # featurization arguments
    parser.add_argument('--cgr', action='store_true', default=False,
                        help='Boolean indicating whether to use CGR. Use for predicting reaction properties.')

    parser.add_argument('--remove_Hs', action='store_true', default=False,
                        help='Boolean indicating whether to remove explicit hydrogens. Do not use for reaction mode.')

    parser.add_argument('--add_Hs', action='store_true', default=False,
                        help='Boolean indicating whether to add explicit hydrogens. Do not use for reaction mode.')

    # model arguments
    model_config = parser.add_argument_group('model_config')
    model_config.add_argument('--gnn_type', type=str, default='dmpnn',
                              choices=['dmpnn', 'gatv2'],
                              help="Type of GNN to use.")

    model_config.add_argument('--gat_heads', type=int, default=1,
                              help='Number of attention heads.')

    model_config.add_argument('--gnn_hidden_size', type=int, default=300,
                              help='Dimensionality of hidden layers in MPN.')

    model_config.add_argument('--gnn_depth', type=int, default=3,
                              help='Number of message passing steps.')

    model_config.add_argument('--share_gnn_weights',  action='store_true', default=False,
                              help='Boolean indicating whether to share weights across graph convolutional layers.')

    model_config.add_argument('--graph_pool', type=str, default='sum',
                              choices=['sum', 'mean', 'max'],
                              help='How to aggregate atom representations to molecule representation.')

    model_config.add_argument('--aggregation_norm', type=int, default=None,  # use 50
                              help='Number by which to divide summed up atomic features.')

    model_config.add_argument('--ffn_depth', type=int, default=3,
                              help='Number of layers in FFN after MPN encoding.')

    model_config.add_argument('--ffn_hidden_size', type=int, default=None,
                              help='Hidden dim for higher-capacity FFN (defaults to gnn_hidden_size).')

    model_config.add_argument('--dropout', type=float, default=0,
                             help='Dropout probability.')

    model_config.add_argument('--act_func', type=str, default='SiLU',
                              choices=['ReLU', 'ELU', 'LeakyReLU',
                                       'SiLU', 'SELU', 'GELU'],
                              help='Activation function.')

    # training arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the parallel data loading (0 means sequential).')

    parser.add_argument('--seed', type=int, default=0,
                        help='Sets the seed for generating random numbers in PyTorch, numpy and Python.')

    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Number of epochs to run.')

    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of epochs during which learning rate increases linearly from init_lr to max_lr.'
                             'Afterwards, learning rate decreases exponentially from max_lr to final_lr.')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')

    parser.add_argument('--lr_scheduler', type=str, default='noam',
                        choices=['noam'],
                        help='Learning rate scheduler to use.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam'],
                        help='Optimizer to use.')

    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Maximum gradient norm (for gradient clipping).')

    args = parser.parse_args(command_line_args)

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.gnn_hidden_size

    config_dict = dict({})
    group_list = ['model_config']
    for group in parser._action_groups:
        if group.title in group_list:
            config_dict[group.title] = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
    config_dict['model_config']['num_targets'] = len(args.targets)

    if args.gnn_type not in ['dmpnn', 'gatv2']:
        raise ValueError(f"Undefined GNN type called {args.gnn_type}")

    return args, config_dict
