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

    parser.add_argument('--targets', nargs='+',
                        help='Name of columns to use as target labels.')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the parallel data loading (0 means sequential).')

    # model arguments
    parser.add_argument('--gnn_type', type=str, default='dmpnn',
                        choices=['dmpnn', 'gatv2'],
                        help="Type of GNN to use.")

    parser.add_argument('--gat_heads', type=int, default=1,
                        help='Number of attention heads.')

    parser.add_argument('--gnn_hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN.')

    parser.add_argument('--gnn_depth', type=int, default=3,
                        help='Number of message passing steps.')

    parser.add_argument('--graph_pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max'],
                        help='How to aggregate atom representations to molecule representation.')

    parser.add_argument('--aggregation_norm', type=int, default=None,  # use 50
                        help='Number by which to divide summed up atomic features.')

    parser.add_argument('--ffn_depth', type=int, default=3,
                        help='Number of layers in FFN after MPN encoding.')

    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size).')

    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability.')

    parser.add_argument('--act_func', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU'],
                        help='Activation function.')

    # training arguments
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
                        choices=['noam'],
                        help='Optimizer to use.')

    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Maximum gradient norm (for gradient clipping).')

    args = parser.parse_args(command_line_args)

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.gnn_hidden_size

    return args
