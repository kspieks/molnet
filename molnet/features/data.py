from argparse import Namespace

from rdkit import Chem

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader

from .featurization import atom_features, bond_features


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    - edge_index: A list of tuples indicating presence of bonds.
    """

    def __init__(self, smiles: str, args: Namespace):
        """Computes the graph structure and featurization of a molecule."""
        self.smiles = smiles    # smiles string
        
        self.n_atoms = 0    # number of atoms
        self.n_bonds = 0    # number of bonds
        self.f_atoms = []   # mapping from atom index to atom features
        self.f_bonds = []   # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []       # mapping from atom index to incoming bond indices
        self.b2a = []       # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []    # mapping from bond index to the index of the reverse bond
        self.edge_index = []    # list of tuples indicating presence of bonds

        # Convert smiles to molecule
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)

        self.n_atoms = mol.GetNumAtoms()
        
        # get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))

        # initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # add self loop to account for smiles that only have 1 atom, like C, N, or, O
        if mol.GetNumBonds() == 0:
            self.edge_index.extend([(0, 0), (0, 0)])
            self.f_bonds.append(bond_features(None))
            self.f_bonds.append(bond_features(None))

        # get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue
                    
                self.edge_index.extend([(a1, a2), (a2, a1)])

                f_bond = bond_features(bond)
                self.f_bonds.append(f_bond)
                self.f_bonds.append(f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class MolDataset(Dataset):

    def __init__(self, args, mode='train'):
        super(MolDataset, self).__init__()

        self.args = args

        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(self.args.split_path, allow_pickle=True)[self.split_idx]

        self.smiles = self.get_smiles()

        self.targets = self.get_targets()
        self.mean = np.mean(self.targets, axis=0, keepdims=True)
        self.std = np.std(self.targets, axis=0, ddof=0, keepdims=True)

        self.node_dim, self.edge_dim = self.get_dims()

    def get_smiles(self):
        """Create list of smiles"""
        df = pd.read_csv(self.args.data_path)
        smiles = df.smi.values

        return [smiles[i] for i in self.split]

    def get_targets(self):
        """Create list of targets"""
        df = pd.read_csv(self.args.data_path)
        targets = df[self.args.targets].values.astype(np.float32)

        return [targets[i] for i in self.split]

    def get_dims(self):
        """Get input dimensions for the node and edge features"""
        test_graph = MolGraph(smiles='[H][H]', args=self.args)
        node_dim = len(test_graph.f_atoms[0])
        edge_dim = len(test_graph.f_bonds[0])

        return node_dim, edge_dim

    def molgraph2data(self, molgraph, key):
        """Convert MolGraph object to pyg Data object"""
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor([self.targets[key]], dtype=torch.float)
        data.smiles = self.smiles[key]

        return data

    def process_key(self, key):
        """Uses the key to return a specific data sample"""
        smi = self.smiles[key]
        molgraph = MolGraph(smi, self.args)
        data = self.molgraph2data(molgraph, key)

        return data

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, key):
        return self.process_key(key)


def construct_loader(args, modes=('train', 'val')):
    """Create PyTorch Geometric DataLoader"""
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        dataset = MolDataset(args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            )
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders
