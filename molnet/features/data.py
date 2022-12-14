from argparse import Namespace

from rdkit import Chem

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from torch_geometric.data import DataLoader, Dataset

from .featurization import ATOMIC_SYMBOLS, atom_features, bond_features

def make_rdkitmol(smi: str, remove_Hs=True, add_Hs=False):
    """
    Builds an RDKit molecule from a SMILES string.

    Args:
        smi: SMILES string.
        remove_Hs: Boolean indicating whether to remove explicit Hs.
                   This does not add Hs. It only keeps them if the smiles has explicit Hs.
        add_Hs: Boolean indicating whether to add explicit hydrogens.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = remove_Hs
    mol = Chem.MolFromSmiles(smi, params)

    if add_Hs:
        mol = Chem.AddHs(mol)

    return mol


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smi: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    - edge_index: A list of tuples indicating presence of bonds.
    """

    def __init__(self, smi: str, args: Namespace):
        """Computes the graph structure and featurization of a molecule."""
        self.smiles = smi   # smiles string

        self.n_atoms = 0    # number of atoms
        self.n_bonds = 0    # number of bonds
        self.f_atoms = []   # mapping from atom index to atom features
        self.f_bonds = []   # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []       # mapping from atom index to incoming bond indices
        self.b2a = []       # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []    # mapping from bond index to the index of the reverse bond
        self.edge_index = []    # list of tuples indicating presence of bonds

        # property prediction of individual molecules
        if not args.cgr:
            # Convert smiles to molecule
            mol = make_rdkitmol(smi, args.remove_Hs, args.add_Hs)

            # define number of atoms
            self.n_atoms = mol.GetNumAtoms()

            if not any(a.GetAtomMapNum() for a in mol.GetAtoms()):
                # this was not an atom-mapped smiles so set an arbitrary atom mapping
                atomMap = list(range(1, self.n_atoms + 1))
                for idx in range(self.n_atoms):
                    atom = mol.GetAtomWithIdx(idx)
                    atom.SetAtomMapNum(atomMap[idx])

            # get atom features
            atoms = sorted(mol.GetAtoms(), key=lambda a: a.GetAtomMapNum())
            self.f_atoms = [atom_features(atom) for atom in atoms]

            # add self loop to account for smiles that only have 1 atom
            if mol.GetNumBonds() == 0:
                self.edge_index.extend([(0, 0), (0, 0)])
                self.f_bonds.append(bond_features(None))
                self.f_bonds.append(bond_features(None))

            # get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    rdkit_idx1 = atoms[a1].GetIdx()
                    rdkit_idx2 = atoms[a2].GetIdx()
                    bond = mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2)

                    if bond is None:
                        continue

                    # create directional graph
                    self.edge_index.extend([(a1, a2), (a2, a1)])
                    self.n_bonds += 2

                    f_bond = bond_features(bond)
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)

        # reaction mode using CGR to represent reactant and product
        else:
            # Convert smiles to molecule
            rmol = make_rdkitmol(smi.split('>')[0], args.remove_Hs, args.add_Hs)
            pmol = make_rdkitmol(smi.split('>')[-1], args.remove_Hs, args.add_Hs)

            # define number of atoms
            if rmol.GetNumAtoms() != pmol.GetNumAtoms():
                raise Exception(
                    f'Reactand and product from {smi} had different number of atoms!\n'
                    f'CGR mode requires balanced reactions!'
                )
            self.n_atoms = rmol.GetNumAtoms()

            if (not any(a.GetAtomMapNum() for a in rmol.GetAtoms()) or
                not any(a.GetAtomMapNum() for a in pmol.GetAtoms())):
                # these were not atom-mapped smiles
                raise Exception(
                    f'{smi} is missing atom map numbers!\n'
                    f'CGR mode requires atom-mapped smiles!'
                )

            # get atom features for each molecule
            r_atoms = sorted(rmol.GetAtoms(), key=lambda a: a.GetAtomMapNum())
            p_atoms = sorted(pmol.GetAtoms(), key=lambda a: a.GetAtomMapNum())
            if [a.GetSymbol() for a in r_atoms] != [a.GetSymbol() for a in p_atoms]:
                raise Exception(f'Atom-mapping is not correct for {smi}')
            f_atoms_reac = [atom_features(atom) for atom in r_atoms]
            f_atoms_prod = [atom_features(atom) for atom in p_atoms]

            # create the diff feature vector
            f_atoms_diff = [list(map(lambda x, y: y - x, ii, jj)) for ii, jj in zip(f_atoms_reac, f_atoms_prod)]

            # reac_diff mode
            self.f_atoms = [x + y[len(ATOMIC_SYMBOLS)+1:] for x, y in zip(f_atoms_reac, f_atoms_diff)]

            # get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    reac_rdkit_idx1 = r_atoms[a1].GetIdx()
                    reac_rdkit_idx2 = r_atoms[a2].GetIdx()
                    bond_reac = rmol.GetBondBetweenAtoms(reac_rdkit_idx1, reac_rdkit_idx2)

                    prod_rdkit_idx1 = p_atoms[a1].GetIdx()
                    prod_rdkit_idx2 = p_atoms[a2].GetIdx()
                    bond_prod = pmol.GetBondBetweenAtoms(prod_rdkit_idx1, prod_rdkit_idx2)

                    if bond_reac is None and bond_prod is None:
                        continue

                    # get bond features for each molecule and create diff feature vector
                    f_bond_reac = bond_features(bond_reac)
                    f_bond_prod = bond_features(bond_prod)
                    f_bond_diff = [y - x for x, y in zip(f_bond_reac, f_bond_prod)]

                    # reac_diff mode
                    f_bond = f_bond_reac + f_bond_diff

                    # create directional graph
                    self.edge_index.extend([(a1, a2), (a2, a1)])
                    self.n_bonds += 2

                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)


class MolDataset(Dataset):
    """
    Class for creating the torch geometric dataset.

    Args:
        args: Command line arguments
        mode: String indicating "train", "val", or "test" mode.
    """
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
        smiles = df.smiles.values

        return [smiles[i] for i in self.split]

    def get_targets(self):
        """Create list of targets"""
        df = pd.read_csv(self.args.data_path)
        targets = df[self.args.targets].values.astype(np.float32)

        return [targets[i] for i in self.split]

    def get_dims(self):
        """Get input dimensions for the node and edge features"""
        if self.args.cgr:
            test_graph = MolGraph(smi='[H:2].[C:1]([H:3])([H:4])[H:4]>>[H:2][C:1]([H:3])([H:4])[H:4]', args=self.args)
            node_dim = len(test_graph.f_atoms[0])
            edge_dim = len(test_graph.f_bonds[0])
        else:
            test_graph = MolGraph(smi='[H][H]', args=self.args)
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

    return loaders
