from typing import List, Tuple, Union

from rdkit import Chem

ATOMIC_SYMBOLS = ['H', 'C', 'N', 'O']
ATOM_FEATURES = {
    'atomic_symbol': ATOMIC_SYMBOLS,
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 0, 1, 2],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}


def onek_encoding_unk(value, choices: List) -> List[int]:
    """
    Creates a one-hot encoding.

    Args:
        value: The value for which the encoding should be one.
        choices: A list of possible values.

    Returns:
        encoding: A one-hot encoding of the value in a list of length len(choices) + 1.
                  If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a RDKit atom.

    Args:
        atom: A RDKit atom

    Returns:
        fatom: A list containing the atom features
    """
    if atom is None:
        fatom = [0] * ATOM_FDIM

    else:
        fatom = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atomic_symbol']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'] + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features

        fatom += [
                atom.IsInRing(),
                atom.IsInRingSize(3),
                atom.IsInRingSize(4),
                atom.IsInRingSize(5),
                atom.IsInRingSize(6),
                atom.IsInRingSize(7),
            ]
 
    return fatom


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a RDKit bond.

    Args:
        bond: A RDKit bond

    Returns:
        fbond: A list containing the bond features
    """

    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
            (bond.IsInRingSize(3) if bt is not None else 0),
            (bond.IsInRingSize(4) if bt is not None else 0),
            (bond.IsInRingSize(5) if bt is not None else 0),
            (bond.IsInRingSize(6) if bt is not None else 0),
            (bond.IsInRingSize(7) if bt is not None else 0),
        ]
        
    return fbond


# set the atom feature sizes
def set_atom_fdim() -> int:
    """Automatically sets the dimensionality of atom features."""
    global ATOM_FDIM
    mol = Chem.MolFromSmiles('[H][H]')
    atoms = [a for a in mol.GetAtoms()]
    ATOM_FDIM = len(atom_features(atoms[0]))
set_atom_fdim()


# set the bond feature sizes
def set_bond_fdim() -> int:
    """Automatically sets the dimensionality of bond features."""
    global BOND_FDIM
    mol = Chem.MolFromSmiles('[H][H]')
    bonds = [b for b in mol.GetBonds()]
    BOND_FDIM = len(bond_features(bonds[0]))
set_bond_fdim()
