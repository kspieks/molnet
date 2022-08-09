# molnet

## Installation
First, clone the repository:
`git clone https://github.com/kspieks/molnet`

These steps will create an environment with python 3.8, pytorch 1.11, and cuda 10.2.

```
# create env
# could use either python 3.7 or 3.8
# conda create -n molnet python=3.7.13 -y
conda create -n molnet python=3.8.13 -y
conda activate molnet

# install pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch -y

# install pytorch geometric
conda install pyg -c pyg -y

# install rdkit
conda install -c conda-forge rdkit -y

# install this repo in editable mode
pip install -e .
```

