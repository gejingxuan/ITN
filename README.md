# Interaction Transformer Net
Interaction Transformer Net (ITN), is a deep-learning-based framework for PpIs prediction, 
which represents the protein and peptide structures at residue-level with three-dimensional and sequence information, 
and encodes the interaction information between protein and peptide as a graph.

# Usage
## 1. Environment
Here，you can construct the conda environment for ITN from the yaml file:
```
conda env create -f environment.yaml
```
## 2. Input Data Generation
For PpIs structure datasets in PDB format, we use `data_gen.py` to extract features and generate the input data for ITN.

For speedding up, you can only consider the interaction pocket of protein rather than the entire protein by setting `distance`
to define the pocket size, like 5 or 8.
Here, we set the distance to 800000 to cover the entire protein as a consideration.
And you can set the number of process:
```
python data_gen.py --num_process 16 --distance 800000
```
## 3. Training and Prediction
Then we use the input data generated in the last step to train and test model.
For different classes of PpIs, you can set padding size by change the `max_ligand_num_residues` and `max_pocket_num_residues`

Taking the pMHC I PpIs as an example: 
```
python prediction.py --max_ligand_num_residues 10 --max_pocket_num_residues 181
```
