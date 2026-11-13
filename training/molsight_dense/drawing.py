"""Parsing an observed drawing must retain separately drawn hydrogen nodes."""
from rdkit import Chem


def drawing_molecule(record):
    options=Chem.SmilesParserParams();options.parseName=False;options.allowCXSMILES=False
    options.removeHs=not record.get('visible_hydrogens',False)
    molecule=Chem.MolFromSmiles(record['SMILES'],options)
    if molecule is None:raise ValueError('Invalid drawing molecule')
    return molecule
