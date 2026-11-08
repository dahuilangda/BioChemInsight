"""Peptide monomer definitions, connections, and molecular assembly."""

from .assembly import assemble
from .biln import from_biln, parse_biln, to_biln
from .decomposition import DecompositionResult, UnidentifiedFragment, decompose_smiles
from .errors import PeptideError
from .helm import from_helm, parse_helm, to_helm
from .library import Attachment, Monomer, MonomerLibrary, core_library
from .matching import ResidueMatch
from .model import AtomicConnection, Chain, ChemicalRecord, Connection, Endpoint, PeptideGraph
from .sequence import from_fasta, from_sequence, to_sequence

__version__ = "0.1.0"
__all__ = [
    "DecompositionResult",
    "UnidentifiedFragment",
    "ResidueMatch",
    "decompose_smiles",
    "from_biln",
    "parse_biln",
    "to_biln",
    "Attachment",
    "AtomicConnection",
    "Chain",
    "ChemicalRecord",
    "Connection",
    "Endpoint",
    "Monomer",
    "MonomerLibrary",
    "PeptideError",
    "PeptideGraph",
    "assemble",
    "core_library",
    "from_fasta",
    "from_helm",
    "from_sequence",
    "parse_helm",
    "to_helm",
    "to_sequence",
]
