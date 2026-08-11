"""Shared imports and path setup for the moe_trainer package.

Every submodule does ``from ._common import *`` so that the original
monolith's import namespace is preserved without each file repeating
30 import lines.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.MolNexTR.model import molnextr  # noqa: E402,F401
from utils.MolNexTR.moe import ATOM_FORMAT, EXPERT_NAMES, MoEDecoder  # noqa: E402,F401
from utils.MolNexTR.moe_confidence import (  # noqa: E402,F401
    confidence_loss,
    fragment_graph_reward,
    smi_tanimoto,
)
from utils.MolNexTR.dataset import TrainDataset, bms_collate  # noqa: E402,F401
from utils.MolNexTR.chemical import convert_graph_to_smiles  # noqa: E402,F401
from training.molnextr_markush.src.moe_dataset import (  # noqa: E402,F401
    FRAGMENT_ATOM_INDEX_ALIGNMENT_METHOD,
    FRAGMENT_DECODER_LINEARIZATION_METHOD,
    MOE_DATA_CONTRACT_VERSION,
    build_moe_df,
    fragment_attachment_contract,
)
from utils.MolNexTR.tokenization import EOS_ID, atomwise_tokenizer  # noqa: E402,F401
from utils.MolNexTR.utils import FORMAT_INFO  # noqa: E402,F401
from training.molnextr_markush.src.moe_sources import (  # noqa: E402,F401
    PRODUCTION_RELATIVE_ROOT,
    discover_available_production_pose_factory,
)

DATA_ROOT = Path(__file__).resolve().parents[2]
