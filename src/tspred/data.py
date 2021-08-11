"""data.py

module containing dataclass, dataset, and dataloader definitions for tspred models.

"""

# data.py

import h5py
import os
from dataclasses import dataclass
from abc import ABC
import torch

# - - dataclasses - - #

@dataclass
class TspredModelOutput(ABC):
    """TspredModelOutput

    Abstract base dataclass for model outputs. Implements a single output estimate value (est).

    Inheriting classes should add values as required. See LfadsOutput for a simple example.
    """
    
    est: torch.Tensor

@dataclass
class LfadsOutput(TspredModelOutput):

    generator_ic_params: torch.Tensor


# - - datasets - - #