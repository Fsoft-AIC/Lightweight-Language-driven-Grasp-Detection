import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

class Mask:
    """
    Wrapper around an mask with some convenient functions.
    """

    def __init__(self, mask):
        self.mask = mask

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.mask, attr)
    
    @classmethod
    def from_file(cls, fname):
        return torch.load(fname)