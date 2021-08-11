"""data.py

module containing dataclass, dataset, and dataloader definitions for tspred models.

"""

# data.py

import h5py
import os
from dataclasses import dataclass
from abc import ABC
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

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
class EcogSrcTrgDataset(Dataset):
    '''
        Dataset module for src/trg pair returns from a given tensor source (file location, hdf5)
    '''
    def __init__(self, file_path, src_len, trg_len=None, split_str='train', transform=None):
        None
        self.file_path  = file_path
        self.split_str  = split_str
        self.read_str   = f'{self.split_str}_ecog'
        self.src_len    = src_len
        if trg_len:
            trg_len         = src_len
        self.trg_len    = trg_len
        with h5py.File(self.file_path,'r') as hf:
            self.shape      = hf[self.read_str].shape
        # assert self.shape[1] >= src_len + trg_len, f"sequence length cannot be longer than 1/2 data sample length ({self.shape[1]})"
        self.transform  = transform

    def __getitem__(self, index):
        with h5py.File(self.file_path,'r') as hf:
            sample = hf[self.read_str][index,:,:]
        src = torch.tensor(sample[:self.src_len,:], dtype=torch.float32)
        trg = torch.tensor(sample[:self.trg_len,:], dtype=torch.float32)
        return (src,trg)

    def __len__(self):
        return self.shape[0]
#TODO: lots of good stuff in here for non-ecog data. Refactor this into a more general srctrg dataset class


# - - LightningDataModules - - #
class GW250(pl.LightningDataModule):
    '''
        Data Module for the (1s max) Goose Wireless dataset.
        
        Data is sampled at 250Hz. No BPF beyond decimation required during downsampling from 1kHz. Dataset has a fixed 80:10:10::train:val:test split.
    '''
    def __init__(self, src_len, trg_len, batch_size, transforms=None, data_device='cpu'):
        super().__init__()

        self.src_len    = src_len
        self.trg_len    = trg_len
        self.batch_size = batch_size
        # this is a hdf5 dataset with the following items (flat structure): dt, train_data, valid_data, test_data.
        file_path       = "D:\\Users\\mickey\\Data\\datasets\\ecog\\goose_wireless\\gw_250_renorm"
        self.file_path  = file_path
        with h5py.File(self.file_path,'r') as hf:
            self.train_dims = hf['train_ecog'].shape
            self.val_dims   = hf['valid_ecog'].shape
            self.test_dims  = hf['test_ecog'].shape
        self.dims       = ( # use this to create model input sizes
            self.train_dims[0] + self.val_dims[0] + self.test_dims[0],    # n_trial
            self.train_dims[1],                                             # n_sample
            self.train_dims[2]                                              # n_channel
        )
        self.transforms     = transforms
        self.data_device    = data_device   # I want to keep the data tensors on the CPU, then read batches to the GPU.

    def prepare_data(self): # run once. 
        assert os.path.exists(self.file_path), "Dataset file not found, check file path string"
        return None

    def setup(self, stage=None): # run on each GPU
        self.train_dataset  = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'train',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)
        self.val_dataset    = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'valid',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)
        self.test_dataset = EcogSrcTrgDataset(
            file_path   = self.file_path,
            split_str   = 'test',
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            transform   = self.transforms
        )#.to(self.data_device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)