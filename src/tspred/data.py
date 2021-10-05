"""data.py

module containing dataclass, dataset, and dataloader definitions for tspred models.

"""

# data.py

import h5py
import os
import platform
from dataclasses import dataclass
from abc import ABC
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pytorch_lightning as pl

# - - dataclasses - - #

@dataclass
class TspredModelOutput(ABC):
    """TspredModelOutput

    Abstract base dataclass for model outputs. Implements a single output estimate value (est).

    Inheriting classes should add values as required. See LfadsOutput for a simple example.

    Args:
        est (torch.Tensor): output timeseries estimate from a tspred model.
    """
    
    est: torch.Tensor

@dataclass
class LfadsOutput(TspredModelOutput):
    """LfadsOutput

    Args:
        est (torch.Tensor): [n_batch, n_sample, n_channel] timeseries data estimate produced from LFADS models
        generator_ic_params (torch.Tensor): [n_batch, D*n_hidden*n_layer] torch
    """

    generator_ic_params: torch.Tensor

@dataclass
class LfadsGeneratorICPrior:
    """LfadsGeneratorICPrior:

    Args:
        mean (torch.float32): prior distribution mean value
        logvar (torch.float32): prior distribution log variance
        mean_opt (bool): switches model optimization of the prior distribution mean value (default: False)
        logvar_opt (bool): switches model optimization of the prior distribution log variance value (default: False)
    """

    mean: torch.float32
    logvar: torch.float32
    mean_opt: bool = False
    logvar_opt: bool = False


# - - datasets - - #
class EcogSrcTrgDataset(Dataset):
    '''
        Dataset module for src/trg ECoG pair returns from a given tensor source (file location, hdf5)
    '''
    def __init__(self, file_path, src_len, trg_len=None, split_str='train', transform=None):
        None
        self.file_path  = file_path
        self.split_str  = split_str
        self.read_str   = f'{self.split_str}_ecog'
        self.src_len    = src_len
        # fix this. Indexing array[:None] gives you the whole array.
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

#TODO: Does it make sense to have one class for both recon and prediction datasets?
class SrcTrgDataset(Dataset):
    '''
        Dataset module for src/trg pair returns from a given tensor source (file location, hdf5)
    '''
    def __init__(self, file_path, src_len, trg_len=None, trial_idx=None, data_key='ecog', mode='recon', transform=None):
        self.file_path  = file_path
        self.data_key   = data_key
        with h5py.File(self.file_path,'r') as hf:
            self.shape  = hf[self.data_key].shape
        assert mode in ['recon', 'pred'], "Invalid argument: 'mode' string must be 'recon' or 'pred'."
        self.src_len    = src_len
        self.mode       = mode
        if self.mode == 'recon':
            assert src_len <= self.shape[1], f"'src_len' invalid, must be less than or equal to {self.shape[1]}."
            self.trg_len = src_len
        elif self.mode == 'pred':
            if trg_len is None:
                trg_len = src_len
            assert src_len + trg_len <= self.shape[1], f"'src_len' and 'trg_len' may not sum larger than {self.shape[1]}."
            self.trg_len = trg_len
        self.transform  = transform

    def __getitem__(self, index):
        with h5py.File(self.file_path,'r') as hf:
            sample  = hf[self.data_key][index,:,:]
        src = torch.tensor(sample[...,:self.src_len,:])
        if self.mode == 'recon':
            trg = src
        elif self.mode == 'pred':
            trg = torch.tensor(sample[self.src_len:(self.src_len+self.trg_len)])
        return (src, trg)

    def __len__(self):
        return self.shape[0]

# - - LightningDataModules - - #
class GW250(pl.LightningDataModule):
    '''
        Data Module for the (1s max) Goose Wireless dataset.
        
        Data is sampled at 250Hz. No BPF beyond decimation required during downsampling from 1kHz. Dataset has a fixed 80:10:10::train:val:test split.
    '''
    def __init__(self, src_len: int, trg_len: int, batch_size: int, transforms=None, data_device: str='cpu', num_workers: int=4):
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
        self.num_workers    = num_workers

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class GW250_v2(pl.LightningDataModule):
    '''
        Data Module for the (1s max) Goose Wireless dataset.
        
        Data is sampled at 250Hz. No BPF beyond decimation required during downsampling from 1kHz. Dataset has a fixed 80:10:10::train:val:test split.
    '''
    def __init__(self, src_len: int, trg_len: int, batch_size: int, partition = (0.7,0.2,0.1), transforms=None, data_device: str='cpu', num_workers: int=4):
        super().__init__()

        self.src_len    = src_len
        self.trg_len    = trg_len
        self.batch_size = batch_size
        # this is a hdf5 dataset with the following items (flat structure): dt, train_data, valid_data, test_data.
        sys_platform = platform.system()
        if sys_platform == 'Windows': # home computer
            file_path       = r"D:\Users\mickey\Data\datasets\ecog\goose_wireless\gw_250_v2"
        if sys_platform == 'Linux': # ws5
            file_path       = r'/home/ws5/manolan/data/datasets/ecog/goose_wireless/gw_250_v2'
        self.file_path  = file_path
        with h5py.File(self.file_path,'r') as hf:
            self.dims = hf['ecog'].shape
        self.partition      = partition
        self.transforms     = transforms
        self.data_device    = data_device   # I want to keep the data tensors on the CPU, then read batches to the GPU.
        self.num_workers    = num_workers

    def prepare_data(self): # run once. 
        assert os.path.exists(self.file_path), "Dataset file not found, check file path string"
        return None

    #TODO: change this for the SrcTrgDataset() model, where data is not partitioned
    def setup(self, stage=None): # run on each GPU
        self.dataset    = SrcTrgDataset(
            file_path   = self.file_path,
            src_len     = self.src_len,
            trg_len     = self.trg_len,
            data_key    = 'ecog',
            mode        = 'recon',
            transform   = None
        )
        train_idx, val_idx, test_idx = self.create_sample_idx(mode='sequential')
        self.train_dataset  = Subset(self.dataset, train_idx)
        self.val_dataset    = Subset(self.dataset, val_idx)
        self.test_dataset   = Subset(self.dataset, test_idx)
    
    def create_sample_idx(self, mode='sequential'):
        assert mode in ['sequential', 'rand'], "'mode' must be either 'sequential' or 'rand'"
        if mode == 'sequential':
            n_trials        = self.dims[0]
            n_train_trials  = int(n_trials * self.partition[0])
            n_valid_trials  = int(n_trials * self.partition[1])
            n_test_trials   = int(n_trials * self.partition[2])
            all_idx         = np.arange(self.dims[0])
            train_idx       = all_idx[:n_train_trials]
            val_idx         = all_idx[n_train_trials:n_train_trials+n_valid_trials]
            test_idx        = all_idx[-n_test_trials:]
        else:
            raise NotImplementedError()
        return train_idx, val_idx, test_idx

    def train_dataloader(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SequentialSampler(self.train_dataset),
            batch_size = self.batch_size,
            drop_last = False
        )
        return DataLoader(self.train_dataset, sampler=sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SequentialSampler(self.val_dataset),
            batch_size = self.batch_size,
            drop_last = False
        )
        return DataLoader(self.val_dataset, sampler=sampler, num_workers=self.num_workers)
    
    def test_dataloader(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SequentialSampler(self.test_dataset),
            batch_size = self.batch_size,
            drop_last = False
        )
        return DataLoader(self.test_dataset, sampler=sampler, num_workers=self.num_workers)