# train_lfads_gw250_template.py
#
# Michael Nolan
# 2021-08

"""train_lfads_gw250_template.py

Template script for training an LFADS model to reconstruct data samples from the goose-wireless 250Hz dataset
"""

import tspred
import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import yaml
import argparse

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparam_filepath', type=str, default=r'D:\Users\mickey\aoLab\code\timeseries_prediction\tests\hyperparameters\lfads_hyperparameters.yml')
    parser.add_argument('--root_dir', type=str, default=r'D:\Users\mickey\Data\models\pytorch-lightning')
    return parser.parse_args()

def load_hyperparameters(hparam_filepath):
    assert os.path.exists(hparam_filepath), f'hyperparameter file not found: {hparam_filepath}'
    with open(hparam_filepath, mode='r') as hpf:
        return yaml.load(hpf, Loader=yaml.Loader)
    
def get_gw250_datamodule(hparams):
    # return tspred.data.GW250(
    #     src_len = hparams['datamodule']['src_len'], 
    #     trg_len = hparams['datamodule']['trg_len'], 
    #     batch_size = hparams['datamodule']['batch_size'],
    # )
    return tspred.data.GW250_v2(
        src_len = hparams['datamodule']['src_len'], 
        trg_len = hparams['datamodule']['trg_len'], 
        batch_size = hparams['datamodule']['batch_size'],
    )
    #TODO: change the batch size to adaptive
    #TODO: test different num_workers counts

def get_lfads_model(hparams,input_size):
    estimate_loss = torch.nn.MSELoss(reduction='mean')
    generator_ic_prior = tspred.data.LfadsGeneratorICPrior(
        mean = hparams['model']['generator_ic_prior']['mean'],
        mean_opt = False,
        logvar = hparams['model']['generator_ic_prior']['logvar'],
        logvar_opt = False
    )
    return tspred.models.Lfads(
        input_size = input_size,
        encoder_hidden_size = hparams['model']['encoder_hidden_size'],
        encoder_num_layers = hparams['model']['encoder_num_layers'],
        encoder_bidirectional = hparams['model']['encoder_bidirectional'],
        generator_hidden_size = hparams['model']['generator_hidden_size'],
        generator_num_layers = hparams['model']['generator_num_layers'],
        generator_bidirectional = hparams['model']['generator_bidirectional'],
        generator_ic_prior = generator_ic_prior,
        dropout = hparams['model']['dropout'],
        # loss function
        estimate_loss = estimate_loss,
        # objective hyperparameters
        objective_hparams = hparams['objective'],
        # optimizer information
        optimizer_hparams = hparams['optimizer'],
    )

def main():
    # parse command line arguments
    cl_args = parse_cl_args()

    # load hyperparameter file
    hparams = load_hyperparameters(cl_args.hparam_filepath)

    # set, create experiment directory
    root_dir = tspred.utils.create_model_pathstr(hparams)
    root_dir = os.path.join(cl_args.root_dir,root_dir)
    Path(root_dir).mkdir(mode = 0o777, parents=True, exist_ok=True)
    
    # create datamodule
    gw250 = get_gw250_datamodule(hparams)

    # create model
    input_size = gw250.dims[-1]
    model = get_lfads_model(hparams,input_size)
    print(model)

    # create training callbacks (checkpoints)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor = 'avg_valid_loss',
        filename = 'lfads-{epoch:03d}-{val_loss:.3f}',
        save_last = True
    )
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor ='avg_valid_loss'
    )

    # create trainer
    gpus = 1
    trainer = pl.Trainer(
        auto_scale_batch_size = False, # this does not work for external datamodules.
        gradient_clip_val = hparams['trainer']['gradient_clip_val'],
        gradient_clip_algorithm = hparams ['trainer']['gradient_clip_algorithm'],
        min_epochs = hparams['trainer']['min_epochs'],
        max_epochs = hparams['trainer']['max_epochs'],
        default_root_dir = root_dir,
        gpus = gpus,
        replace_sampler_ddp = False,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # train
    trainer.fit(model,gw250)

if __name__ == "__main__":
    main()