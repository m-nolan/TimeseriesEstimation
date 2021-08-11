"""tspred.py

Package containing pytorch-lightning models for time series data estimation tasks (reconstruction, prediction, forecasting, etc).
"""


from os import stat
import numpy as np
import torch
from torch.nn.modules.loss import _Loss
import pytorch_lightning as pl

from dataclasses import dataclass
from typing import Any

from torch.nn.modules.linear import Identity

@dataclass
class TimeseriesSample:
    """ TimeseriesSample

    Dataclass for timeseries estimation model outputs. Contains data tensor and dt float values.
    """
    data: torch.Tensor
    latent: Any
    dt: torch.float32
#TODO move this to a data file with other LightningDataModules
# also, do I need this?


class TimeseriesEstimator(pl.LightningModule):
    """TimeseriesEstimator

    Parent class for timeseries estimation models. Not meant to be implemented as-is. Inherit and modify
    """

    def __init__(self,*args,**kwargs):
        """TimeseriesEstimator class initialization method.
        """
        super().__init__()
        # define the pytorch model of child classes here.
        pass

    def forward(self,input: torch.Tensor):
        """TimeseriesEstimator

        Args:
            input (Tensor): timeseries input data formatted as a pytorch tensor. input.size() = [n_batch x n_time x ...]

        Outputs:
            output (Tensor): timeseries output estimate, formatted as a pytorch tensor.
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams['lr'])
        return optimizer

    def _step(self,batch,batch_idx):
        src, trg = batch
        pred = self(src)
        loss, loss_dict = self.loss(pred, trg)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        #TODO: refactor this into a single _step method that all hooks call?
        loss, loss_dict = self._step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are set before this hook is called.
        loss, loss_dict = self._step(batch, batch_idx)
        #TODO: add all other evaluation metrics here, do a nice composition here for those methods
        return loss

    def loss(self,pred,trg):
        pass

class Lfads(TimeseriesEstimator):
    """LFADS: Latent Factor Analysis with Dynamical Systems

    An all-to-all timeseries estimation model constructed from a pair of GRUs, encoder and generator.
    The final encoder state is used to parameterize gaussian distributions from which generator initial conditions are sampled.
    Generator output sequences are linearly combined to create the output estimate.

    LFADS optimization is driven by a target reconstruction error objective (commonly MSE) and two regularization terms.
    The first regularization is a variational restriction of the KL divergence between the distributions parameterized by encoder outputs and a prior distribution model.
    The second regularization is a L2 norm on the generator parameters to restrict model complexity.

    """

    def __init__(self, input_size: int, encoder_hidden_size: int, encoder_num_layers: int, encoder_bidirectional: bool,
                 generator_hidden_size: int, generator_num_layers: int, generator_bidirectional: bool, dropout: float, 
                 estimate_loss: _Loss):
        
        super().__init__()
        
        # distribution priors
        self.prior_mean = torch.tensor(0.)
        self.prior_logvar = torch.tensor(1.)
        self.l2_weight = 5e-2
        self.kl_weight = 5e-2
        #TODO: turn this into an input argument

        # assign loss function (composition!)
        self.estimate_loss = estimate_loss

        # create dropout layer
        encoder_dropout = dropout
        generator_dropout = dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        # calculate encoder/decoder size constants
        self.encoder_out_scale = 2 if encoder_bidirectional else 1
        self.generator_out_scale = 2 if generator_bidirectional else 1
        encoder_state_size = self.encoder_out_scale * encoder_num_layers * encoder_hidden_size
        generator_state_size = self.generator_out_scale * generator_num_layers * generator_hidden_size
        generator_ic_param_size = 2 * generator_state_size

        # create encoder cell
        self.encoder = torch.nn.GRU(
            input_size = input_size,
            hidden_size = encoder_hidden_size,
            num_layers = encoder_num_layers,
            batch_first = True,
            dropout = encoder_dropout,
            bidirectional = encoder_bidirectional
        )
        
        # encoder outputs are split into the mean and logvar of gaussian distributions
        # generator IC values are samples from these distributions
        # -> n_generator_ic_distributions = 1/2 encoder_outputs = 1/2 * n_encoder_dir * encoder_hidden_size * encoder_num_layers = n_generator_dir * generator_hidden_size * generator_num_layers
        # ergo we need a linear layer, no bias, to link these two together if they are different.
        if generator_ic_param_size == encoder_state_size:
            self.enc2gen = torch.nn.Identity()
        else:
            self.enc2gen = torch.nn.Linear(
                in_features = encoder_state_size,
                out_features = generator_ic_param_size,
                bias = False
            )
        #TODO: ^ wrap this into a simpler "encoder output to generator IC" module

        # create generator cell: GRU stack, typically unidirectional. Can receive inputs with a controller module.
        self.generator = torch.nn.GRU(
            input_size = 1, # no direct control inputs to the generator without a controller
            hidden_size = generator_hidden_size,
            num_layers = generator_num_layers,
            batch_first = True,
            dropout = generator_dropout,
            bidirectional = generator_bidirectional
        )
        
        # older LFADS models had intermediate outputs before computing poisson firing rates.
        # this model maps the generator hidden states directly to the outputs.
        
        self.gen2out = torch.nn.Linear(
            in_features = self.generator_out_scale * generator_hidden_size,
            out_features = input_size,
            bias = True
        )

        self.save_hyperparameters()
        # note: all input arguments are saved into a `hparams` struct.
        # access these values from self.hparams.input_features, etc

    def forward(self,src: torch.Tensor):
        """LFADS forward pass

        Args:
            src (torch.Tensor): LFADS source signal. [n_batch, seq_len, n_ch]

        Returns:
            pred (torch.tensor): LFADS signal estimate. Depending on the training task, this may be a reconstruction of the input src or a prediction of a separate target output (trg).
        """
        batch_size, seq_len, input_size = src.shape
        enc_out, enc_last = self.encoder(src)
        # enc_last = self.dropout(enc_last)
        enc_last = enc_last.permute(1,0,2).reshape(batch_size,-1)
        generator_ic_params = self.enc2gen(enc_last)
        generator_ic = self.sample_generator_ic(generator_ic_params)
        generator_ic = generator_ic.reshape(batch_size,self.generator_out_scale*self.hparams.generator_num_layers,self.hparams.generator_hidden_size).permute(1,0,2)
        gen_out, gen_last = self.generator(torch.empty(batch_size,seq_len,1),self.dropout(generator_ic))
        out = self.gen2out(gen_out)
        return (out, generator_ic_params)
        #TODO: consider expanding the output to include the encoder, generator outputs.
        # this would be a good use case for a dataclass instead of just a dict.

    @staticmethod
    def split_generator_ic_params(params: torch.Tensor):
        batch_size, n_param = params.shape
        assert n_param % 2 == 0, f'params input not even in size, size {n_param}'
        n_dist = n_param // 2
        mean, logvar = torch.split(params,n_dist,dim=-1)
        return mean, logvar

    @staticmethod
    def sample_generator_ic(params: torch.Tensor):
        """sample_generator_ic

        Samples gaussian random values for a provided parameter tensor. Parameters are split evenly along the last dimension to create mean, logvariance values.

        Args:
            params (torch.Tensor): mean and logvariance parameters concatenated along their last dimension.

        Returns:
            sample (torch.Tensor): gaussian random samples drawn from distributions parameterized by input params.
        """
        # split the params into mean, logvar
        mean, logvar = Lfads.split_generator_ic_params(params) # is this broken? should it be refactored?
        sample = mean + torch.randn(mean.shape) * torch.exp(logvar).sqrt()
        return sample
    
    def loss(self, pred, trg: torch.Tensor):
        est, generator_ic_params = pred
        err = self.estimate_loss(est,trg)
        kl_div = self.kl_weight*self.generator_ic_kl_div(generator_ic_params)
        gen_ih_l2_norm, gen_hh_l2_norm = self.generator_l2_norm()
        l2_norm = self.l2_weight*gen_hh_l2_norm
        total = err + kl_div + l2_norm
        return total, {'err': err, 'kl_div': kl_div, 'l2_norm': l2_norm, 'total': total}

    def generator_ic_kl_div(self,generator_ic_params):
        post_mean, post_logvar = self.split_generator_ic_params(generator_ic_params)
        kl_div = kldiv_gaussian_gaussian(post_mean, post_logvar, self.prior_mean, self.prior_logvar)
        return kl_div

    def generator_l2_norm(self):
        """Computes the L2 norm of generator parameters

        Returns:
            gen_ih_l2_norm (torch.Tensor): L2 norm of input-hidden interaction matrices in LFADS GRU generator.
            gen_hh_l2_norm (torch.Tensor): L2 norm of hidden-hidden interaction matrices in LFADS GRU generator.
        """
        gen_ih_l2_norm = self.generator.weight_ih_l0.pow(2).sum().sqrt()
        gen_hh_l2_norm = self.generator.weight_hh_l0.pow(2).sum().sqrt()
        return gen_ih_l2_norm, gen_hh_l2_norm


def kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv)

    KL-Divergence between a prior and posterior diagonal Gaussian distribution.

    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).mean(dim=0).sum()
    return klc