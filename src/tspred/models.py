"""tspred.py

Package containing pytorch-lightning models for time series data estimation tasks (reconstruction, prediction, forecasting, etc).
"""

from typing import Optional

import torch
from torch import optim
from torch._C import device
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import pytorch_lightning as pl

from .data import TspredModelOutput, LfadsOutput, LfadsGeneratorICPrior


# - - model classes - - #
class TimeseriesEstimator(pl.LightningModule):
    """TimeseriesEstimator

    Parent class for timeseries estimation models. Not meant to be implemented as-is. Inherit and modify
    """

    def __init__(self,optimizer_hparams: dict):
        """TimeseriesEstimator class initialization method.
        """
        super().__init__()
        self.optimizer_hparams = optimizer_hparams

    def forward(self,input: torch.Tensor):
        """TimeseriesEstimator

        Args:
            input (Tensor): timeseries input data formatted as a pytorch tensor. input.size() = [n_batch x n_time x ...]

        Outputs:
            output (Tensor): timeseries output estimate, formatted as a pytorch tensor.
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], 
            lr = self.optimizer_hparams['lr'],
            betas = self.optimizer_hparams['betas'],
            eps = self.optimizer_hparams['eps']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min', # default
            min_lr = self.optimizer_hparams['min_lr'],
            factor = self.optimizer_hparams['factor'],
            patience = self.optimizer_hparams['patience'],
            cooldown = self.optimizer_hparams['cooldown']
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'avg_valid_loss'
            }
        }

    def _step(self,batch,batch_idx,step_key):
        src, trg = batch
        pred = self(src)
        loss, loss_dict = self.loss(pred, trg)
        self.log(f'{step_key}_loss', loss)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx, 'valid')
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_valid_loss',avg_loss)
        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are set before this hook is called.
        loss, loss_dict = self._step(batch, batch_idx, 'test')
        #TODO: add all other evaluation metrics here, do a nice composition here for those methods
        # I will need to expand the _step() method to return the estimates for spectral estimates
        return loss

    def loss(self, pred: TspredModelOutput, trg: torch.Tensor):
        pass

class MvarCell(nn.Module):
    """Multivariate Autoregression model (MVAR) cell. Computes forward pass of a single time window.

    Args:
        input_size (int): input dimension
        ar_order (int): order of MVAR model. Sum of linear xforms
    """

    def __init__(self, input_size: int, ar_order: int):
        super().__init__()

        self.input_size = input_size
        self.ar_order = ar_order

        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size,out_features=input_size,bias=False) for _ in range(ar_order)]
        )
    
    def forward(self, src: torch.Tensor):
        """MVAR cell forward method. Computes the next output from an ORDER-length initialization.

        Args:
            src (torch.Tensor): [n_batch, ar_order, input_size] tensor of inputs

        Returns:
            out (torch.Tensor): [n_batch, input_size] tensor of outputs for the next time point
        """
        out = torch.zeros(src.shape[0],src.shape[-1])
        for p_idx, layer in enumerate(self.layers):
            out += layer(src[:,p_idx,:])

        return out

class Mvar(TimeseriesEstimator):
    """Multivariate Autoregression model (MVAR) built in the TimeseriesEstimator framework.

    Args:
        input_size (int): input dimension
        ar_order (int): order of MVAR model. Sum of linear xforms
        estimate_loss (_Loss): loss functional for assessing reconstruction accuracy
        optimizer_hyperparams (dict): dictionary of ADAM optimization hyperparameters
    """

    def __init__(self, input_size: int, ar_order: int, estimate_loss: _Loss, optimizer_hparams: dict):
        super().__init__(optimizer_hparams = optimizer_hparams)

        self.input_size = input_size
        self.ar_order = ar_order
        self.estimate_loss = estimate_loss
        
        # create MVAR stack
        self.ar_cell = MvarCell(input_size=input_size, ar_order=ar_order)

    def forward(self, src: torch.Tensor):
        """Forward method for MVAR model. Computes an output estimate for each time point as a linear combination of the previous ar_order time points.

        Args:
            src (torch.Tensor): [n_batch, n_time, input_size] tensor containing a multidimensional source signal

        Returns:
            est (TspredModelOutput): TspredModelOutput object containing the output estimate
        """

        # loop across time points to compute output estimate
        n_batch, n_time, _ = src.shape
        est = torch.zeros(src.shape)
        for t_idx in range(self.ar_order,n_time):
            est[:,t_idx,:] = self.ar_cell(src[:,(t_idx-self.ar_order):t_idx,:])
        est[:,:self.ar_order,:] = est[:,self.ar_order,:].unsqueeze(1)

        return TspredModelOutput(est=est)

    def forecast(self, init: torch.Tensor, n_out: int):
        """MVAR signal forecast. Predict the next n_out time steps from the I.C. init.

        Args:
            init (torch.Tensor): [n_batch, ar_order, input_size] tensor defining the initial condition
            n_out (int): number of time steps to predict
        """
        pass

    def loss(self, pred: TspredModelOutput, trg: torch.Tensor):
        loss = self.estimate_loss(pred.est,trg)
        return loss, {'loss', loss}


# - - S4ID implementation (Subspace linear system ID) - - #

class S4Id(TimeseriesEstimator):

    def __init__():
        pass

    def forward():
        pass

    def loss():
        pass
        

# - - LFADS implementations - - #
class LfadsCell(nn.Module):
    """LfadsCell

    Individual encoder/generator module for larger LFADS models. Refactored to enable multicell composition.
    """

    def __init__(self, input_size: int, encoder_hidden_size: int, encoder_num_layers: int, encoder_bidirectional: bool,
             generator_hidden_size: int, generator_num_layers: int, generator_bidirectional: bool, 
             generator_ic_prior: LfadsGeneratorICPrior, dropout: float, clip_val: float = 5.0):
             
        super().__init__()

        self.generator_num_layers = generator_num_layers
        self.generator_hidden_size = generator_hidden_size # important for reshaping later

        # create dropout layer
        encoder_dropout = dropout
        generator_dropout = dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        # calculate encoder/decoder size constants
        self.encoder_out_scale = 2 if encoder_bidirectional else 1
        self.generator_out_scale = 2 if generator_bidirectional else 1
        encoder_state_size = self.encoder_out_scale * encoder_num_layers * encoder_hidden_size
        self.generator_state_size = self.generator_out_scale * generator_num_layers * generator_hidden_size
        generator_ic_param_size = 2 * self.generator_state_size

        # create encoder cell
        self.encoder = torch.nn.GRU(
            input_size = input_size,
            hidden_size = encoder_hidden_size,
            num_layers = encoder_num_layers,
            batch_first = True,
            dropout = encoder_dropout,
            bidirectional = encoder_bidirectional,
            bias = False
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
            bidirectional = generator_bidirectional,
            bias = False
        )

        # initialize distribution priors
        self.initialize_generator_ic_prior(generator_ic_prior)

        # set generator clipping value
        self.clip_val = clip_val

    def forward(self, src: torch.Tensor):
        """Forward pass method for an individual LFADS cell. Computes an output estimate of equal length to the input tensor src.

        Args:
            src (torch.Tensor): [n_batch, n_time, n_ch] tensor of timeseries data

        Returns:
            gen_out (torch.Tensor): LfadsCell generator outputs
            generator_ic_params (torch.Tensor): n_batch x _ generator IC distribution parameters. Used for KL div. regularization
        """
        batch_size, seq_len, input_size = src.shape
        enc_out, enc_last = self.encoder(src)
        # enc_last = self.dropout(enc_last)
        enc_last = enc_last.permute(1,0,2).reshape(batch_size,-1)
        generator_ic_params = self.enc2gen(enc_last)
        generator_ic = self.sample_generator_ic(generator_ic_params)
        generator_ic = generator_ic.reshape(batch_size,self.generator_out_scale*self.generator_num_layers,self.generator_hidden_size).permute(1,0,2)
        gen_out, gen_last = self.generator(torch.empty(batch_size,seq_len,1),self.dropout(generator_ic))
        gen_out = gen_out.clamp(min=-self.clip_val, max=self.clip_val)
        if gen_out.isnan().any():
            breakpoint()
        return gen_out, generator_ic_params

    @staticmethod
    def split_generator_ic_params(params: torch.Tensor):
        """Splits generator initial condition distribution parameters into mean and log-variance values

        Args:
            params (torch.Tensor): [n_batch, 2*n_dir*n_layer*n_hidden] parameter tensor

        Returns:
            mean (torch.tensor): gaussian distribution mean parameters
            logvar (torch.tensor): gaussian distribution logvar parameters
        """
        batch_size, n_param = params.shape
        assert n_param % 2 == 0, f'params input not even in size, size {n_param}'
        n_dist = n_param // 2
        mean, logvar = torch.split(params,n_dist,dim=-1)
        return mean, logvar

    def sample_generator_ic(self, params: torch.Tensor):
        """sample_generator_ic

        Samples gaussian random values for a provided parameter tensor. Parameters are split evenly along the last dimension to create mean, logvariance values.

        Args:
            params (torch.Tensor): mean and logvariance parameters concatenated along their last dimension.

        Returns:
            sample (torch.Tensor): gaussian random samples drawn from distributions parameterized by input params.
        """
        # split the params into mean, logvar
        mean, logvar = self.split_generator_ic_params(params)
        sample = mean + torch.randn(mean.shape).type_as(mean) * torch.exp(logvar).sqrt()
        return sample

    def initialize_generator_ic_prior(self,generator_ic_prior: LfadsGeneratorICPrior):
        # initialize mean
        self.prior_mean = nn.Parameter(
            torch.ones(self.generator_state_size) * generator_ic_prior.mean,
            requires_grad = generator_ic_prior.mean_opt
        )
        # initialize logvar
        self.prior_logvar = nn.Parameter(
            torch.ones(self.generator_state_size) * generator_ic_prior.logvar,
            requires_grad = generator_ic_prior.logvar_opt
        )

    def generator_ic_kl_div(self,generator_ic_params):
        """Compute variational constraint (KL div.) from generator I.C. parameters

        Args:
            generator_ic_params (torch.Tensor): [n_batch, 2*n_dir*n_layer*n_hidden] parameter tensor

        Returns:
            kl_div (torch.Tensor): Singleton value measuring divergence of the current distribution estimate from the model prior.
        """
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

#TODO: This is a massive class duplication. I should merge this with the original Lfads class.
# What I don't want is a bunch of if/elses in the class methods though. Will revisit.
class LfadsCellModifiedGru(nn.Module):
    """LfadsCell

    Individual encoder/generator module for larger LFADS models. Refactored to enable multicell composition.
    """

    def __init__(self, input_size: int, encoder_hidden_size: int, encoder_num_layers: int, encoder_bidirectional: bool,
             generator_hidden_size: int, generator_num_layers: int, generator_bidirectional: bool, 
             generator_ic_prior: LfadsGeneratorICPrior, dropout: float, clip_val: float = 5.0):
             
        super().__init__()

        self.generator_input_size = 0
        self.generator_num_layers = generator_num_layers
        self.generator_hidden_size = generator_hidden_size # important for reshaping later

        # create dropout layer
        encoder_dropout = dropout
        generator_dropout = dropout
        self.dropout = torch.nn.Dropout(p=dropout)

        # calculate encoder/decoder size constants
        self.encoder_out_scale = 2 if encoder_bidirectional else 1
        self.generator_out_scale = 2 if generator_bidirectional else 1
        encoder_state_size = self.encoder_out_scale * encoder_num_layers * encoder_hidden_size
        self.generator_state_size = self.generator_out_scale * generator_num_layers * generator_hidden_size
        generator_ic_param_size = 2 * self.generator_state_size

        # create encoder cell
        self.encoder = torch.nn.GRU(
            input_size = input_size,
            hidden_size = encoder_hidden_size,
            num_layers = encoder_num_layers,
            batch_first = True,
            dropout = encoder_dropout,
            bidirectional = encoder_bidirectional,
            bias = False,
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
                bias = False,
            )
        #TODO: ^ wrap this into a simpler "encoder output to generator IC" module

        # create generator cell: GRU stack, typically unidirectional. Can receive inputs with a controller module.
        self.generator = GruModified(
            input_size = self.generator_input_size,
            hidden_size = generator_hidden_size,
            num_layers = generator_num_layers,
            batch_first = True,
            dropout = generator_dropout,
            bias = False,
        )

        # initialize distribution priors
        self.initialize_generator_ic_prior(generator_ic_prior)

        # set generator clipping value
        self.clip_val = clip_val

    def forward(self, src: torch.Tensor):
        """Forward pass method for an individual LFADS cell. Computes an output estimate of equal length to the input tensor src.

        Args:
            src (torch.Tensor): [n_batch, n_time, n_ch] tensor of timeseries data

        Returns:
            gen_out (torch.Tensor): LfadsCell generator outputs
            generator_ic_params (torch.Tensor): n_batch x _ generator IC distribution parameters. Used for KL div. regularization
        """
        if len(src.shape) > 3:
            _, batch_size, seq_len, input_size = src.shape
        else:
            batch_size, seq_len, input_size = src.shape
        enc_out, enc_last = self.encoder(src)
        # enc_last = self.dropout(enc_last)
        enc_last = enc_last.permute(1,0,2).reshape(batch_size,-1)
        generator_ic_params = self.enc2gen(enc_last)
        generator_ic = self.sample_generator_ic(generator_ic_params)
        generator_ic = generator_ic.reshape(batch_size,self.generator_out_scale*self.generator_num_layers,self.generator_hidden_size).permute(1,0,2)
        gen_input = torch.empty(batch_size,seq_len,self.generator_input_size).type_as(generator_ic)
        gen_out, gen_last = self.generator(gen_input,self.dropout(generator_ic))
        gen_out = gen_out.clamp(min=-self.clip_val, max=self.clip_val)
        return gen_out, generator_ic_params

    @staticmethod
    def split_generator_ic_params(params: torch.Tensor):
        """Splits generator initial condition distribution parameters into mean and log-variance values

        Args:
            params (torch.Tensor): [n_batch, 2*n_dir*n_layer*n_hidden] parameter tensor

        Returns:
            mean (torch.tensor): gaussian distribution mean parameters
            logvar (torch.tensor): gaussian distribution logvar parameters
        """
        batch_size, n_param = params.shape
        assert n_param % 2 == 0, f'params input not even in size, size {n_param}'
        n_dist = n_param // 2
        mean, logvar = torch.split(params,n_dist,dim=-1)
        return mean, logvar

    def sample_generator_ic(self, params: torch.Tensor):
        """sample_generator_ic

        Samples gaussian random values for a provided parameter tensor. Parameters are split evenly along the last dimension to create mean, logvariance values.

        Args:
            params (torch.Tensor): mean and logvariance parameters concatenated along their last dimension.

        Returns:
            sample (torch.Tensor): gaussian random samples drawn from distributions parameterized by input params.
        """
        # split the params into mean, logvar
        mean, logvar = self.split_generator_ic_params(params)
        z_sample = torch.randn(mean.shape).type_as(mean)
        return mean + z_sample * torch.exp(logvar).sqrt()

    def initialize_generator_ic_prior(self,generator_ic_prior: LfadsGeneratorICPrior):
        # initialize mean
        self.prior_mean = nn.Parameter(
            torch.ones(self.generator_state_size) * generator_ic_prior.mean,
            requires_grad = generator_ic_prior.mean_opt
        )
        # initialize logvar
        self.prior_logvar = nn.Parameter(
            torch.ones(self.generator_state_size) * generator_ic_prior.logvar,
            requires_grad = generator_ic_prior.logvar_opt
        )

    def generator_ic_kl_div(self,generator_ic_params):
        """Compute variational constraint (KL div.) from generator I.C. parameters

        Args:
            generator_ic_params (torch.Tensor): [n_batch, 2*n_dir*n_layer*n_hidden] parameter tensor

        Returns:
            kl_div (torch.Tensor): Singleton value measuring divergence of the current distribution estimate from the model prior.
        """
        post_mean, post_logvar = self.split_generator_ic_params(generator_ic_params)
        kl_div = kldiv_gaussian_gaussian(post_mean, post_logvar, self.prior_mean, self.prior_logvar)
        return kl_div

    def generator_l2_norm(self):
        """Computes the L2 norm of generator parameters

        Returns:
            gen_ih_l2_norm (torch.Tensor): L2 norm of input-hidden interaction matrices in LFADS GRU generator.
            gen_hh_l2_norm (torch.Tensor): L2 norm of hidden-hidden interaction matrices in LFADS GRU generator.
        """
        # old builtin GRU method, should extend
        # gen_ih_l2_norm = self.generator.weight_ih_l0.pow(2).sum().sqrt()
        # gen_hh_l2_norm = self.generator.weight_hh_l0.pow(2).sum().sqrt()
        gen_ih_l2_norm = None
        gen_hh_l2_norm = self.generator.hidden_weight_l2_norm()
        return gen_ih_l2_norm, gen_hh_l2_norm

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
                 generator_ic_prior: LfadsGeneratorICPrior, estimate_loss: _Loss, objective_hparams: dict, optimizer_hparams: dict):
        
        super().__init__(optimizer_hparams = optimizer_hparams)

        # regularization weights
        self.objective_hparams = objective_hparams

        # assign loss function (composition!)
        self.estimate_loss = estimate_loss
        self.lfads_cell = LfadsCellModifiedGru(input_size, encoder_hidden_size, encoder_num_layers, encoder_bidirectional,
                                    generator_hidden_size, generator_num_layers, generator_bidirectional, 
                                    generator_ic_prior, dropout)
        
        # older LFADS models had intermediate outputs before computing poisson firing rates.
        # this model maps the generator hidden states directly to the outputs.
        
        self.gen2out = torch.nn.Linear(
            in_features = self.lfads_cell.generator_out_scale * generator_hidden_size,
            out_features = input_size,
            bias = True
        )

        self.save_hyperparameters()
        # note: all input arguments are saved into a `hparams` struct.
        # access these values from self.hparams.input_features, etc

    def forward(self, src: torch.Tensor):
        """LFADS forward pass

        Args:
            src (torch.Tensor): LFADS source signal. [n_batch, seq_len, n_ch]

        Returns:
            pred (torch.tensor): LFADS signal estimate. Depending on the training task, this may be a reconstruction of the input src or a prediction of a separate target output (trg).
        """
        gen_out, generator_ic_params = self.lfads_cell(src)
        out = self.gen2out(gen_out)
        return LfadsOutput(out, generator_ic_params)
    
    def loss(self, pred: LfadsOutput, trg: torch.Tensor):
        """Compute the LFADS model loss objective for a given prediction and target pair.

        Args:
            pred (LfadsOutput): Output struct containing the estimate (est) and generator IC parameters
            trg (torch.Tensor): Target sequence that pred.est estimates

        Returns:
            total (torch.Tensor): Total LFADS loss objective
            loss_dict (dict): Dictionary of individual loss objective terms: err, kl_div, l2_norm, total.
        """
        est_nan_trials = pred.est.reshape(pred.est.shape[0],-1).isnan().sum(dim=-1) > 0
        err = self.estimate_loss(pred.est[~est_nan_trials,],trg[~est_nan_trials,])
        kl_div = self.objective_hparams['kl']['weight']*self.generator_ic_kl_div(pred.generator_ic_params)
        gen_ih_l2_norm, gen_hh_l2_norm = self.generator_l2_norm()
        l2_norm = self.objective_hparams['l2']['weight']*gen_hh_l2_norm
        total = err + kl_div + l2_norm
        return total, {'err': err, 'kl_div': kl_div, 'l2_norm': l2_norm, 'total': total}

    def validation_epoch_end(self, outputs):
        # update KL, L2 weight scales here
        super().validation_epoch_end(outputs)
        self.update_kl_weight(self.current_epoch)
        self.update_l2_weight(self.current_epoch)

    def _update_regularizer_weight(self,key,epoch):
        kwd = self.objective_hparams[key] # key weight dict
        weight_step = max(epoch - kwd['schedule_start'], 0)
        gated_weight_update = min(
            kwd['max'] * weight_step / kwd['schedule_dur'],
            kwd['max']
        )
        self.objective_hparams[key]['weight'] = max(gated_weight_update, kwd['min'])
        #TODO: this is highly reliant on a known dict structure. This might benefit from a more static dataclass for reg. weights

    def update_kl_weight(self,epoch):
        self._update_regularizer_weight('kl',epoch)

    def update_l2_weight(self,epoch):
        self._update_regularizer_weight('l2',epoch)

    # for multiblock models, this can be expanded by looping across each cell in `lfads_cells`
    def generator_ic_kl_div(self,generator_ic_params):
        return self.lfads_cell.generator_ic_kl_div(generator_ic_params)

    def generator_l2_norm(self):
        """Computes the L2 norm of generator parameters

        Returns:
            gen_ih_l2_norm (torch.Tensor): L2 norm of input-hidden interaction matrices in LFADS GRU generator.
            gen_hh_l2_norm (torch.Tensor): L2 norm of hidden-hidden interaction matrices in LFADS GRU generator.
        """
        return self.lfads_cell.generator_l2_norm()
    #TODO: is there a smarter/prettier way to make these method passthroughs?


# - - GRU modifications - - #

class GruModified(nn.Module):
    """GRU Modified

    Collection of layered GRU cells, modified with update-bias (see GRUCellModified)
    Currently one-directional only.

    Args:
        input_size (int): input dimension. Can be 0 for homogeneous GRU
        hidden_size (int): hidden dimension of each GRU layer
        num_layers (int): number of GRU cells in GRU block. Each layer output serves as input to the next layer.
        batch_first (bool): True if the first dimension of input tensors is the batch size. default=True
        dropout (float): dropout layer probability. default=0.0
        bias (bool): True enables affine candidate hidden state updates. default=False
    """

    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.0, bias=False):
        #TODO: bidirectional RNN with no inputs is just a 2x seq_len unidirectional RNN with different parameters for the 2nd half. I don't want it yet.

    # input_size = 1, # no direct control inputs to the generator without a controller
    #         hidden_size = generator_hidden_size,
    #         num_layers = generator_num_layers,
    #         batch_first = True,
    #         dropout = generator_dropout,
    #         bidirectional = generator_bidirectional,
    #         bias = False
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        # ID to the first layer, then dropout between all subsequent layers, no dropout at output.
        self.dropout_layers = nn.ModuleList([nn.Dropout(p=self.dropout) for _ in range(num_layers-1)])
        self.bias = bias

        # create GRU layers
        gru_layers = []
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                _layer_input_size = self.input_size
            else:
                _layer_input_size = self.hidden_size
            gru_layers.append(
                GruCellModified(
                    input_size = _layer_input_size,
                    hidden_size = self.hidden_size
                ) # not currently modifying input_bias
                #TODO: expand to make input_bias a hyperparameter
            )
        self.gru_layers = nn.ModuleList(gru_layers)

    def forward(self, input: torch.Tensor, h0: torch.Tensor):
        """GRU forward method, modified.

        Args:
            input (torch.Tensor): [n_batch, n_time, input_size] Input sequence data
            h0 (torch.Tensor): [n_batch, hidden_size, num_layers] Initial condition for all GRU hidden states

        Returns:
            output (torch.tensor): [n_batch, n_time, hidden_size] GRU outputs from the last layer
            hn (torch.tensor): [n_batch, hidden_size, num_layers] Last output from all GRU layers
        """

        # reshape input to time index first
        if self.batch_first:
            input = input.permute(1,0,2)
        n_step, n_batch, _ = input.shape

        # stack layers
        if len(h0.shape) > 2:
            h0 = h0.reshape(n_batch,-1)
        
        # preallocate hidden state activity
        hidden = torch.zeros(n_step, n_batch, self.hidden_size).type_as(input)
        for idx in range(n_step):
            _hidden = h0 if idx == 0 else hidden[idx-1,:,:]
            _hidden_next = torch.zeros(n_batch, self.num_layers * self.hidden_size).type_as(input)
            _in = input[idx,]
            for layer_idx, mod in enumerate(self.gru_layers):
                _hout_idx_range = torch.arange(layer_idx*self.hidden_size,(layer_idx+1)*self.hidden_size)
                if layer_idx > 0:
                    _in = self.dropout_layers[layer_idx-1](_in)
                _hidden_next[:,_hout_idx_range] = mod(_in, _hidden[:,_hout_idx_range])
                _in = _hidden_next[:,_hout_idx_range]
            hidden[idx,] = _hidden_next[:,-self.hidden_size:]
        
        # return: last layer's outputs, last output for all layers.
        # reshape if batch_first
        if self.batch_first:
            hidden = hidden.permute(1,0,2)
        return hidden[:,:,_hout_idx_range], _hidden_next.reshape(n_batch,self.hidden_size,self.num_layers)

    def hidden_weight_l2_norm(self):
        return sum(mod.hidden_weight_l2_norm() for mod in self.gru_layers)

#TODO: consider @torch.script.jit for this, it might make things faster
class GruCellModified(nn.Module):
    """GRUCellModified

    Standard GRUCell implementation with a forced bias on the update gate z.
    https://en.wikipedia.org/wiki/Gated_recurrent_unit
    Currently one-directional only.

    Args:
        input_size (int):       dimension of input
        hidden_size (int):      dimension of GRU hidden state
        bias (bool):            enable hidden state update activation function biases. default=False
        update_bias (float):    offset value to update gate activation function. Forces nonzero inputs during learning. default=1.0 

    Returns:
        [type]: [description]
    """

    def __init__(self, input_size, hidden_size, bias=False, update_bias=1.0):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.bias           = bias
        self.update_bias    = update_bias

        # reset, update gate activations are concatenated
        self._ru_size       = self.hidden_size*2

        # create input maps if inputs exist
        if self.input_size > 0:
            # input to reset-update
            self.fc_x_ru    = nn.Linear(in_features=self.input_size, out_features=self._ru_size, bias=False)
            # input to candidate
            self.fc_x_c     = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        
        # hidden to reset-update
        self.fc_h_ru    = nn.Linear(in_features=self.hidden_size, out_features=self._ru_size, bias=self.bias)
        # hidden to candidate
        self.fc_rh_c    = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=self.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """GRUCellModified forward pass method.

        Args:
            x (torch.Tensor): [n_batch, input_size] input tensor value
            h (torch.Tensor): [n_batch, hidden_size] hidden state tensor value

        Returns:
            h_new (torch.tensor): [n_batch, hidden_size] hidden state tensor output
        """

        # compute input update to reset, update gates
        if self.input_size > 0 and x is not None:
            r_x, u_x = torch.split(self.fc_x_ru(x), split_size_or_sections=self.hidden_size, dim=1)
        else:
            r_x = 0
            u_x = 0
        
        # compute hidden update to reset, update gates
        r_h, u_h = torch.split(self.fc_h_ru(h), split_size_or_sections=self.hidden_size, dim=1)

        # compute reset, update gate activity
        r = torch.sigmoid(r_x + r_h)
        u = torch.sigmoid(u_x + u_h + self.update_bias)

        # compute input update to candidate hidden state
        c_x = self.fc_x_c(x) if self.input_size > 0 and x is not None else 0

        # compute candidate hidden state
        c = torch.tanh(c_x + self.fc_rh_c(r*h))

        # compute new hidden state
        h_new = u * h + (1 - u) * c

        return h_new

    def hidden_weight_l2_norm(self):
        l2_norm = lambda x: x.norm(2).pow(2)/x.numel()
        return l2_norm(self.fc_h_ru.weight) + l2_norm(self.fc_rh_c.weight)


# - - utils, etc - - #

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

