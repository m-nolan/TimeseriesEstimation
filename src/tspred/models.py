"""tspred.py

Package containing pytorch-lightning models for time series data estimation tasks (reconstruction, prediction, forecasting, etc).
"""

from typing import Optional

import torch
from torch import optim
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
        return optimizer

    def _step(self,batch,batch_idx):
        src, trg = batch
        if src.isnan().sum() > 0:
            print('nan values found in sample')
            breakpoint()
            #TODO: get rid of this once you've cleared the src values
        pred = self(src)
        loss, loss_dict = self.loss(pred, trg)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are set before this hook is called.
        loss, loss_dict = self._step(batch, batch_idx)
        #TODO: add all other evaluation metrics here, do a nice composition here for those methods
        # I will need to expand the _step() method to return the estimates for spectral estimates
        return loss

    def loss(self, pred: TspredModelOutput, trg: torch.Tensor):
        pass

class LfadsCell(torch.nn.modules.Module):
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

    @classmethod
    def sample_generator_ic(cls, params: torch.Tensor):
        """sample_generator_ic

        Samples gaussian random values for a provided parameter tensor. Parameters are split evenly along the last dimension to create mean, logvariance values.

        Args:
            params (torch.Tensor): mean and logvariance parameters concatenated along their last dimension.

        Returns:
            sample (torch.Tensor): gaussian random samples drawn from distributions parameterized by input params.
        """
        # split the params into mean, logvar
        mean, logvar = cls.split_generator_ic_params(params)
        sample = mean + torch.randn(mean.shape) * torch.exp(logvar).sqrt()
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

class LfadsCellModifiedGru(torch.nn.modules.Module):
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
        self.generator = GruModified(
            input_size = self.generator_input_size,
            hidden_size = generator_hidden_size,
            num_layers = generator_num_layers,
            batch_first = True,
            dropout = generator_dropout,
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
        gen_out, gen_last = self.generator(torch.empty(batch_size,seq_len,self.generator_input_size),self.dropout(generator_ic))
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

    @classmethod
    def sample_generator_ic(cls, params: torch.Tensor):
        """sample_generator_ic

        Samples gaussian random values for a provided parameter tensor. Parameters are split evenly along the last dimension to create mean, logvariance values.

        Args:
            params (torch.Tensor): mean and logvariance parameters concatenated along their last dimension.

        Returns:
            sample (torch.Tensor): gaussian random samples drawn from distributions parameterized by input params.
        """
        # split the params into mean, logvar
        mean, logvar = cls.split_generator_ic_params(params)
        sample = mean + torch.randn(mean.shape) * torch.exp(logvar).sqrt()
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
                 generator_ic_prior: LfadsGeneratorICPrior, estimate_loss: _Loss, optimizer_hparams: dict):
        
        super().__init__(optimizer_hparams = optimizer_hparams)

        # regularization weights
        self.l2_weight = 5e-2
        self.kl_weight = 5e-2
        #TODO: move these to model inputs

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
        kl_div = self.kl_weight*self.generator_ic_kl_div(pred.generator_ic_params)
        gen_ih_l2_norm, gen_hh_l2_norm = self.generator_l2_norm()
        l2_norm = self.l2_weight*gen_hh_l2_norm
        total = err + kl_div + l2_norm
        return total, {'err': err, 'kl_div': kl_div, 'l2_norm': l2_norm, 'total': total}

    def validation_epoch_end(self, outputs):
        # update KL, L2 weight scales here
        pass

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
        hidden = torch.zeros(n_step, n_batch, self.hidden_size)
        for idx in range(n_step):
            _hidden = h0 if idx == 0 else hidden[idx-1,:,:]
            _hidden_next = torch.zeros(n_batch, self.num_layers * self.hidden_size)
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

