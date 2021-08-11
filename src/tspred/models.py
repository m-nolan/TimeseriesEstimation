"""tspred.py

Package containing pytorch-lightning models for time series data estimation tasks (reconstruction, prediction, forecasting, etc).
"""

import torch
from torch.nn.modules.loss import _Loss
import pytorch_lightning as pl

from .data import TspredModelOutput, LfadsOutput


# - - model classes - - #
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
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.needs_grad], lr = self.hparams['lr'])
        return optimizer

    def _step(self,batch,batch_idx):
        src, trg = batch
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
             generator_hidden_size: int, generator_num_layers: int, generator_bidirectional: bool, dropout: float):
             
        super().__init__()

        self.generator_num_layers = generator_num_layers
        self.generator_hidden_size = generator_hidden_size # important for reshaping later

        # distribution priors
        self.prior_mean = torch.tensor(0.)
        self.prior_logvar = torch.tensor(1.)
        #TODO: turn this into an input argument

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

        # regularization weights
        self.l2_weight = 5e-2
        self.kl_weight = 5e-2
        #TODO: move these to model inputs

        # assign loss function (composition!)
        self.estimate_loss = estimate_loss

        self.lfads_cell = LfadsCell(input_size, encoder_hidden_size, encoder_num_layers, encoder_bidirectional,
                                    generator_hidden_size, generator_num_layers, generator_bidirectional, dropout)
        
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
        err = self.estimate_loss(pred.est,trg)
        kl_div = self.kl_weight*self.generator_ic_kl_div(pred.generator_ic_params)
        gen_ih_l2_norm, gen_hh_l2_norm = self.generator_l2_norm()
        l2_norm = self.l2_weight*gen_hh_l2_norm
        total = err + kl_div + l2_norm
        return total, {'err': err, 'kl_div': kl_div, 'l2_norm': l2_norm, 'total': total}

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