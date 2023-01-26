import os
import sys
import math
from absl import logging
import typing
from tqdm import tqdm

import numpy as np 
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import pytorch_lightning as pl
# third party
from tsalib import dim_vars as dvs, get_dim_vars, update_dim_vars_len
from tsalib import warp
# custom modules
from .base import FunctionFlowModule
from ..nets import MLP, SetFourierFeatures, SetTransformerEncoder, DeepSetsEncoder
from ..utils import frange_cycle_linear
# plotting
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('off')

_B, _S, _C, _T, _L = dvs('Batch(b):10 SetSize(s):1 Context(c):1 Target(t):1 SeqLength(l):1', exists_ok=True)
_X, _Y, _Z, _R = dvs('DimX(x):1 DimY(y):1 DimZ(z):1 DimR(r):1', exists_ok=True)

class NeuralProcess(FunctionFlowModule):
  
  def __init__(self,
               x_dim: int,
               y_dim: int, 
               r_dim: int, 
               z_dim: int,
               batch_size: int,
               n_hidden_units_enc: int = 16, 
               n_hidden_units_dec: int = 16, 
               n_hidden_layers: int = 2,      
               n_inds: int = 0,
               y_sigma_lb: float = 0.05,          
               epsilon: float = 1e-4,
               use_fourier_features: bool = False,
               num_frequencies: int = 20,
               rff_init_std: float = 1.0,
               set_encoder_type: str = 'set_transformer',
               training_objective: str = 'elbo', 
               n_latent_samples: int = 20,
               optim_lr: float = 0.0001, 
               weight_decay: float = 0):
    
    """
    Neural Processes with Set Transformer Encoders.
    
    Args:
    ---------------------------------------------------------
    x_dim: int
        Dimension of inputs
        
    y_dim: int
        Dimension of outputs
        
    r_dim: int
        Dimension of representations
        
    z_dim: int
        Dimension of latent variables
      
    batch_size: int
        Number of examples in one batch
        
    n_hidden_units_enc: int, optional
        Number of hidden units in MLPs in the encoders
        
    n_hidden_units_dec: int, optional
        Number of hidden units in MLPs in the decoders
        
    n_hidden_layers: int, optional
        Number of MLP layers
        
    epsilon: float, optional
        A small float number to avoid numberical issue.
        
    optim_lr: float, optional
        Learning rate of optimizers.
        
    profiler: BaseProfiler, optional
        Tools to profile your training/testing/inference run can help you identify 
        bottlenecks in your code.
    """
    
    super().__init__(has_context_set=True,
                     optim_lr=optim_lr, 
                     weight_decay=weight_decay)
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.ori_x_dim = self.x_dim 
    self.r_dim = r_dim 
    self.z_dim = z_dim
    self.n_hidden_layers = n_hidden_layers
    self.n_inds = n_inds 
    self.n_hidden_units_enc = n_hidden_units_enc
    self.n_hidden_units_dec = n_hidden_units_dec
    self.batch_size = batch_size 
    self.y_sigma_lb = y_sigma_lb
    self.z_sigma_lb = self.y_sigma_lb
    self.epsilon = torch.ones([], device=self.device) * epsilon 
    self.training_objective = training_objective
    self.n_latent_samples = n_latent_samples
    self.set_encoder_type = set_encoder_type
    
    self.use_fourier_features = use_fourier_features
    self.num_frequencies = num_frequencies
    self.rff_init_std = rff_init_std
    
    if self.use_fourier_features:
      self.fourier_features = SetFourierFeatures(x_dim=x_dim, 
                                                 num_frequencies=self.num_frequencies, 
                                                 learnable=False,
                                                 init_std=rff_init_std)
      self.x_dim = 2 * self.num_frequencies

    self._dec_mlp = MLP(self.z_dim+self.x_dim, 
                        [self.n_hidden_units_dec] * self.n_hidden_layers
                        + [2*self.y_dim]).to(self.device)
    
    if self.set_encoder_type == 'set_transformer':
      self._set_encoder = SetTransformerEncoder(dim_input=self.x_dim+self.y_dim,
                                                dim_output=self.r_dim,
                                                dim_hidden=self.n_hidden_units_enc,
                                                n_sabs=self.n_hidden_layers,
                                                n_inds=self.n_inds,
                                                aggregation=True).to(self.device)
    elif self.set_encoder_type == 'deep_sets':
      self._set_encoder = DeepSetsEncoder(dim_input=self.x_dim+self.y_dim,
                                          dim_output=self.r_dim,
                                          dim_hidden=self.n_hidden_units_enc,
                                          n_layers=self.n_hidden_layers,
                                          aggregation=True).to(self.device)
    else:
      raise NotImplementedError

    self._mu_sigma_mlp = MLP(self.r_dim, [2*self.z_dim]).to(self.device)
    
    self.params = self.parameters()
    
    update_dim_vars_len({'b': self.batch_size,
                         'x': self.x_dim, 
                         'y': self.y_dim, 
                         'r': self.r_dim,
                         'z': self.z_dim})
    
    logging.debug("---------       Neural Processes       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
  
  def _sample_with_reparameterization(self, normal_dist):
    eps = torch.normal(mean=0., std=1., size=normal_dist.mean.shape, device=self.device)
    samples =  eps * normal_dist.stddev + normal_dist.mean
    return samples# .detach()
    
  def infer(self, 
            X_set: torch.Tensor, 
            Y_set: torch.Tensor,
            update: bool =False) -> D.Distribution:
    """
    Infer Gaussian distribution over latent variables given a set of points.
    
    Args:
    -------------------------------------------------------------------------
    X_set: torch.Tensor
        Inputs to the encoders, tensor of shape [Batch_Size, Num_Points, X_Dim]
        
    Y_set: torch.Tensor
        Outputs to the encoders, tensor of shape [Batch_Size, Num_Points, X_Dim]
    
    Returns:
    ----------------------------------------------------------------------------
    torch.distributions.Normal, Gaussian Distributions with mean and std of 
    shape [Batch_Size, Z_Dim].
    
    """
    if X_set is None or Y_set is None:
      z_mu, z_sigma = torch.zeros(self.batch_size, self.z_dim, device=self.device), torch.ones(self.batch_size, self.z_dim, device=self.device)
      return D.Normal(loc=z_mu, scale=z_sigma)
    r: 'b,r' = self._set_encoder(torch.cat([X_set, Y_set], dim=-1), update)
    out = self._mu_sigma_mlp(r)
    z_mu = out[:, :self.z_dim]
    z_sigma = (1-self.z_sigma_lb) * torch.sigmoid(out[:, self.z_dim:]) + self.z_sigma_lb
    return D.Normal(loc=z_mu, scale=z_sigma)
  
  def predict(self, 
              z: torch.Tensor, 
              X_t: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Given latent variables z, predict outputs Y_t given the query inputs X_t
    
    Args:
    ----------------------------------------------------------------------
    z: torch.Tensor
        Latent variables, tensor of shape [Batch_Size, Z_Dim]
    
    X_t: torch.Tensor
        Query (target) inputs, tensor of shape [Batch_size, Num_Target_Points, X_Dim]
    
    Returns:
    ----------------------------------------------------------------------
    Predictive mean and std.
    Mean: Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    Std: Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    """
    
    update_dim_vars_len({'t': X_t.shape[1], 'b': X_t.shape[0]})
    _B, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b c t l x y z r')
      
    z = warp(z, 'b,z -> b,1,z -> b,t,z -> b*t,z', 'arv')
    out: 'b*t,2*y' = self._dec_mlp(torch.cat([warp(X_t, 'b,t,x -> b*t,x', 's'), z], dim=-1))
    y_mu: 'b*t,y' = out[:, :self.y_dim]
    y_sigma: 'b*t,y' = (1-self.y_sigma_lb) * torch.nn.Softplus()(out[:, self.y_dim:]) + self.y_sigma_lb
    y_mu, y_sigma = warp(y_mu, 'b*t,y -> b,t,y', 'v'), warp(y_sigma, 'b*t,y -> b,t,y', 'v')
    return y_mu, y_sigma

    
  def forward(self, 
              X_t: torch.Tensor, 
              X_c: torch.Tensor, 
              Y_c: torch.Tensor,
              update: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Given the context set, produces the emprical prior. Sample latents from the prior, 
    and condition on it to make predictions.
    
    Args:
    ---------------------------------------------------------
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
    
    X_c: torch.Tensor
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, X_Dim]
    
    Y_c: torch.Tensor
        Context outputs, tensor of shape [Batch_Size, Num_Context_Points, Y_Dim]
        
    Returns
    ---------------------------------------------------------
    Predictive mean and std.
    Mean: Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    Std: Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    """
      
    update_dim_vars_len({'c': X_c.shape[1], 't': X_t.shape[1], 'b': X_c.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    
    prior = self.infer(X_c, Y_c, update)
    z = self._sample_with_reparameterization(prior)
    y_mu, y_sigma = self.predict(z, X_t)
    return y_mu, y_sigma 
  
  def elbo(self, 
           X_t: torch.Tensor, 
           Y_t: torch.Tensor, 
           X_c: torch.Tensor, 
           Y_c: torch.Tensor) -> typing.Tuple[torch.Tensor, dict]:
    """
    Compute ELBO from the context and target sets.
    
    Args:
    ---------------------------------------------------------
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
    
    Y_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    X_c: torch.Tensor
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, X_Dim]
    
    Y_c: torch.Tensor
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, Y_Dim]
        
    Returns:
    ---------------------------------------------------------
     
    """
    
    update_dim_vars_len({'c': X_c.shape[1], 't': X_t.shape[1], 'b': X_c.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    # Prior distribution conditioning on the context
    q_z_pri = self.infer(X_c, Y_c) 
    # Posterior distribution conditioning on the context & target
    X_ct = torch.cat([X_c, X_t], dim=1)
    Y_ct = torch.cat([Y_c, Y_t], dim=1)
    q_z_pos = self.infer(X_ct, Y_ct) 
    
    kld = torch.distributions.kl.kl_divergence(q_z_pos, q_z_pri).sum(dim=1) / Y_t.shape[0]
    z: 'b,z' = self._sample_with_reparameterization(q_z_pos)
    y_mu, y_sigma = self.predict(z, X_t)
    log_likelihood = D.Normal(loc=y_mu, scale=y_sigma).log_prob(Y_t).sum(dim=-1).mean(dim=1)
    # Compute ELBO normalised by the number of target points
    normalized_elbo = log_likelihood - kld
    return normalized_elbo, {"nll": -log_likelihood, "kld": kld}
  
  def mll(self, 
          X_t: torch.Tensor, 
          Y_t: torch.Tensor, 
          X_c: typing.Union[torch.Tensor, None] = None, 
          Y_c: typing.Union[torch.Tensor, None] = None,
          n_latent_samples: int = 100,
          use_importance_sampling: bool = True) -> torch.Tensor:
    """
    Estimate marginal log likelihood using n_latent_samples Latent samples.
    
    \log p(Y_t | X_t, X_c, Y_c) = \log (1/L \sum_{l=1}^{L} (p(z_l | X_c, Y_c) p(Y_t | X_t, z_l) / p(z_l | X_t, Y_t, X_c, Y_c)))
    where z_l \sim q(z | X_t, Y_t, X_c, Y_c).
    
    Args:
    ---------------------------------------------------------
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
    
    Y_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    X_c: torch.Tensor, optional
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, X_Dim]
    
    Y_c: torch.Tensor, optional
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, Y_Dim]
        
    n_latent_samples: int, optional
        Number of latent samples used to estimate the marginal log likelihood.
        
    Returns:
    ---------------------------------------------------------
    Marginal log likelihood, tensor of shape [Batch_Size,]
    
    """
    ls = []
    for _ in range(n_latent_samples):
      if use_importance_sampling:
        if self.has_context_set:
          X_ct = torch.cat([X_c, X_t], dim=1)
          Y_ct = torch.cat([Y_c, Y_t], dim=1)
          p_z_pos = self.infer(X_ct, Y_ct)
          z = self._sample_with_reparameterization(p_z_pos)
          y_mu, y_sigma = self.predict(z, X_t)
          log_likelihood = D.Normal(loc=y_mu, scale=y_sigma).log_prob(Y_t)
          log_likelihood = log_likelihood.sum(dim=-1).sum(dim=1) 
          log_prior = self.infer(X_c, Y_c).log_prob(z).sum(dim=1)
          log_pos = self.infer(X_ct, Y_ct).log_prob(z).sum(dim=1)
        else:
          p_z_pos = self.infer(X_t, Y_t)
          z = self._sample_with_reparameterization(p_z_pos)
          y_mu, y_sigma = self.predict(z, X_t)
          log_likelihood = D.Normal(loc=y_mu, scale=y_sigma).log_prob(Y_t)
          log_likelihood = log_likelihood.sum(dim=-1).sum(dim=1) 
          log_prior = self.infer(None, None).log_prob(z).sum(dim=1)
          log_pos = self.infer(X_t, Y_t).log_prob(z).sum(dim=1)
        ls.append(log_prior + log_likelihood - log_pos)
      else:
        if self.has_context_set:
          p_z_pri = self.infer(X_c, Y_c)
          z = self._sample_with_reparameterization(p_z_pri)
          y_mu, y_sigma = self.predict(z, X_t)
          log_likelihood = D.Normal(loc=y_mu, scale=y_sigma).log_prob(Y_t)
          log_likelihood = log_likelihood.sum(dim=-1).sum(dim=1) 
        else:
          p_z_pri = self.infer(None, None)
          z = self._sample_with_reparameterization(p_z_pri)
          y_mu, y_sigma = self.predict(z, X_t)
          log_likelihood = D.Normal(loc=y_mu, scale=y_sigma).log_prob(Y_t)
          log_likelihood = log_likelihood.sum(dim=-1).sum(dim=1) 
        ls.append(log_likelihood)
    return (torch.logsumexp(torch.stack(ls, dim=0), dim=0) - math.log(n_latent_samples)) / Y_t.shape[1] 
  
  def compute_loss_and_metrics(self, X_t, Y_t, X_c, Y_c):
    
    """
    Compute loss for optimization, and metrics for monitoring.
    """
      
    if self.use_fourier_features:
      X_c = self.fourier_features(X_c)
      X_t = self.fourier_features(X_t)
      
    if self.training_objective == 'elbo':
      elbo, log = self.elbo(X_t, Y_t, X_c, Y_c)
    
      loss = - elbo.mean()
      logs = {
        "loss": loss,
        "elbo": elbo,
        "nll": log['nll'],
        "kld": log['kld'],
      }
        
    elif self.training_objective == 'mll':
      mll = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=False)
      loss = - mll.mean()  
      logs = {
        "mll": mll,
        "loss": loss,
      }
    elif self.training_objective == 'mll_is':
      mll = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=True)
      loss = - mll.mean()  
      logs = {
        "mll": mll,
        "loss": loss,
      }
    return loss, logs
  
  def evaluate(self, X_t, Y_t, X_c=None, Y_c=None):
    """
    Evaluate the model by estimating the marginal_log_likelihood.
    """
    if self.use_fourier_features:
      if X_c is not None:
        X_c = self.fourier_features(X_c)
      X_t = self.fourier_features(X_t)
    mll = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=50, use_importance_sampling=False)
    mll_is = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=50, use_importance_sampling=True)
    return {
      "marginal_log_likelihood": mll.mean(),
      "marginal_log_likelihood (Importance Sampling)": mll_is.mean(),
    }
  
  # def sample(self, X_t, X_c, Y_c):
  #   update_dim_vars_len({'t': X_t.shape[1]})
  #   _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
  #   mu, sigma = self.forward(X_t, X_c, Y_c)
  #   return mu
  
  def _visualize_1d(self, X_t, Y_t, X_c, Y_c, save_to=None):
    n_unique_samples = X_t.shape[0]
    n_context_points = [3, 5, 10, 20, 40] # [3, 5, 10, 15, 20]
    n_cols = len(n_context_points)
    n_rows = n_unique_samples
    n_samples = 5
    
    fig = plt.figure(figsize=(5*n_cols, 2.5*n_rows))
    for col in range(n_cols):
      _X_c = X_c[:, :n_context_points[col]].clone()
      _Y_c = Y_c[:, :n_context_points[col]].clone()
      X_t_ff, X_c_ff, Y_c_ff = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)
      if self.use_fourier_features:
        X_c_ff = self.fourier_features(X_c_ff)
        X_t_ff = self.fourier_features(X_t_ff)
      mus, sigmas = [], []
      for _ in range(n_samples):
        mu, sigma = self.forward(X_t_ff, X_c_ff, Y_c_ff)
        mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
        mus.append(mu)
        sigmas.append(sigma)
      for row in range(n_rows):
        index = row * n_cols + col 
        ax = fig.add_subplot(n_rows, n_cols, index+1)
        ax = sns.scatterplot(x=_X_c[row, :, 0], y=_Y_c[row, :, 0], ax=ax, facecolor="black", marker='+')
        ax = sns.lineplot(x=X_t[row, :, 0], y=Y_t[row, :, 0], ax=ax, color="forestgreen")
        for k in range(n_samples):
          # Plot predictive means for the target points
          ax = sns.lineplot(x=X_t[row, :, 0], y=mus[k][row, :, 0], ax=ax, color="royalblue", alpha=0.5)
          # Plot predictive std for the target points
          ax.fill_between(x=X_t[row, :, 0], y1=mus[k][row, :, 0]-sigmas[k][row, :, 0], y2=mus[k][row, :, 0]+sigmas[k][row, :, 0], alpha=0.1, color="royalblue")
        
    fig.tight_layout()
    if save_to is not None:
      fig.savefig(save_to)
    else:
      return fig
    
    
  def _visualize_2d(self, X_t, Y_t, X_c, Y_c, 
                    save_to=None, 
                    resolution=128, 
                    mode='mean_std'):
    
    assert mode in ['mean_std', 'sample', 'ar_sample'], "mode should be in ['mean_std', 'sample', 'ar_sample'] but got {}".format(mode)
    
    n_unique_samples = X_t.shape[0]
    n_context_points = [10, 30, 50, 100, 150, 200, 512]
    n_cols = len(n_context_points) + 1
    n_rows = n_unique_samples
    
    fig_mean = plt.figure(figsize=(3*n_cols, 3*n_rows))
    fig_std = plt.figure(figsize=(3*n_cols, 3*n_rows))
    
    for col in range(n_cols):
      if col > 0:
        logging.info("Number of Context Points {}".format(n_context_points[col-1]))
        _X_c = X_c[:, :n_context_points[col-1]].clone()
        _Y_c = Y_c[:, :n_context_points[col-1]].clone()
        X_t_ff, X_c_ff, Y_c_ff = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)
        if self.use_fourier_features:
          X_c_ff = self.fourier_features(X_c_ff)
          X_t_ff = self.fourier_features(X_t_ff)
        mus, sigmas = [], []
        prior = self.infer(X_c_ff, Y_c_ff)
        z = self._sample_with_reparameterization(prior)
        chunk_size = resolution
        if mode == 'ar_sample':
          chunk_size = 1
        _X_c = (_X_c + 1.) / 2. * (resolution - 1)
        for j in tqdm(range(X_t_ff.shape[1] // chunk_size)):
          mu, sigma = self.predict(z, X_t_ff[:, j*chunk_size:(j+1)*chunk_size].clone())
          if mode == 'mean_std':
            pass
          elif mode == 'sample':
            mu = torch.normal(mu,  sigma)
          elif mode == 'ar_sample':
            prior = self.infer(X_c_ff, Y_c_ff)
            z = self._sample_with_reparameterization(prior)
            X_t_ff_sample = X_t_ff[:, j*chunk_size:(j+1)*chunk_size].clone()
            Y_t_ff_sample = torch.normal(*self.predict(z, X_t_ff_sample))
            X_t_ff_sample, Y_t_ff_sample = X_t_ff_sample.detach(), Y_t_ff_sample.detach()
            X_c_ff, Y_c_ff = torch.cat([X_c_ff, X_t_ff_sample], dim=1), torch.cat([Y_c_ff, Y_t_ff_sample], dim=1)
            sigma = mu = Y_t_ff_sample
          mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
          mus.append(mu)
          sigmas.append(sigma)

        mu = np.reshape(np.concatenate(mus, axis=1), (-1, resolution, resolution))
        sigma = np.reshape(np.concatenate(sigmas, axis=1), (-1, resolution, resolution))
      else:
        mu = Y_t.cpu().detach().numpy().reshape((-1, resolution, resolution))
        sigma = mu 
  
      for row in range(n_rows):
        index = row * n_cols + col 
        img = mu[row]
        ax = fig_mean.add_subplot(n_rows, n_cols, index + 1)
        ax.imshow(img, cmap='Greys',  interpolation='nearest', vmin=-1, vmax=1)
        if col > 0:
          Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] < 0.0)[..., 0])
          ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
          Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] >= 0.0)[..., 0])
          ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        img = 1. - sigma[row]
        ax = fig_std.add_subplot(n_rows, n_cols, index + 1)
        ax.imshow(img, cmap='Greys',  interpolation='nearest', vmin=-1, vmax=1)
        if col > 0:
          Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] < 0.0)[..., 0])
          ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
          Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] >= 0.0)[..., 0])
          ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig_mean.tight_layout()
    fig_std.tight_layout()
    if save_to is not None:
      if mode == 'mean_std':
        fig_mean.savefig(save_to + "_mean.png")
        fig_std.savefig(save_to + "_std.png")
      elif mode == 'sample':
        fig_mean.savefig(save_to + "_sample.png")
      elif mode == 'ar_sample':
        fig_mean.savefig(save_to + "_ar_sample.png")
    else:
      return fig_mean, fig_std
    
  def visualize(self, X_t, Y_t, X_c, Y_c, save_to=None, resolution=None):
    if self.ori_x_dim == 1:
      return self._visualize_1d(X_t, Y_t, X_c, Y_c, save_to)
    elif self.ori_x_dim == 2:
      self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to, resolution, 'mean')
      self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to, resolution, 'sample')
      # self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to, resolution, 'ar_sample')
      return 
    else:
      raise NotImplementedError


  def evaluate_masks(self, X_t, Y_t, X_c=None, Y_c=None):
    """
    Evaluate the model by estimating the marginal_log_likelihood with different types of masks.
    """

    batch_size = X_t.shape[0]
    self.batch_size = 1
    ori_X_c, ori_X_t, ori_Y_c, ori_Y_t = X_c, X_t, Y_c, Y_t
    all_mll = []
    all_mll_is = []
    for idx in range(batch_size):
      X_c, X_t, Y_c, Y_t = ori_X_c[idx:idx+1], ori_X_t[idx:idx+1], ori_Y_c[idx:idx+1], ori_Y_t[idx:idx+1]
      resolution = 64
      n_ori_context_points = X_c.shape[1]
      n_unique_samples = X_t.shape[0]
      n_context_points = 160 # [20, 40, 80, 160, "top", "left", 'center']
      n_rows = n_unique_samples

      indices = torch.randperm(X_t.shape[1]).to(device=X_c.device)
      X_c = torch.index_select(X_t, dim=1, index=indices)
      Y_c = torch.index_select(Y_t, dim=1, index=indices)
      
      if isinstance(n_context_points, int):
        logging.info("Number of Context Points {}".format(n_context_points))
        _X_c = X_c[:, :n_context_points].clone()
        _Y_c = Y_c[:, :n_context_points].clone()
      elif isinstance(n_context_points, str):
        logging.info("Regional Context: {}".format(n_context_points))
        _X_t = X_t.reshape((n_unique_samples, resolution, resolution, self.ori_x_dim))
        _Y_t = Y_t.reshape((n_unique_samples, resolution, resolution, self.y_dim))
        nc = resolution // 2 
        if n_context_points == 'top':
          _X_c = _X_t[:, :nc]
          _Y_c = _Y_t[:, :nc]
        elif n_context_points == 'bottom':
          _X_c = _X_t[:, nc:]
          _Y_c = _Y_t[:, nc:]
        elif n_context_points == 'left':
          _X_c = _X_t[:, :, :nc]
          _Y_c = _Y_t[:, :, :nc]
        elif n_context_points == 'right':
          _X_c = _X_t[:, :, nc:]
          _Y_c = _Y_t[:, :, nc:]
        elif n_context_points == 'center':
          _X_c = _X_t[:, nc//2:-nc//2, nc//2:-nc//2]
          _Y_c = _Y_t[:, nc//2:-nc//2, nc//2:-nc//2]
        _X_c = _X_c.reshape((n_unique_samples, -1, self.ori_x_dim)).clone()
        _Y_c = _Y_c.reshape((n_unique_samples, -1, self.y_dim)).clone()

      X_t_ff, X_c_ff, Y_c = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)

      n_target_points = torch.randint(low=1, high=256 + 2 - n_ori_context_points, size=())
      indices = torch.randperm(X_t_ff.shape[1])[:n_target_points].to(device=X_t_ff.device)
      X_t_ff = torch.index_select(X_t_ff, dim=1, index=indices)
      Y_t = torch.index_select(Y_t, dim=1, index=indices)

      if self.use_fourier_features:
        X_c_ff = self.fourier_features(X_c_ff)
        X_t_ff = self.fourier_features(X_t_ff)

      mll = self.mll(X_t_ff, Y_t, X_c_ff, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=False)
      mll_is = self.mll(X_t_ff, Y_t, X_c_ff, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=True)

      all_mll.append(mll)
      all_mll_is.append(mll_is)

    mll = torch.cat(all_mll, dim=0)
    mll_is = torch.cat(all_mll_is, dim=0)
    print(torch.mean(mll_is), torch.std(mll_is))

    return {
      "marginal_log_likelihood": mll.mean(),
      "marginal_log_likelihood (Importance Sampling)": mll_is.mean(),
    }
    
  

  
  
