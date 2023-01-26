import os
import sys
from collections import OrderedDict
import time 
from tqdm import tqdm

from sklearn.covariance import log_likelihood
from absl import logging
from fflow.nets.flow import ConditionalRationalQuadraticTransformation
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
import pytorch_lightning as pl
from nflows.distributions.normal import StandardNormal
from .base import FunctionFlowModule
import typing
from tsalib import dim_vars as dvs, get_dim_vars, update_dim_vars_len
from tsalib import warp
from pytorch_lightning.profiler import BaseProfiler
from ..nets import (
  MLP, 
  SetTransformerEncoder, 
  DeepSetsEncoder,
  ConditionalLinearTransformation, 
  ConditionalRationalQuadraticTransformation, 
  SetFourierFeatures
)

import math
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('off')
from ..utils import batch_image_display

_B, _S, _C, _T, _L = dvs('Batch(b):10 SetSize(s):1 Context(c):1 Target(t):1 SeqLength(l):1', exists_ok=True)
_X, _Y, _Z, _R = dvs('DimX(x):1 DimY(y):1 DimZ(z):1 DimR(r):1', exists_ok=True)


class ConditionalSetCIF(pl.LightningModule, torch.nn.Module):
  
  """
  Continuously Indexed Flows (CIF) for sets of input-output pairs. Given 
  a sequence of latent variables, the output y of each datapoint evolves independently
  conditioning on the latent variables and the corresponding input x.
  
  [1] Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows. 
      (https://arxiv.org/abs/1909.13833)
  [2] Variational inference with continuously-indexed normalizing flows.
      (https://proceedings.mlr.press/v161/caterini21a.html)
      
  """
  
  def __init__(self,
               n_steps: int, 
               x_dim: int,
               y_dim: int, 
               r_dim: int, 
               z_dim: int,
               batch_size: int,
               base_flow: str ='linear',
               n_hidden_units: int = 16,
               n_hidden_layers: int = 3,
               has_context_set: bool = False):
    
    """
    Args:
    ---------------------------------------------------------
    n_steps: int
        Number of functional Markov chain steps
    
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
        
    base_flow: str, optional
        Flow types, currently there are two options ['linear', 'rational_quadratic_spline']
        'linear': Linear flow, used to demonstrate that when the flow is linear, and there
                  is only one step, the model is roughly equivalent to neural processes.
        'rational_quadratic_spline'[1]: Use reural spline flows.
        
    n_hidden_units: int, optional
        Number of hidden units in MLPs
        
    n_hidden_layers: int, optional
        Number of MLP layers in the flows
        
        
    [1] Neural Spline Flows (https://arxiv.org/abs/1906.04032)
    """
    
    super().__init__()
    self.n_steps = n_steps
    self.y_dim = y_dim 
    self.x_dim = x_dim 
    self.r_dim = r_dim 
    self.z_dim = z_dim 
    self.n_hidden_units = n_hidden_units
    self.n_hidden_layers = n_hidden_layers
    self.batch_size = batch_size
    self.base_flow = base_flow
    self.has_context_set = has_context_set
    
    # point-wise flows
    self.flows = []
    for t in range(n_steps):
      if base_flow == 'linear':
        flow = ConditionalLinearTransformation(x_dim=self.x_dim, 
                                               y_dim=self.y_dim, 
                                               c_dim=self.z_dim + self.x_dim,
                                               hidden_units=self.n_hidden_units, 
                                               n_hidden_layers=self.n_hidden_layers)
      elif base_flow == 'rational_quadratic_spline':
        flow = ConditionalRationalQuadraticTransformation(x_dim=self.x_dim, 
                                                          y_dim=self.y_dim, 
                                                          c_dim=self.z_dim + self.x_dim,
                                                          n_bins=10,
                                                          hidden_units=self.n_hidden_units,
                                                          n_hidden_layers=self.n_hidden_layers)
      self.flows.append(flow)
    self.flows = torch.nn.ModuleList(self.flows)
    self.q0 = StandardNormal([self.y_dim])
    
    update_dim_vars_len({'b': self.batch_size,
                         'l': self.n_steps,
                         'x': self.x_dim, 
                         'y': self.y_dim, 
                         'r': self.r_dim,
                         'z': self.z_dim})
    
  def log_prob(self, 
               z_seq: torch.Tensor, 
               X_t: torch.Tensor, 
               Y_t: torch.Tensor,
               aggregate=True):
    """
    log p(Y_t | z_seq, X_t, X_c, Y_c)
    
    Args:
    ---------------------------------------------------
    z_seq: torch.Tensor 
        Sequence of latent variables, tensor fo shape [seq_length, batch_size, z_dim]
    
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
    
    Y_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    Return:
    ---------------------------------------------------
    log_prob: [Batch_Size,]
    
    """
    update_dim_vars_len({'t': X_t.shape[1], 'b': X_t.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    
    z_seq = warp(z_seq, 's,b,z -> s,b,1,z -> s,b,t,z -> s,b*t,z', 'arv')
    Y_t = warp(Y_t, 'b,t,y -> b*t,y', 's')
    X_t = warp(X_t, 'b,t,x -> b*t,x', 's')
    #
    Y: 'b*t,y' = Y_t
    
    logabsdet_seq = []
    for t in range(self.n_steps-1, -1, -1):
      context: 'b*t,x+z' = torch.cat([X_t, z_seq[t]], dim=-1)
      Y, logabsdet = self.flows[t].jacobian_determinant(Y, 
                                  context=context)
      logabsdet_seq.append(-logabsdet)
    #
    log_prob: 'b*t' = self.q0.log_prob(Y)
    log_prob: 'b*t' = torch.stack(logabsdet_seq + [log_prob]).sum(dim=0)
    if aggregate:
      log_prob: 'b' = log_prob.reshape([self.batch_size, -1]).sum(dim=1)
    else:
      log_prob: 'b,t' = log_prob.reshape([self.batch_size, -1])
    return log_prob
  
  def sample(self, 
             z_seq: torch.Tensor, 
             X_t: torch.Tensor, 
             return_sample_path: bool = False):
    """
    Args:
    ---------------------------------------------------
    z_seq: torch.Tensor 
        Sequence of latent variables, tensor fo shape [Seq_Length, Batch_Size, Z_Dim]
    
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
        
    return_sample_path: bool, optional
        Whether to return the whole sequence of sampled y at each step.
        
    Returns:
    ---------------------------------------------------
    Y_t_pred: torch.Tensor
        Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
        
    """
    update_dim_vars_len({'t': X_t.shape[1], 'b': X_t.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    
    z_seq = warp(z_seq, 's,b,z -> s,b,1,z -> s,b,t,z -> s,b*t,z', 'arv')
    X_t = warp(X_t, 'b,t,x -> b*t,x', 'v')
    #
    Ys = []
    Y_t = self.q0.sample(num_samples=X_t.shape[0])
    Ys.append(Y_t)
    for t in range(self.n_steps):
      context = torch.cat([X_t, z_seq[t]], dim=-1)
      Y_t, _ = self.flows[t](Y_t, context=context)
      Ys.append(Y_t)
    Y_t = warp(Y_t, 'b*t,y -> b,t,y', 'v')
    if return_sample_path:
      return warp(torch.stack(Ys), 'l+1,b*t,y -> l+1,b,t,y', 'v')
    else:
      return Y_t
    
  def sample_mean_std(self, 
             z_seq: torch.Tensor, 
             X_t: torch.Tensor,
             return_sample_path: False):
    """
    This method is for comparing our model with neural processes.
    When n_step=1 and the base_flow type is linear, this function
    will give predictive mean and std just as the original neural 
    process model.
    
    When base_flow type is not linear, this function still gives std, 
    but it is just Flows(Y_mean+Y_std) - Flows(Y_mean).
    
    Args:
    ---------------------------------------------------
    z_seq: torch.Tensor 
        Sequence of latent variables, tensor fo shape [Seq_Length, Batch_Size, Z_Dim]
    
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
        
    return_sample_path: bool, optional
        Whether to return the whole sequence of sampled y at each step.
        
    Returns:
    ---------------------------------------------------
    Mean: torch.Tensor
        Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    Std: torch.Tensor
        Tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
        
    """
    update_dim_vars_len({'t': X_t.shape[1], 'b': X_t.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
      
    z_seq = warp(z_seq, 's,b,z -> s,b,1,z -> s,b,t,z -> s,b*t,z', 'arv')
    X_t = warp(X_t, 'b,t,x -> b*t,x', 'v')
    
    _Y_t = self.q0.sample(num_samples=X_t.shape[0])
    Y_t = torch.zeros_like(_Y_t, device=_Y_t.device)
    Y_t_mean_steps = []
    Y_t_mean_steps.append(warp(Y_t, 'b*t,y -> b,t,y', 'v'))
    for t in range(self.n_steps):
      context = torch.cat([X_t, z_seq[t]], dim=-1)
      Y_t, _ = self.flows[t](Y_t, context=context)
      Y_t_mean_steps.append(warp(Y_t, 'b*t,y -> b,t,y', 'v'))
    Y_t = warp(Y_t, 'b*t,y -> b,t,y', 'v')  
    Y_mean = Y_t
    
    Y_t = torch.ones_like(_Y_t, device=_Y_t.device)
    Y_t_std_steps = []
    Y_t_std_steps.append(warp(Y_t, 'b*t,y -> b,t,y', 'v') - Y_t_mean_steps[0])
    for t in range(self.n_steps):
      if self.has_context_set:
        context = torch.cat([X_t, z_seq[t]], dim=-1)
      Y_t, _ = self.flows[t](Y_t, context=context)
      Y_t_std_steps.append(warp(Y_t, 'b*t,y -> b,t,y', 'v') - Y_t_mean_steps[t+1])
    Y_t = warp(Y_t, 'b*t,y -> b,t,y', 'v')  
    Y_std = Y_t - Y_mean
    if return_sample_path:
      return torch.stack(Y_t_mean_steps), torch.stack(Y_t_std_steps)
    else:
      return Y_mean, Y_std

class PermutationInvariantInferenceNetwork(pl.LightningModule, torch.nn.Module):
  
  """
  Inference network that is permutation invariant the order of the input datapoints.
  
  It takes in a set of (x, y) pairs and produces a sequence of Gaussian distributions 
  over latent variables.
  
  """
  
  def __init__(self,
               n_steps: int, 
               x_dim: int,
               y_dim: int, 
               r_dim: int, 
               z_dim: int,
               batch_size: int,
               flows: ConditionalSetCIF,
               set_encoder_type: str = 'set_transformer',
               n_hidden_units: int = 16, 
               n_hidden_layers: int = 3,  
               n_inds: int = 0,
               has_context_set: bool = False):
    super().__init__()
    self.n_steps = n_steps 
    self.x_dim = x_dim
    self.y_dim = y_dim 
    self.r_dim = r_dim
    self.z_dim = z_dim
    self.n_hidden_layers = n_hidden_layers
    self.n_inds = n_inds 
    self.n_hidden_units = n_hidden_units
    self.batch_size = batch_size 
    self.flows = flows 
    self.has_context_set = has_context_set 
    self.set_encoder_type = set_encoder_type 
    # Set Encoder
    # The set encoder which produces a permutation invariant representation
    # by taking in a set of input-output pairs. It is shared by all steps.
    
    if self.set_encoder_type == 'set_transformer':
      self.set_encoder = SetTransformerEncoder(dim_input=self.x_dim+self.y_dim,
                                                dim_output=self.r_dim,
                                                dim_hidden=self.n_hidden_units,
                                                n_sabs=self.n_hidden_layers,
                                                n_inds=self.n_inds,
                                                aggregation=True).to(self.device)
    elif self.set_encoder_type == 'deep_sets':
      self.set_encoder = DeepSetsEncoder(dim_input=self.x_dim+self.y_dim,
                                          dim_output=self.r_dim,
                                          dim_hidden=self.n_hidden_units,
                                          n_layers=self.n_hidden_layers,
                                          aggregation=True).to(self.device)
    else:
      raise NotImplementedError
    
    self._mu_sigma_mlps = []
    for _ in range(self.n_steps):
      self._mu_sigma_mlps.append(MLP(z_dim+r_dim, [2 * z_dim]).to(self.device))
      # self._mu_sigma_mlps.append(MLP(z_dim+r_dim, [self.n_hidden_units] * self.n_hidden_layers + [2 * z_dim]).to(self.device))
    self._mu_sigma_mlps = torch.nn.ModuleList(self._mu_sigma_mlps)
    
    update_dim_vars_len({'b': self.batch_size,
                         'l': self.n_steps,
                         'x': self.x_dim, 
                         'y': self.y_dim, 
                         'r': self.r_dim,
                         'z': self.z_dim})
            
  
  def log_prob(self, 
               z_seq: torch.Tensor, 
               X_set: torch.Tensor, 
               Y_set: torch.Tensor):
    """
    Args:
    ---------------------------------------------------
    z_seq: torch.Tensor 
        Sequence of latent variables, tensor fo shape [Seq_Length, Batch_Size, Z_Dim]
    
    X_set: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Points, X_Dim]
    
    Y_set: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Points, Y_Dim]
        
    Returns:
    ---------------------------------------------------
      log_prob: [Batch_Size,]
    """
    update_dim_vars_len({'s': X_set.shape[1], 'b': X_set.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    
    z_prev: 'b,z' = torch.zeros([_B, _Z], device=self.device)
    log_prob: 'b' = torch.zeros([_B,], device=self.device)
    Y = Y_set
    p_seq = [] # Sequence of Gaussians over latent variables with length n_steps
    for i in range(self.n_steps):
      # reverse the order, infer the latent variables at last step first
      t = self.n_steps-i-1 
      # produce set representation using inputs, and the ys from previous step.
      r: 'b,r' = self.set_encoder(torch.cat([X_set, Y], dim=-1))
      # condition on z from previous step and the set representation
      cond: 'b,z+r' = torch.cat([z_prev, r], dim=-1)
      mu: 'b,z' = self._mu_sigma_mlps[t](cond)[:, :self.z_dim] # self._mlp_mus[t](cond)
      sigma: 'b,z' = 0.95 * torch.sigmoid(self._mu_sigma_mlps[t](cond)[:, self.z_dim:]) + 0.05
      z_prev: 'b,z' = z_seq[t]
      log_prob += Normal(loc=mu, scale=sigma).log_prob(z_prev).sum(dim=-1)
      p_seq.append(Normal(loc=mu, scale=sigma))
      # 
      z = warp(z_prev, 'b,z -> b,1,z -> b,s,z -> b*s,z', 'arv')
      context = torch.cat([warp(X_set, 'b,s,x -> b*s,x', 's'), z], dim=-1)
      # Sharing flows in the generative models just as in continuous normalizing flows
      Y: 'b*s,y' = self.flows[t].jacobian_determinant(warp(Y, 'b,s,y -> b*s,y', 's'), context=context)[0]
      Y = warp(Y, 'b*s,y -> b,s,y', 's')
    p_seq.reverse()
    return log_prob, p_seq
  
  
  def sample(self, 
             X_set: torch.Tensor, 
             Y_set: torch.Tensor):
    """
    Sample the sequential latent variables.
    
    Args:
    -------------------------------------------------------
    X_set: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Points, X_Dim]
    
    Y_set: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Points, Y_Dim]
        
    Returns:
    -------------------------------------------------------
        z_seq: [Seq_Length, Batch_Size, Z_Dim]
      
    """
    update_dim_vars_len({'s': X_set.shape[1], 'b': X_set.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    
    z_prev = torch.zeros([_B, _Z], device=self.device)
    Y = Y_set
    zs = []
    for i in range(self.n_steps):
      t = self.n_steps-i-1
      r: 'b,r' = self.set_encoder(torch.cat([X_set, Y], dim=-1))
      cond: 'b,z+r' = torch.cat([z_prev, r], dim=-1)
      mu: 'b,z' = self._mu_sigma_mlps[t](cond)[:, :self.z_dim]
      sigma: 'b,z' = 0.95 * torch.sigmoid(self._mu_sigma_mlps[t](cond)[:, self.z_dim:]) + 0.05
      eps = torch.normal(mean=0., std=1., size=mu.shape, device=self.device)
      z = eps * sigma + mu 
      zs.append(z)
      z_prev = z
      z = warp(z, 'b,z -> b,1,z -> b,s,z -> b*s,z', 'arv')
      context = torch.cat([warp(X_set, 'b,s,x -> b*s,x', 's'), z], dim=-1)
      Y: 'b*s,y' = self.flows[t].jacobian_determinant(warp(Y, 'b,s,y -> b*s,y', 's'), context=context)[0]
      Y = warp(Y, 'b*s,y -> b,s,y', 's')
    zs: 's,b,z' = torch.stack(zs, dim=0).flip(dims=[0])
    return zs

class DiscreteFunctionFlowModule(FunctionFlowModule):
  
  def __init__(self, 
               n_steps: int, 
               x_dim: int,
               y_dim: int, 
               r_dim: int, 
               z_dim: int,
               batch_size: int,
               base_flow: str = 'linear',
               set_encoder_type: str = 'set_transformer',
               n_hidden_units: int = 16, 
               n_hidden_units_cif: int = 16, 
               n_hidden_layers: int = 3,        
               n_inds: int = 0,  
               has_context_set: bool = False,    
               use_fourier_features: bool = False,
               num_frequencies: int = 20,
               rff_init_std: float = 1.0,
               training_objective: str = 'elbo', 
               n_latent_samples: int = 20,
               optim_lr: float = 0.0001, 
               weight_decay: float=0):
    
    """
    Neural Functional Markov Chain.  
    
    Args:
    ---------------------------------------------------------
    n_steps: int
        Number of functional Markov chain steps
        
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
        
    base_flow: str, optional
        Flow types, currently there are two options ['linear', 'rational_quadratic_spline']
        'linear': Linear flow, used to demonstrate that when the flow is linear, and there
                  is only one step, the model is roughly equivalent to neural processes.
        'rational_quadratic_spline'[1]: Use reural spline flows.
        
    n_hidden_units: int, optional
        Number of hidden units in MLPs
        
    n_hidden_units_cif: int, optional
        Number of hidden units in MLPs in the CIF module
        
    n_hidden_layers: int, optional
        Number of MLP layers in the flows
        
    optim_lr: float, optional
        Learning rate of optimizers.
        
    profiler: BaseProfiler, optional
        Tools to profile your training/testing/inference run can help you identify 
        bottlenecks in your code.
    """
    
    super().__init__(has_context_set=has_context_set,
                     optim_lr=optim_lr, 
                     weight_decay=weight_decay)
    self.n_steps = n_steps
    self.y_dim = y_dim
    self.x_dim = x_dim
    self.ori_x_dim = x_dim
    self.r_dim = r_dim 
    self.z_dim = z_dim
    
    self.base_flow  = base_flow 
    self.n_hidden_layers = n_hidden_layers
    self.n_hidden_units = n_hidden_units
    self.n_hidden_units_cif = n_hidden_units_cif
    self.batch_size = batch_size 
    self.n_inds = n_inds 
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
    
    # generative model
    self.cond_cif = ConditionalSetCIF(n_steps=self.n_steps, 
                                      y_dim=self.y_dim, 
                                      x_dim=self.x_dim, 
                                      r_dim=self.r_dim, 
                                      z_dim=self.z_dim, 
                                      batch_size=self.batch_size,
                                      base_flow=self.base_flow,
                                      n_hidden_units=self.n_hidden_units_cif, 
                                      n_hidden_layers=self.n_hidden_layers,
                                      has_context_set=self.has_context_set)
    # inference model, for both the conditional prior and posterior
    self.infer_net = PermutationInvariantInferenceNetwork(n_steps=self.n_steps, 
                                                        y_dim=self.y_dim, 
                                                        x_dim=self.x_dim, 
                                                        r_dim=self.r_dim, 
                                                        z_dim=self.z_dim, 
                                                        batch_size=self.batch_size, 
                                                        flows=self.cond_cif.flows,
                                                        set_encoder_type=self.set_encoder_type,
                                                        n_hidden_units=self.n_hidden_units, 
                                                        n_hidden_layers=self.n_hidden_layers,
                                                        n_inds=self.n_inds,
                                                        has_context_set=self.has_context_set)    
      
    self.params = self.parameters()
    
    update_dim_vars_len({'b': self.batch_size,
                         'l': self.n_steps,
                         'x': self.x_dim, 
                         'y': self.y_dim, 
                         'r': self.r_dim,
                         'z': self.z_dim})
    
    logging.debug("---------       Discrete Function Flow       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
    

  def elbo(self,
           X_t: torch.Tensor, 
           Y_t: torch.Tensor, 
           X_c: typing.Union[torch.Tensor, None] = None, 
           Y_c: typing.Union[torch.Tensor, None] = None):
    """
    Computing ELBO using the context and the target set
    
    Args:
    ---------------------------------------------------------------
    X_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, X_Dim]
    
    Y_t: torch.Tensor
        Target inputs, tensor of shape [Batch_Size, Num_Target_Points, Y_Dim]
    
    X_c: torch.Tensor, optional
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, X_Dim]
    
    Y_c: torch.Tensor, optional
        Context inputs, tensor of shape [Batch_Size, Num_Context_Points, Y_Dim]
        
    Return:
    ---------------------------------------------------------------
    ELBO: Tensor of shape [Batch_Size,]
    """
      
    if self.has_context_set:
      X_ct = torch.cat([X_c, X_t], dim=1)
      Y_ct = torch.cat([Y_c, Y_t], dim=1)
      # sample form posterior
      z_seq = self.infer_net.sample(X_ct, Y_ct)
      # likelihood
      log_likelihood = self.cond_cif.log_prob(z_seq, X_t, Y_t) / Y_t.shape[1]    
      # compute the sequential prior and posterior
      _, prior_seq = self.infer_net.log_prob(z_seq, X_c, Y_c)
      _, pos_seq = self.infer_net.log_prob(z_seq, X_ct, Y_ct)
    else:
      z_seq = self.infer_net.sample(X_t, Y_t)
      log_likelihood = self.cond_cif.log_prob(z_seq, X_t, Y_t) / Y_t.shape[1]    
      _, prior_seq = self.infer_net.log_prob(z_seq)
      _, pos_seq = self.infer_net.log_prob(z_seq, X_t, Y_t)
      
    kld = [torch.distributions.kl.kl_divergence(pos_seq[i], prior_seq[i]).sum(dim=1) / Y_t.shape[1] for i in range(self.n_steps)]
    kld = torch.stack(kld, dim=-1).sum(dim=-1)
    elbo = log_likelihood - kld 
    log = {'log_likelihood': log_likelihood, 'kl': kld, 'elbo': elbo}
    return elbo, log
  
  def mll(self, 
          X_t: torch.Tensor, 
          Y_t: torch.Tensor, 
          X_c: typing.Union[torch.Tensor, None] = None, 
          Y_c: typing.Union[torch.Tensor, None] = None,
          n_latent_samples: int = 20,
          use_importance_sampling: bool = True):
    
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
          z_seq = self.infer_net.sample(X_ct, Y_ct)
          log_likelihood = self.cond_cif.log_prob(z_seq, X_t, Y_t)
          log_prior = self.infer_net.log_prob(z_seq, X_c, Y_c)[0]
          log_pos = self.infer_net.log_prob(z_seq, X_ct, Y_ct)[0]
        ls.append(log_prior + log_likelihood - log_pos)
      else:
        if self.has_context_set:
          z_seq = self.infer_net.sample(X_c, Y_c)
          log_likelihood = self.cond_cif.log_prob(z_seq, X_t, Y_t)
        ls.append(log_likelihood)
    return (torch.logsumexp(torch.stack(ls, dim=0), dim=0) - math.log(n_latent_samples)) / Y_t.shape[1] 
  
  def predict(self, z_seq, X_t, return_sample_path=False):
    y_mu, y_sigma = self.cond_cif.sample_mean_std(z_seq, X_t, return_sample_path=return_sample_path)
    return y_mu, y_sigma
  
  def forward(self, 
              X_t, 
              X_c=None, 
              Y_c=None, 
              update=False,
              return_sample_path=False):
      
    update_dim_vars_len({'t': X_t.shape[1], 'c': X_c.shape[1], 'b': X_t.shape[0]})
    _B, _S, _C, _T, _L, _X, _Y, _Z, _R = get_dim_vars('b s c t l x y z r')
    if self.has_context_set:
      z_seq = self.infer_net.sample(X_c, Y_c)
    else:
      z_seq = self.infer_net.sample()
    y_mu, y_sigma = self.predict(z_seq, X_t, return_sample_path=return_sample_path)
    return y_mu, y_sigma 
          
  def sample(self, X_t, X_c=None, Y_c=None, return_sample_path=False):
      
    z_seq = self.infer_net.sample(X_c, Y_c)
    return self.cond_cif.sample(z_seq, X_t, return_sample_path=return_sample_path)
    
    
  def _visualize_1d(self, X_t, Y_t, X_c, Y_c, save_to=None):
    n_unique_samples = X_t.shape[0]
    n_context_points = [3, 5, 10, 20, 40] # [1, 2, 4, 8] # [3, 5, 10, 15, 20]
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
        ax = sns.lineplot(x=X_t[row, :, 0], y=Y_t[row, :, 0], ax=ax, color="forestgreen")
        for k in range(n_samples):
          # Plot predictive means for the target points
          ax = sns.lineplot(x=X_t[row, :, 0], y=mus[k][row, :, 0], ax=ax, color="royalblue", alpha=0.5)
          # Plot predictive std for the target points
          ax.fill_between(x=X_t[row, :, 0], y1=mus[k][row, :, 0]-sigmas[k][row, :, 0], y2=mus[k][row, :, 0]+sigmas[k][row, :, 0], alpha=0.1, color="royalblue")
        ax = sns.scatterplot(x=_X_c[row, :, 0], y=_Y_c[row, :, 0], ax=ax, facecolor="black", marker='+')
        
    fig.tight_layout()
    if save_to is not None:
      fig.savefig(save_to)
    else:
      return fig

  def _visualize_1d_iterations(self, X_t, Y_t, X_c, Y_c, save_to=None):
    n_unique_samples = self.n_steps + 1
    n_context_points = [2, 4, 8, 16]# [3, 5, 10, 15, 20] # [1, 2, 4, 8] # [3, 5, 10, 15, 20]
    n_cols = len(n_context_points)
    n_rows = n_unique_samples
    n_samples = 5

    X_t, Y_t, X_c, Y_c = X_t[0:1], Y_t[0:1], X_c[0:1], Y_c[0:1]
    
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    for col in range(n_cols):
      _X_c = X_c[:, :n_context_points[col]].clone()
      _Y_c = Y_c[:, :n_context_points[col]].clone()
      X_t_ff, X_c_ff, Y_c_ff = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)
      if self.use_fourier_features:
        X_c_ff = self.fourier_features(X_c_ff)
        X_t_ff = self.fourier_features(X_t_ff)
      mus, sigmas = [], []
      for _ in range(n_samples):
        mu, sigma = self.forward(X_t_ff, X_c_ff, Y_c_ff, return_sample_path=True)
        mu, sigma = mu[:, 0], sigma[:, 0]
        mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
        mus.append(mu)
        sigmas.append(sigma)

      X_t, Y_t, X_c, Y_c = X_t.repeat(n_unique_samples, 1, 1), Y_t.repeat(n_unique_samples, 1, 1), X_c.repeat(n_unique_samples, 1, 1), Y_c.repeat(n_unique_samples, 1, 1)
      _X_c, _Y_c = _X_c.repeat(n_unique_samples, 1, 1), _Y_c.repeat(n_unique_samples, 1, 1)

      for row in range(n_rows):
        index = row * n_cols + col 
        ax = fig.add_subplot(n_rows, n_cols, index+1)
        ax = sns.lineplot(x=X_t[row, :, 0], y=Y_t[row, :, 0], ax=ax, color="forestgreen")
        for k in range(n_samples):
          # Plot predictive means for the target points
          ax = sns.lineplot(x=X_t[row, :, 0], y=mus[k][row, :, 0], ax=ax, color="royalblue", alpha=0.5)
          # Plot predictive std for the target points
          ax.fill_between(x=X_t[row, :, 0], y1=mus[k][row, :, 0]-sigmas[k][row, :, 0], y2=mus[k][row, :, 0]+sigmas[k][row, :, 0], alpha=0.1, color="royalblue")
        ax = sns.scatterplot(x=_X_c[row, :, 0], y=_Y_c[row, :, 0], ax=ax, facecolor="black", marker='+', s=150)
        
    fig.tight_layout()
    if save_to is not None:
      fig.savefig(save_to)
    else:
      return fig


    # n_unique_samples = self.n_steps # X_t.shape[0]
    # n_context_points = [3, 5, 10, 15, 20] # [1, 2, 4, 8] # [3, 5, 10, 15, 20]
    # n_cols = len(n_context_points)
    # n_rows = n_unique_samples
    # n_samples = 5
    
    # fig = plt.figure(figsize=(3*n_cols, 3*n_rows))
    # for col in range(n_cols):
    #   _X_c = X_c[:, :n_context_points].clone()
    #   _Y_c = Y_c[:, :n_context_points].clone()
    #   X_t_ff, X_c_ff, Y_c_ff = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)
    #   if self.use_fourier_features:
    #     X_c_ff = self.fourier_features(X_c_ff)
    #     X_t_ff = self.fourier_features(X_t_ff)
    #   mus, sigmas = [], []
    #   for _ in range(n_samples):
    #     mu_steps, sigma_steps = self.forward(X_t_ff, X_c_ff, Y_c_ff, return_sample_path=True)
    #     mu, sigma = mu_steps[:, 0], sigma_steps[:, 0]
    #     mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
    #     mus.append(mu)
    #     sigmas.append(sigma)

    #   X_t_1, Y_t_1, X_c_1, Y_c_1 = X_t.repeat(self.n_steps, 1, 1), Y_t.repeat(self.n_steps, 1, 1), X_c.repeat(self.n_steps, 1, 1), Y_c.repeat(self.n_steps, 1, 1)
    #   _Y_c_1, _X_c_1 = _X_c.repeat(self.n_steps, 1, 1), _Y_c.repeat(self.n_steps, 1, 1)

    #   for row in range(n_rows):
    #     index = row * n_cols + col 
    #     ax = fig.add_subplot(n_rows, n_cols, index+1)
    #     ax = sns.lineplot(x=X_t_1[row, :, 0], y=Y_t_1[row, :, 0], ax=ax, color="forestgreen")
    #     for k in range(n_samples):
    #       # Plot predictive means for the target points
    #       ax = sns.lineplot(x=X_t_1[row, :, 0], y=mus[k][row, :, 0], ax=ax, color="blueviolet", alpha=0.5)
    #       # Plot predictive std for the target points
    #       ax.fill_between(x=X_t_1[row, :, 0], y1=mus[k][row, :, 0]-sigmas[k][row, :, 0], y2=mus[k][row, :, 0]+sigmas[k][row, :, 0], alpha=0.1, color="blueviolet")
    #     ax = sns.scatterplot(x=_X_c_1[row, :, 0], y=_Y_c_1[row, :, 0], ax=ax, facecolor="black", marker='+')
        
    # fig.tight_layout()
    # if save_to is not None:
    #   fig.savefig(save_to)
    # else:
    #   return fig
  
    
  def _visualize_2d(self, X_t, Y_t, X_c, Y_c, 
                    save_to=None, 
                    resolution=128, 
                    mode='mean_std'):
    
    assert mode in ['mean_std', 'sample', 'ar_sample'], "mode should be in ['mean_std', 'sample', 'ar_sample'] but got {}".format(mode)
    
    n_unique_samples = X_t.shape[0]
    n_context_points = [5, 10, 15, 20, 'top', 'left'] # [20, 40, 80, 160, "top", "left", 'center'] # [10, 30, 50, 100, 150, 200, 512] # [10, 30, 50, 100, 150, 200]
    n_cols = len(n_context_points) + 1
    n_rows = n_unique_samples
    
    mean_vis = []
    std_vis = []
    
    for col in range(n_cols):
      if col > 0:
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

        X_t_ff, X_c_ff, Y_c_ff = X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device)
        if self.use_fourier_features:
          X_c_ff = self.fourier_features(X_c_ff)
          X_t_ff = self.fourier_features(X_t_ff)
        mus, sigmas = [], []
        z_seq = self.infer_net.sample(X_c_ff, Y_c_ff)

        chunk_size = resolution
        if mode == 'ar_sample':
          chunk_size = 1
        for j in tqdm(range(X_t_ff.shape[1] // chunk_size)):
          if mode == 'mean_std':
            mu, sigma = self.predict(z_seq, X_t_ff[:, j*chunk_size:(j+1)*chunk_size].clone())
          elif mode == 'sample':
            mu = self.cond_cif.sample(z_seq, X_t_ff[:, j*chunk_size:(j+1)*chunk_size].clone())
            sigma = mu 
          elif mode == 'ar_sample':
            z_seq = self.infer_net.sample(X_c_ff, Y_c_ff)
            X_t_ff_sample = X_t_ff[:, j*chunk_size:(j+1)*chunk_size].clone()
            Y_t_ff_sample = self.cond_cif.sample(z_seq, X_t_ff_sample)
            X_t_ff_sample, Y_t_ff_sample = X_t_ff_sample.detach(), Y_t_ff_sample.detach()
            X_c_ff, Y_c_ff = torch.cat([X_c_ff, X_t_ff_sample], dim=1), torch.cat([Y_c_ff, Y_t_ff_sample], dim=1)
            sigma = mu = Y_t_ff_sample
          mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
          mus.append(mu)
          sigmas.append(sigma)

        mu = np.reshape(np.concatenate(mus, axis=1), (-1, resolution, resolution, 1))
        mu = np.repeat(mu, 3, axis=-1)
        sigma = np.reshape(np.concatenate(sigmas, axis=1), (-1, resolution, resolution, 1))
        sigma = np.repeat(sigma, 3, axis=-1)

        X_c_np = np.round((_X_c.cpu().detach().numpy() + 1.) / 2. * (resolution-1)).astype(np.int)
        Y_c_np = _Y_c.cpu().detach().numpy()

        for k in range(n_unique_samples):
          ind_x, ind_y = X_c_np[k][:, 0], X_c_np[k][:, 1]
          mu[k, ind_x, ind_y] = Y_c_np[k] * np.array([[1, -0.5, -1.]])
      else:
        mu = Y_t.cpu().detach().numpy().reshape((-1, resolution, resolution, 1))
        mu = np.repeat(mu, 3, axis=-1)
        sigma = mu 

      mean_vis.append(mu)
      std_vis.append(sigma)
    mean_vis = -np.stack(mean_vis, axis=1)
    std_vis = np.stack(std_vis, axis=1)

    # mean_vis = np.repeat(mean_vis, 3, axis=-1)
    # std_vis = np.repeat(std_vis, 3, axis=-1)

    if save_to is not None:
      if mode == 'mean_std':
        batch_image_display(mean_vis, figname=save_to + "_mean.png")
        batch_image_display(std_vis, vmin=0., vmax=1., figname=save_to + "_std.png")
      elif mode == 'sample':
        batch_image_display(mean_vis, figname=save_to + "_sample.png")
      elif mode == 'ar_sample':
        batch_image_display(mean_vis, figname=save_to + "_ar_sample.png")
    

    #   for row in range(n_rows):
    #     index = row * n_cols + col 
    #     # img = np.clip(0.5 - mu[row], 0.0, 1.0)
    #     img = mu[row]
    #     ax = fig_mean.add_subplot(n_rows, n_cols, index + 1)
    #     ax.imshow(img, cmap='Greys',  interpolation='nearest', vmin=-1, vmax=1)
    #     if col > 0:
    #       Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] < 0.0)[..., 0])
    #       ax.scatter(x=Xs[:, 1], y=Xs[:, 0], marker=',', linewidths=0.0)
    #       Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] >= 0.0)[..., 0])
    #       ax.scatter(x=Xs[:, 1], y=Xs[:, 0], marker=',', linewidths=0.0)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
        
    #     img = 1. - sigma[row]
    #     ax = fig_std.add_subplot(n_rows, n_cols, index + 1)
    #     ax.imshow(img, cmap='Greys',  interpolation='nearest', vmin=-1, vmax=1)
    #     if col > 0:
    #       Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] < 0.0)[..., 0])
    #       ax.scatter(x=Xs[:, 1], y=Xs[:, 0], marker=',', linewidths=0.0)
    #       Xs = torch.index_select(_X_c[row], dim=0, index=torch.argwhere(_Y_c[row, :, 0] >= 0.0)[..., 0])
    #       ax.scatter(x=Xs[:, 1], y=Xs[:, 0], marker=',', linewidths=0.0)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    # fig_mean.tight_layout()
    # fig_std.tight_layout()

    # if save_to is not None:
    #   if mode == 'mean_std':
    #     fig_mean.savefig(save_to + "_mean.png")
    #     fig_std.savefig(save_to + "_std.png")
    #   elif mode == 'sample':
    #     fig_mean.savefig(save_to + "_sample.png")
    #   elif mode == 'ar_sample':
    #     fig_mean.savefig(save_to + "_ar_sample.png")
    # else:
    #   return fig_mean, fig_std
    
  def visualize(self, X_t, Y_t, X_c, Y_c, save_to=None, resolution=None):
    if self.ori_x_dim == 1:
      #return self._visualize_1d_iterations(X_t, Y_t, X_c, Y_c, save_to)
      return self._visualize_1d(X_t, Y_t, X_c, Y_c, save_to)
    elif self.ori_x_dim == 2:
      for k in range(5):
        print(k, "------------")
        torch.manual_seed(k)
        self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to + "_v" + str(k), resolution, mode='mean_std')
        self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to + "_v" + str(k), resolution, mode='sample')
      #self._visualize_2d(X_t, Y_t, X_c, Y_c, save_to, resolution, mode='ar_sample')
      return
    else:
      raise NotImplementedError
    
          
  def compute_loss_and_metrics(self, X_t, Y_t, X_c=None, Y_c=None):
    
    """
    Compute loss for optimization, and metrics for monitoring.
    """
    if self.use_fourier_features:
      if X_c is not None:
        X_c = self.fourier_features(X_c)
      X_t = self.fourier_features(X_t)
      
    if self.training_objective == 'elbo':
      elbo, log = self.elbo(X_t, Y_t, X_c, Y_c)
      # Y_t_pred, _ = self.forward(X_t, X_c, Y_c)
      # mdiff = torch.abs(Y_t_pred - Y_t).mean()
      # if self.trainer.global_step % 200 == 1:
      #   logging.info(mdiff.detach().cpu().numpy())
    
      loss = - elbo.mean() 
      logs = {
        "elbo": elbo,
        "loss": loss,
        # "mean_diff": mdiff,
        #"log_prior": log['log_prior'].mean(),
        "log_likelihood": log['log_likelihood'].mean(),
        #'log_posterior': log['log_pos'].mean()
        "kl": log['kl'].mean()
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
      
    mll = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=False)
    mll_is = self.mll(X_t, Y_t, X_c, Y_c, n_latent_samples=self.n_latent_samples, use_importance_sampling=True)
    return {
      "marginal_log_likelihood": mll.mean(),
      "marginal_log_likelihood (Importance Sampling)": mll_is.mean(),
    }

  def evaluate_masks(self, X_t, Y_t, X_c=None, Y_c=None):
    """
    Evaluate the model by estimating the marginal_log_likelihood with different types of masks.
    """

    batch_size = X_t.shape[0]
    self.batch_size = 1
    self.cond_cif.batch_size = 1
    self.infer_net.batch_size = 1
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

