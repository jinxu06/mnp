import os
import sys
import math
from absl import logging

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
import pytorch_lightning as pl

import typing
from tsalib import dim_vars as dvs, get_dim_vars, update_dim_vars_len
from tsalib import warp
from pytorch_lightning.profiler import BaseProfiler
import gpytorch

from .base import FunctionFlowModule
from ..nets import ConditionalLinearTransformation, ConditionalRationalQuadraticTransformation

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('off')

_B, _S, _C, _T = dvs('Batch(b):10 SetSize(s):1 Context(c):1 Target(t):1', exists_ok=True)
_X, _Y = dvs('DimX(x):1 DimY(y):1', exists_ok=True)

class ExactGPModel(gpytorch.models.ExactGP):
  
  def __init__(self, 
               X_c: torch.Tensor, 
               Y_c: torch.Tensor, 
               likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood(), 
               mean_module: gpytorch.means.Mean = gpytorch.means.ConstantMean(),
               covar_module: gpytorch.kernels.Kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())):
    super(ExactGPModel, self).__init__(X_c, Y_c, likelihood)
    self.likelihood = likelihood
    self.mean_module = mean_module
    self.covar_module = covar_module

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
  

class GaussianCopulaProcess(FunctionFlowModule):
  
  def __init__(self,
               x_dim: int,
               y_dim: int,
               n_hidden_layers: int,
               n_hidden_units: int,
               n_flow_layers: int,
               batch_size: int,
               likelihood: str = 'Gaussian', 
               initial_noise_std: float = 0.001,
               mean_cls: str = 'zero',
               kernel_cls: str = 'rbf',
               optim_lr: float = 0.0001,
               profiler: typing.Union[torch.Tensor, None] = None):
    
    """
    Gaussian Processes.
    
    Args:
    ---------------------------------------------------------
    x_dim: int
        Dimension of inputs
        
    y_dim: int
        Dimension of outputs
      
    batch_size: int
        Number of examples in one batch
        
    likelihood: str, optional
        Type of Gaussian process likelihood functions
    
    mean_cls: str, optional
        Type of Gaussian process mean functions
        
    kernel_cls: str, optional
        Type of Gaussian processes kernel functions 
        
    optim_lr: float, optional
        Learning rate of optimizers.
        
    profiler: BaseProfiler, optional
        Tools to profile your training/testing/inference run can help you identify 
        bottlenecks in your code.
    """
    
    super().__init__(has_context_set=True,
                     optim_lr=optim_lr, 
                     profiler=profiler)
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.batch_size = batch_size
    self.n_hidden_layers = n_hidden_layers
    self.n_hidden_units = n_hidden_units
    self.n_flow_layers = n_flow_layers
    self.is_oracle = False
    batch_shape = torch.Size([batch_size,])

    if likelihood == 'Gaussian':
      self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-7))
      self.likelihood.noise = initial_noise_std ** 2
    if mean_cls == 'zero':
      self.mean_module = gpytorch.means.ConstantMean()
    if kernel_cls == 'rbf':
      self.covar_module = gpytorch.kernels.RBFKernel()
      if self.is_oracle:
        self.covar_module.lengthscale = 0.25
      else:
        self.covar_module.lengthscale = 1.0
    elif kernel_cls == 'matern':
      self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
      if self.is_oracle:
        self.covar_module.lengthscale = 0.5
      else:
        self.covar_module.lengthscale = 1.0
    elif kernel_cls == 'periodic':
      self.covar_module = gpytorch.kernels.PeriodicKernel()
      if self.is_oracle:
        self.covar_module.lengthscale = 0.5
        self.covar_module.period_length = 0.5
      else:
        self.covar_module.lengthscale = 1.0
        self.covar_module.period_length = 1.0
    elif kernel_cls == 'composition':
      rbf_kernel = gpytorch.kernels.RBFKernel()
      rbf_kernel.lengthscale = 1.0
      matern_kernel = gpytorch.kernels.MaternKernel()
      matern_kernel.lengthscale = 1.0
      periodic_kernel = gpytorch.kernels.PeriodicKernel()
      periodic_kernel.lengthscale = 1.0
      periodic_kernel.period_length = 1.0
      self.covar_module = gpytorch.kernels.AdditiveKernel(rbf_kernel, matern_kernel, periodic_kernel)
    self.gp_module = ExactGPModel(None, 
                                  None, 
                                  self.likelihood, 
                                  self.mean_module, 
                                  self.covar_module)
    
    self.flows = torch.nn.ModuleList([ConditionalRationalQuadraticTransformation(x_dim=self.x_dim, 
                                                      y_dim=self.y_dim, 
                                                      c_dim=self.x_dim,
                                                      n_bins=10,
                                                      hidden_units=self.n_hidden_units,
                                                      n_hidden_layers=self.n_hidden_layers) for _ in range(self.n_flow_layers)])
    
    self.params = self.parameters()
      
    update_dim_vars_len({'b': self.batch_size,
                        'x': self.x_dim, 
                        'y': self.y_dim})
    
    logging.debug("---------       Gaussian Processes       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
    
  
  def predict(self,
              X_t: torch.Tensor, 
              X_c: typing.Union[torch.Tensor, None] = None, 
              Y_c: typing.Union[torch.Tensor, None] = None):
    update_dim_vars_len({'c': X_c.shape[1], 't': X_t.shape[1]})
    _B, _S, _C, _T, _X, _Y = get_dim_vars('b s c t x y')
 
    # Y_c, _ = self.flow.jacobian_determinant(warp(Y_c, 'b,c,y -> b*c,y', 'v'), 
    #                                                    context=warp(X_c, 'b,c,x -> b*c,x', 'v'))
    Y_c = warp(Y_c, 'b,c,y -> b*c,y', 'v')
    for k in range(self.n_flow_layers):
      Y_c, _ = self.flows[-k-1].jacobian_determinant(Y_c, context=warp(X_c, 'b,c,x -> b*c,x', 'v'))
    Y_c = warp(Y_c, 'b*c,y -> b,c,y', 'v')
    
    self.eval()
    self.gp_module.set_train_data(X_c, Y_c[..., 0], strict=False)
    f_preds = self.gp_module(X_t)
    f_mean = f_preds.mean.unsqueeze(-1)
    f_var = f_preds.variance.unsqueeze(-1)
    f1 = f_mean + f_var ** 0.5
    # f_mean, _ = self.flow(warp(f_mean, 'b,t,y -> b*t,y', 'v'), context=warp(X_t, 'b,t,x -> b*t,x', 'v'))
    # f1, _ = self.flow(warp(f1, 'b,t,y -> b*t,y', 'v'), context=warp(X_t, 'b,t,x -> b*t,x', 'v'))
    f_mean = warp(f_mean, 'b,t,y -> b*t,y', 'v')
    for k in range(self.n_flow_layers):
      f_mean, _ = self.flows[k](f_mean, context=warp(X_t, 'b,t,x -> b*t,x', 'v'))
    f1 = warp(f1, 'b,t,y -> b*t,y', 'v')
    for k in range(self.n_flow_layers):
      f1, _ = self.flows[k](f1, context=warp(X_t, 'b,t,x -> b*t,x', 'v'))

    f_mean = warp(f_mean, 'b*t,y -> b,t,y', 'v')
    f1 = warp(f1, 'b*t,y -> b,t,y', 'v')
    f_std = f1 - f_mean
    return f_mean, f_std
    
  def forward(self, 
              X_t: torch.Tensor, 
              X_c: torch.Tensor, 
              Y_c: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    y_mu, y_sigma = self.predict(X_t, X_c, Y_c)
    return y_mu, y_sigma 
  
  def mll(self, 
      X_t: torch.Tensor, 
      Y_t: torch.Tensor, 
      X_c: typing.Union[torch.Tensor, None] = None, 
      Y_c: typing.Union[torch.Tensor, None] = None,
      conditional: bool = False):
    """
    Compute the exact marginal log likelihood.
    
    If conditional:
        \log p(Y_t | X_t, X_c, Y_c)
    else
        \log p(Y_c, Y_t | X_c, X_t)
    
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
        
    conditional: bool, optional
        if conditional is true, return \log p(Y_t | X_t, X_c, Y_c)
        else return \log p(Y_c, Y_t | X_c, X_t)
        
    Returns:
    ---------------------------------------------------------
    Marginal log likelihood, tensor of shape [Batch_Size,]
    """
    update_dim_vars_len({'c': X_c.shape[1], 't': X_t.shape[1]})
    _B, _S, _C, _T, _X, _Y = get_dim_vars('b s c t x y')

    X_ct: 'b,c+t,x' = torch.cat([X_c, X_t], dim=1)
    Y_ct: 'b,c+t,y' = torch.cat([Y_c, Y_t], dim=1)    
    # Y_ct, logabsdet = self.flow.jacobian_determinant(warp(Y_ct, 'b,c+t,y -> b*(c+t),y', 'v'), 
    #                                                    context=warp(X_ct, 'b,c+t,x -> b*(c+t),x', 'v'))
    Y_ct = warp(Y_ct, 'b,c+t,y -> b*(c+t),y', 'v')
    logabsdet = 0
    for k in range(self.n_flow_layers):
      Y_ct, lad = self.flows[-k-1].jacobian_determinant(Y_ct, 
                                    context=warp(X_ct, 'b,c+t,x -> b*(c+t),x', 'v'))
      logabsdet = logabsdet + lad
    Y_ct = warp(Y_ct, 'b*(c+t),y -> b,c+t,y', 'v')
    logabsdet = logabsdet.view(_B, (_C+_T))[:, _C:].sum(-1)
    if conditional:
      self.gp_module.eval()
      self.gp_module.set_train_data(X_c, Y_ct[:, :_C, 0], strict=False)
      mll_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_module)
      output = self.gp_module(X_t)
      return mll_func(output, Y_ct[:, _C:, 0]) - logabsdet / Y_t.shape[1]
    else:
      self.gp_module.set_train_data(X_ct, Y_ct[..., 0], strict=False)
      mll_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_module)
      output = self.gp_module(X_ct)
      return mll_func(output, Y_ct[..., 0]) - logabsdet / Y_ct.shape[1]

  
  def sample(self, 
             X_t: torch.Tensor, 
             X_c: torch.Tensor,
             Y_c: torch.Tensor):
    pass 
  
  def compute_loss_and_metrics(self, 
                               X_t: torch.Tensor, 
                               Y_t: torch.Tensor, 
                               X_c: torch.Tensor, 
                               Y_c: torch.Tensor):
    """
    Compute loss for optimization, and metrics for monitoring.
    """
    mll = self.mll(X_t, Y_t, X_c, Y_c, conditional=True)
    loss = - mll.mean()
    logs = {
      "loss": loss,
      "mll": mll
    }
    return loss, logs
  
  def evaluate(self, 
               X_t: torch.Tensor, 
               Y_t: torch.Tensor, 
               X_c: typing.Union[torch.Tensor, None] = None, 
               Y_c: typing.Union[torch.Tensor, None] = None):
    """
    Evaluate the model by estimating the marginal_log_likelihood.
    """
    mll = self.mll(X_t, Y_t, X_c, Y_c, conditional=True)
    return {"conditional_marginal_log_likelihood": mll.mean()}
  
  def visualize(self, X_t, Y_t, X_c, Y_c, save_to=None, resolution=None):
    # Batch_Size subplots
    # Layout subplots
    n_plots = X_t.shape[0]
    n_cols = int(n_plots**0.5) 
    n_rows = int(math.ceil(n_plots / n_cols))
    fig = plt.figure(figsize=(3*n_cols, 2*n_rows))
    n_samples = 10
    for i in range(n_plots):
      n_context_points = torch.randint(low=1, high=X_c.shape[1], size=[])
      _X_c, _Y_c = X_c[:, :n_context_points], Y_c[:, :n_context_points]
      ax = fig.add_subplot(n_rows, n_cols, i+1)
      # Scatter plots for all context points
      ax = sns.scatterplot(x=_X_c[i, :, 0], y=_Y_c[i, :, 0], ax=ax, facecolor="forestgreen", marker='o')
      # Plot the ground truth for target points
      ax = sns.lineplot(x=X_t[i, :, 0], y=Y_t[i, :, 0], ax=ax, color="forestgreen")

      mu, sigma = self.forward(X_t.to(self.device), _X_c.to(self.device), _Y_c.to(self.device))
      mu, sigma = mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()
      # Plot predictive means for the target points
      ax = sns.lineplot(x=X_t[i, :, 0], y=mu[i, :, 0], ax=ax, color="blueviolet", alpha=0.5)
      # Plot predictive std for the target points
      ax.fill_between(x=X_t[i, :, 0], y1=mu[i, :, 0]-sigma[i, :, 0], y2=mu[i, :, 0]+sigma[i, :, 0], alpha=0.1, color="blueviolet")
      
    fig.tight_layout()
    if save_to is not None:
      fig.savefig(save_to)
    else:
      return fig