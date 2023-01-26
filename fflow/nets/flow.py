import torch
from torch.nn import functional as F
import numpy as np

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
# from nflows.transforms.permutations import RandomPermutation, ReversePermutation

from .mlp import MLP
from .transform import unconstrained_rational_quadratic_spline

class ConditionalLinearTransformation(torch.nn.Module):
  
  def __init__(self,
               # n_flow_layers,
               x_dim,
               y_dim,
               c_dim,
               hidden_units,
               n_hidden_layers=3,
               epsilon=1e-4):
    super().__init__()
    # self.n_flow_layers = n_flow_layers 
    self.x_dim = x_dim
    if y_dim > 1:
      raise NotImplementedError
    self.y_dim = y_dim 
    self.c_dim = c_dim 
    self.hidden_units = hidden_units
    self.n_hidden_layers = n_hidden_layers
    dim_output = 2
    self._mlp = MLP(c_dim, [hidden_units] * n_hidden_layers + [dim_output])
    self.epsilon = epsilon 
    
  def _get_params(self, context=None):
    out = self._mlp(context)
    beta = out[:, :1]
    # alpha = 0.95 * torch.nn.Softplus()(out[:, 1:]) + 0.05
    alpha = torch.exp(torch.nn.Tanh()(out[:, 1:]))
    return alpha, beta
    
  def jacobian_determinant(self, Y, context=None):
    alpha, beta = self._get_params(context)
    Y = (Y - beta) / alpha
    logabsdet = torch.log(torch.abs(alpha[:, 0]))
    return Y, logabsdet
  
  def forward(self, Y, context=None):
    """
    Args:
      Y: [batch_size, y_dim]
      X: [batch_size, x_dim]
      context: [batch_size, c_dim]
    """
    alpha, beta = self._get_params(context)
    Y = alpha * Y + beta
    logabsdet = torch.log(torch.abs(alpha).sum(dim=-1))
    return Y, logabsdet
  
  def inverse(self, Y, context=None):
    alpha, beta = self._get_params(context)
    Y = (Y - beta) / alpha
    return Y

class ConditionalRationalQuadraticTransformation(torch.nn.Module):
  
  def __init__(self,
               # n_flow_layers,
               x_dim,
               y_dim,
               c_dim,
               n_bins,
               hidden_units,
               n_hidden_layers=3):
    super().__init__()
    # self.n_flow_layers = n_flow_layers 
    self.x_dim = x_dim
    self.y_dim = y_dim 
    self.c_dim = c_dim 
    self.n_bins = n_bins
    self.hidden_units = hidden_units
    self.n_hidden_layers = n_hidden_layers
    dim_output = 3 * n_bins - 1
    self._mlp = MLP(c_dim, [hidden_units] * n_hidden_layers + [dim_output])
  
  def _get_params(self, context=None):
    out = self._mlp(context)
    unnormalized_widths, unnormalized_heights, unnormalized_derivatives = out[:, :self.n_bins], out[:, self.n_bins:2*self.n_bins], out[:, 2*self.n_bins:]
    return unnormalized_widths, unnormalized_heights, unnormalized_derivatives
    
  def forward(self, Y, context=None):
    unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_params(context)
    Y, logabsdet = unconstrained_rational_quadratic_spline(Y[:, 0], unnormalized_widths, 
                                                           unnormalized_heights, 
                                                           unnormalized_derivatives,
                                                           tail_bound=5.,
                                                           inverse=True)
    Y = torch.unsqueeze(Y, dim=-1)
    return Y, logabsdet
  
  def inverse(self, Y, context=None):
    unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_params(context)
    Y, _ = unconstrained_rational_quadratic_spline(Y[:, 0], unnormalized_widths, 
                                                           unnormalized_heights, 
                                                           unnormalized_derivatives,
                                                           tail_bound=5.,
                                                           inverse=False)
    Y = torch.unsqueeze(Y, dim=-1)
    return Y
  
  def jacobian_determinant(self, Y, context=None):
    unnormalized_widths, unnormalized_heights, unnormalized_derivatives = self._get_params(context)
    Y, logabsdet = unconstrained_rational_quadratic_spline(Y[:, 0], unnormalized_widths, 
                                                           unnormalized_heights, 
                                                           unnormalized_derivatives,
                                                           tail_bound=5.,
                                                           inverse=False)
    Y = torch.unsqueeze(Y, dim=-1)
    return Y, -logabsdet
    

class ConditionalMaskedAutoregressiveTransformation(torch.nn.Module):
  
  def __init__(self,
               n_flow_layers,
               n_dim,
               c_dim,
               hidden_units,
               n_blocks_per_layer=2):
    super().__init__()
    self.n_flow_layers = n_flow_layers 
    self.n_dim = n_dim
    self.c_dim = c_dim 
    self.hidden_units = hidden_units
    
    self._layers = []
    for _ in range(n_flow_layers):
      self._layers.append(
          MaskedAffineAutoregressiveTransform(
              features=n_dim,
              hidden_features=hidden_units,
              context_features=c_dim,
              num_blocks=n_blocks_per_layer,
              use_residual_blocks=True,
              random_mask=False,
              activation=F.relu,
              dropout_probability=0.0,
              use_batch_norm=False,
          )
      )
      
    self._transform = CompositeTransform(self._layers)
      # layers.append(BatchNorm(n_dim))     
    # if base_distribution is None:
    #   base_distribution = StandardNormal([n_dim])
    # self._flow = Flow(
    #     transform=CompositeTransform(layers),
    #     distribution=base_distribution,
    # )
    
  def jacobian_determinant(self, z, context=None):
    embedded_context = context #self._layers._embedding_net(context)
    noise, logabsdet = self._transform(z, context=embedded_context)
    return noise, logabsdet
  
  def forward(self, z, context=None):
    return self._transform(z, context=context)
  
  def inverse(self, z, context=None):
    return self._transform.inverse(z, context=context)
  
  # def log_prob(self, z, context=None):
  #   return self._flow.log_prob(z, context=context)
  
  # def sample(self, num_samples=1, context=None):
  #   return self._flow.sample(num_samples=num_samples, context=context)
  
  

  
  








  

