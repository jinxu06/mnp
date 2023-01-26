import os
import sys
import numpy as np 
from scipy.interpolate import PchipInterpolator
import torch 
import torch.distributions as D
from pyro.distributions import InverseGamma
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from .base import DataModule

class MonotonicFunctionDataset(torch.utils.data.Dataset):
  
  def __init__(self, 
               n_samples,
               min_max=(-5., 5.),
               n_points=128,
               normalize=True,
               add_noise=True):
    """
    Torch dataset of random 1D monotonic functions. Generate data for the whole epoch.
    Using PCHIP 1-D monotonic cubic interpolation.
    
    Args:
    ----------------------------------------------------------------------------- 
    n_samples : int, optional
        Number of sampled functions contained in dataset.
        
    min_max : tuple of floats, optional
        Min and max point at which to evaluate the function (bounds).

    n_points : int, optional
        Number of points at which to evaluate f(x) for x in min_max.
    """
    
    self.n_samples = n_samples
    self.n_points = n_points 
    self.x_min, self.x_max = min_max
    self.normalize = normalize
    self.add_noise = add_noise
    
    self.X, self.Y = [], []
    for _ in range(self.n_samples):
      X, Y = self._generate_function()
      self.X.append(X)
      self.Y.append(Y)
    self.X = torch.stack(self.X)
    self.Y = torch.stack(self.Y)
    
  def _generate_function(self):

    N = int(D.Poisson(rate=5.0).sample().numpy())
    X_increments = torch.distributions.Dirichlet(concentration=torch.tensor([0.5 for _ in range(N+1)])).sample()
    X_increments = X_increments + 0.01
    X_increments = X_increments / X_increments.sum()
    X = torch.cumsum(X_increments * (self.x_max-self.x_min), dim=0) + self.x_min
    X = torch.cat([torch.tensor([self.x_min]), X])
    # X = torch.rand((N,)) * (self.x_max-self.x_min) + self.x_min
    # X = torch.cat([torch.tensor([self.x_min]), X, torch.tensor([self.x_max])])
    # # X, _ = X.sort(axis=0)
    # X = torch.unique(X, sorted=True, dim=0)
    Y = D.Gamma(2., 1.).sample([int(X.shape[0]),])
    Y = torch.cumsum(Y, dim=0) 
    interpolator = PchipInterpolator(X, Y, axis=0, extrapolate=True)
    
    X = np.random.uniform(size=(self.n_points,)) * (self.x_max-self.x_min) + self.x_min
    X = np.sort(X, axis=0)
    Y = interpolator(X)
    X = torch.from_numpy(X).unsqueeze(dim=-1).to(dtype=torch.float)
    Y = torch.from_numpy(Y).unsqueeze(dim=-1).to(dtype=torch.float)
    Y -= torch.from_numpy(interpolator(np.random.uniform(self.x_min, self.x_max, size=[1,]))).unsqueeze(dim=-1).to(dtype=torch.float)
    
    if self.normalize:
      Y = Y / (Y.max() - Y.min()) * 2
      Y = Y - Y.min() - 1

    if self.add_noise:
      Y = Y + 0.01 * torch.randn(Y.shape)

    return X, Y

  def __len__(self):
    return self.n_samples
  
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
  
  
class MonotonicFunctionDataModule(DataModule):
  
  def __init__(self, 
               x_dim,
               y_dim,
               n_context_points,
               n_total_points,
               batch_size,
               x_min_max=(-5., 5.),
               random_n_target_points=True,
               train_set_size=10000,
               val_set_size=2000,
               test_set_size=2000,
               shuffle=False, 
               num_workers=0, 
               datadir="", 
               random_seed=123):
    
    """
    Args:
    ----------------------------------------------------------------
    x_dim: int
        Dimension of inputs
        
    y_dim: int
        Dimension of outputs
        
    n_context_points: int or tuple of inputs
        Numober of context points. If it is a tuple [min, max], the number of context points are uniformly sampled between the range.
        
    n_total_points: int
        Total numober of data points.
        
    batch_size: int
        Number of function instances in a batch.
        
    x_min_max: tuple of ints, optional
        min, max values of x
        
    random_n_target_points: bool, optional
        Whether to randomly sample target points. If False, use all avaible data points as target points
        
    train_set_size: int, optional
        Number of samples in the training set.
        
    val_set_size: int, optional
        Number of samples in the validation set.
        
    test_set_size: int, optional
        Number of samples in the test set.
        
    shuffle: bool, optional
        Whether to shuffle the data along the batch axis 
    
    num_workers: int, optional
        Number of workers for the data loader
    
    datadir: str: optional
        Location to data storage
    
    random_seed: int, optional
        Random seed, default to None (not set)
    
    """
    
    super().__init__(x_dim=x_dim,
                    y_dim=y_dim,
                    n_context_points=n_context_points,
                    n_total_points=n_total_points,
                    batch_size=batch_size,
                    train_set_size=train_set_size,
                    val_set_size=val_set_size,
                    test_set_size=test_set_size,
                    shuffle=shuffle, 
                    num_workers=num_workers, 
                    datadir=datadir, 
                    random_seed=random_seed)
    self.x_min_max = x_min_max
    self.random_n_target_points = random_n_target_points
    
  def setup(self, stage=None):
    self.val_set = MonotonicFunctionDataset(n_samples=self.val_set_size,
                                            min_max=self.x_min_max,
                                            n_points=self.n_total_points)
    self.test_set = MonotonicFunctionDataset(n_samples=self.test_set_size,
                                             min_max=self.x_min_max,
                                             n_points=self.n_total_points)
    
    cntxt_trgt_getter = CntxtTrgtGetter(
          contexts_getter=GetRandomIndcs(a=self.min_n_context_points, b=self.max_n_context_points), targets_getter=get_all_indcs
    )
    
    def collate_fn_train(samples):
      xs, ys = [], []
      for x, y in samples:
        xs.append(x), ys.append(y)
      X, Y = torch.stack(xs), torch.stack(ys)
      X_cntxt, Y_cntxt, X_trgt, Y_trgt = cntxt_trgt_getter(X, Y)
      n_context_points = X_cntxt.shape[1]
      if self.random_n_target_points:
        # If use random number of target points, randomly sample max_n_context_points + 1 - n_context_points target points
        n_target_points = torch.randint(low=1, high=self.max_n_context_points + 2 - n_context_points, size=())
        indices = torch.randperm(self.n_total_points, generator=self.torch_rng)[:n_target_points]
        X_trgt = torch.index_select(X_trgt, dim=1, index=indices)
        Y_trgt = torch.index_select(Y_trgt, dim=1, index=indices)
      X_trgt = torch.cat([X_cntxt, X_trgt], dim=1)
      Y_trgt = torch.cat([Y_cntxt, Y_trgt], dim=1)
      return X_cntxt, Y_cntxt, X_trgt, Y_trgt
    
    def collate_fn_eval(samples):
      xs, ys = [], []
      for x, y in samples:
        xs.append(x), ys.append(y)
      X, Y = torch.stack(xs), torch.stack(ys)
      X_cntxt, Y_cntxt, X_trgt, Y_trgt = cntxt_trgt_getter(X, Y)
      return X_cntxt, Y_cntxt, X_trgt, Y_trgt

    # def collate_fn_eval(samples):
    #   xs, ys = [], []
    #   for x, y in samples:
    #     xs.append(x), ys.append(y)
    #   X, Y = torch.stack(xs), torch.stack(ys)
    #   X_cntxt, Y_cntxt, X_trgt, Y_trgt = cntxt_trgt_getter(X, Y)
    #   n_context_points = X_cntxt.shape[1]
    #   if self.random_n_target_points:
    #     # If use random number of target points, randomly sample max_n_context_points + 1 - n_context_points target points
    #     n_target_points = torch.randint(low=1, high=self.max_n_context_points + 2 - n_context_points, size=())
    #     indices = torch.randperm(self.n_total_points, generator=self.torch_rng)[:n_target_points]
    #     X_trgt = torch.index_select(X_trgt, dim=1, index=indices)
    #     Y_trgt = torch.index_select(Y_trgt, dim=1, index=indices)
    #   return X_cntxt, Y_cntxt, X_trgt, Y_trgt
    
    self.collate_fn_train = collate_fn_train
    self.collate_fn_eval = collate_fn_eval
    
  def train_dataloader(self):
    self.train_set = MonotonicFunctionDataset(n_samples=self.train_set_size,
                                              min_max=self.x_min_max,
                                              n_points=self.n_total_points)
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_train,
                      num_workers=self.num_workers,
                      pin_memory=True)
  