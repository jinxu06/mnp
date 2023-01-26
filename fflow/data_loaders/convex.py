import os
import sys
import numpy as np 
from scipy.interpolate import PchipInterpolator
import torch 
import torch.distributions as D
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from .base import DataModule
from .monotonic import MonotonicFunctionDataset

class ConvexFunctionDataset(torch.utils.data.Dataset):
  
  def __init__(self, 
               n_samples,
               min_max=(-5., 5.),
               n_points=128,
               normalize=True,
               add_noise=False):
    
    """
    Torch dataset of random 1D convex functions. Generate data for the whole epoch.
    We use monotonic functions generated from MonotonicFunctionDataset as derivatives 
    to generate these convex functions
    
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
    
    dataset = MonotonicFunctionDataset(n_samples, min_max, n_points, normalize=False, add_noise=False)
    X, Y = dataset.X, dataset.Y
    X_diff = torch.diff(X, dim=1)
    X_diff = torch.cat([X_diff[:, :1], X_diff], dim=1)
    Y = torch.cumsum(Y * X_diff, dim=1) 
    Y = (Y - Y.amin(dim=1, keepdim=True)) / (Y.amax(dim=1, keepdim=True) - Y.amin(dim=1, keepdim=True))
    Y += D.Uniform(-1., 1.).sample([Y.shape[0], 1, 1])
    Y *= D.Gamma(3., 1.).sample([Y.shape[0], 1, 1])
    
    if self.normalize:
      Y = Y / (Y.amax(dim=1, keepdim=True) - Y.amin(dim=1, keepdim=True)) * 2
      Y = Y - Y.amin(dim=1, keepdim=True) - 1

    if self.add_noise:
      Y = Y + 0.01 * torch.randn(Y.shape)
    
    self.X, self.Y = X, Y

  def __len__(self):
    return self.n_samples
  
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
  
  
class ConvexFunctionDataModule(DataModule):
  
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
        Min, max values of x
        
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
    self.val_set = ConvexFunctionDataset(n_samples=self.val_set_size,
                                         min_max=self.x_min_max,
                                         n_points=self.n_total_points)
    self.test_set = ConvexFunctionDataset(n_samples=self.test_set_size,
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
    self.train_set = ConvexFunctionDataset(n_samples=self.train_set_size,
                                           min_max=self.x_min_max,
                                           n_points=self.n_total_points)
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_train,
                      num_workers=self.num_workers,
                      pin_memory=True)

  
  

    