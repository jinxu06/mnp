import collections
import math
import numpy as np 
import torch 
import pytorch_lightning as pl

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class DataModule(pl.LightningDataModule):
  
  """
  Base Class for Data Modules
  """
  
  def __init__(self, 
               x_dim,
               y_dim,
               n_context_points,
               n_total_points,
               batch_size,
               train_set_size=10000,
               val_set_size=2000,
               test_set_size=2000,
               shuffle=False, 
               num_workers=0, 
               datadir="", 
               random_seed=None):
    
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
    
    super().__init__()
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.n_context_points = n_context_points
    if isinstance(self.n_context_points, collections.Sequence) and len(self.n_context_points) == 2:
      self.min_n_context_points, self.max_n_context_points = self.n_context_points
    else:
      self.min_n_context_points, self.max_n_context_points = self.n_context_points, self.n_context_points
    self.n_total_points = n_total_points
    self.batch_size = batch_size
    self.train_set_size = train_set_size
    self.val_set_size = val_set_size
    self.test_set_size = test_set_size
    
    self.shuffle = shuffle    
    self.num_workers = num_workers
    self.datadir = datadir 
    self.random_seed = random_seed
    self.numpy_rng = np.random.RandomState(random_seed)
    self.torch_rng = torch.Generator()
    self.torch_rng.manual_seed(random_seed)
    
    self.train_set = None 
    self.val_set = None 
    self.test_set = None 
    self.collate_fn_train = None  
    self.collate_fn_eval = None  
    
  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_train,
                      num_workers=self.num_workers,
                      pin_memory=True)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_eval,
                      num_workers=self.num_workers,
                      pin_memory=True)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_set, 
                      batch_size=self.batch_size, 
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_eval,
                      num_workers=self.num_workers,
                      pin_memory=True)
    
  def vis_dataloader(self):
    return torch.utils.data.DataLoader(self.val_set, 
                      batch_size=self.batch_size, 
                      shuffle=False, 
                      collate_fn=self.collate_fn_eval,
                      num_workers=self.num_workers,
                      pin_memory=True)
    
  def visualize(self, save_to=None):
    dataloader = self.vis_dataloader()
    _, _, X, Y = next(iter(dataloader)) # Only use the target points
    if self.x_dim == 1 and self.y_dim == 1:
      # subplots:
      n_plots = X.shape[0]
      n_cols = int(n_plots**0.5) 
      n_rows = int(math.ceil(n_plots / n_cols))
      fig = plt.figure(figsize=(3*n_cols, 2*n_rows))
      for i in range(X.shape[0]):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax = sns.lineplot(x=X[i, :, 0], y=Y[i, :, 0], ax=ax, color="forestgreen")
      fig.tight_layout()
      if save_to is not None:
        fig.savefig(save_to + "_subplots.png")
      else:
        raise ValueError
      # combined 
      fig = plt.figure(figsize=(9, 6))
      ax = fig.add_subplot(1, 1, 1)
      for i in range(X.shape[0]):
        ax = sns.lineplot(x=X[i, :, 0], y=Y[i, :, 0], ax=ax, color="forestgreen")
      fig.tight_layout()
      if save_to is not None:
        fig.savefig(save_to + "_combined.png")
      else:
        raise ValueError
    else:
      raise NotImplementedError