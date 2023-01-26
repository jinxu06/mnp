import os
from tqdm import tqdm

from absl import logging
import numpy as np
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import sdeint

from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from .base import DataModule

def load_or_generate_data(n_samples, 
                          n_points = 501, 
                          min_max = (-5., 5.),
                          which_set = 'train', 
                          datadir = "/data/ziz/not-backed-up/jxu/data"):
  x_min, x_max = min_max
  tspan = np.linspace(x_min, x_max, n_points)
  filename = datadir + "/sde_s{0}_p{1}_{2}.npy".format(n_samples, n_points, which_set)
  if os.path.exists(filename):
    all_data = np.load(filename)
  else:
    logging.info("Generate {0} data,  n_samples={1}, n_points={2}......".format(which_set, n_samples, n_points))
    a, b = 0.1, 0.1
    x0 = np.random.uniform(0.2, 0.6, size=(n_samples,))
    def f(x, t):
        return -(a + x*b**2)*(1 - x**2)
    def g(x, t):
        return b * (1 - x**2)
    all_data = []
    for i in tqdm(range(n_samples)):
      all_data.append(sdeint.stratKP2iS(f, g, x0[i], tspan)[:, 0])
    all_data = np.array(all_data)
    np.save(filename, all_data)
  return tspan, all_data

class StratonovichSDEDataset(torch.utils.data.Dataset):
  
  def __init__(self,
               n_samples = 1000,
               n_points = 501,
               min_max = (-5., 5.),
               which_set='train'):
    self._n_samples = n_samples
    self._n_points = n_points 
    self.which_set = which_set 
    self.X, self.Y = load_or_generate_data(n_samples, n_points, min_max, which_set)
    self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
    self.Y = torch.tensor(self.Y, dtype=torch.float32).unsqueeze(-1)
  
  def __len__(self):
    return self._n_samples
  
  def __getitem__(self, idx):
    return self.X, self.Y[idx]
  

class StratonovichSDEDataModule(DataModule):
  
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
    self.val_set = StratonovichSDEDataset(n_samples=self.val_set_size,
                                             n_points=self.n_total_points,
                                             min_max=self.x_min_max,
                                             which_set='val')
    self.test_set = StratonovichSDEDataset(n_samples=self.test_set_size,
                                             n_points=self.n_total_points,
                                             min_max=self.x_min_max,
                                             which_set='test')
    
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
    self.train_set = StratonovichSDEDataset(n_samples=self.train_set_size,
                                               n_points=self.n_total_points,
                                               min_max=self.x_min_max,
                                               which_set='train')
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_train,
                      num_workers=self.num_workers,
                      pin_memory=True)










  
  

