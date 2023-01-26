import os
import sys
import numpy as np 
import torch 
from sklearn.gaussian_process.kernels import (
    RBF, 
    WhiteKernel, 
    Matern, 
    ExpSineSquared
)
from neural_process_family.utils.data import GPDataset
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from .base import DataModule

class GaussianProcessDataset(torch.utils.data.Dataset):
  
  def __init__(self, 
               x_dim = 1,
               kernel = (WhiteKernel(1e-4) + RBF(length_scale=1.0)),
               hyperparameters_generators=None,
               min_max=(-2, 2),
               n_samples=1000,
               n_points=128,
               is_vary_kernel_hyp=False,
               save_file=None,
               n_same_samples=20):
    """
    Torch dataset. Generate data for the whole epoch (may not be memory efficient).
    
    Args:
    ------------------------------------------------------------------------------
    x_dim: int
        dimension of inputs
    
    kernel : sklearn.gaussian_process.kernels or list
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.
        
    hyperparameters_generators: list of functions
        Generators for kernel hyperparameters.

    min_max : tuple of floats, optional
        Min and max point at which to evaluate the function (bounds).

    n_samples : int, optional
        Number of sampled functions contained in dataset.

    n_points : int, optional
        Number of points at which to evaluate f(x) for x in min_max.

    is_vary_kernel_hyp : bool, optional
        Whether to sample each example from a kernel with random hyperparameters,
        that are sampled uniformly in the kernel hyperparameters `*_bounds`. If False, 
        hyperparameters_generators is not used.

    save_file : string or tuple of strings, optional
        Where to save and load the dataset. If tuple `(file, group)`, save in
        the hdf5 under the given group. If `None` regenerate samples indefinitely.
        Note that if the saved dataset has been completely used,
        it will generate a new sub-dataset for every epoch and save it for future
        use.

    n_same_samples : int, optional
        Number of samples with same kernel hyperparameters and X. This makes the
        sampling quicker.
    """
    self._dataset = GPDataset(x_dim=x_dim,
                              kernel=kernel,
                              min_max=min_max,
                              n_samples=n_samples,
                              n_points=n_points,
                              is_vary_kernel_hyp=is_vary_kernel_hyp,
                              hyperparameters_generators=hyperparameters_generators,
                              save_file=save_file,
                              n_same_samples=n_same_samples,
                              is_reuse_across_epochs=False)
    self._dataset_size = n_samples
    self.n_points = n_points 
    self.X, self.Y = self._dataset.get_samples(n_samples=n_samples)

  def __len__(self):
    return self._dataset_size
  
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

class GaussianProcessDataModule(DataModule):
  
  """
  Pytorch lightning wrapper class
  """
  
  def __init__(self, 
               x_dim,
               y_dim,
               n_context_points,
               n_total_points,
               batch_size,
               x_min_max=(-2., 2.),
               kernel_type= "rbf_noise",
               is_vary_kernel_hyp=False,
               random_n_target_points=True,
               n_sample_samples=20,
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
        Total number of data points.
        
    batch_size: int
        Number of function instances in a batch.
        
    x_min_max: tuple of ints, optional
        min, max values of x
        
    kernel_type: str, optional
        Type of GP kernels, current options includes ['rbf_noise', 'matern_noise', 'periodic_noise', 'periodic'], default to 'rbf_noise'.
        
    is_vary_kernel_hyp : bool, optional
        Whether to sample each example from a kernel with random hyperparameters,
        that are sampled uniformly in the kernel hyperparameters `*_bounds`. If False, 
        hyperparameters_generators is not used.
        
    random_n_target_points: bool, optional
        Whether to randomly sample target points. If False, use all avaible data points as target points
    
    n_same_samples : int, optional
        Number of samples with same kernel hyperparameters and X. This makes the
        sampling quicker.
        
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
    self.is_vary_kernel_hyp = is_vary_kernel_hyp
    self.n_sample_samples = n_sample_samples
    self.random_n_target_points = random_n_target_points
    
    ## Setting the GP kernels and configure the hyperparameter generators
    if kernel_type == 'rbf_noise':
      self.kernel = (WhiteKernel(1e-4) + RBF(length_scale=0.25))
      self.hyperparameters_generators = [
        lambda: 10**(-self.numpy_rng.uniform(5., 7.)), # white noise variance
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # lengthscales
      ]
    elif kernel_type == 'rbf':
      self.kernel = (RBF(length_scale=0.25))
      self.hyperparameters_generators = [
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # lengthscales
      ]
    elif kernel_type == 'matern_noise':
      #self.kernel = (WhiteKernel(1e-4) + Matern(length_scale=0.25, nu=2.5))
      self.kernel = (WhiteKernel(1e-4) + Matern(length_scale=0.5, nu=2.5))
      self.hyperparameters_generators = [
        lambda: 10**(-self.numpy_rng.uniform(5., 7.)), # white noise variance
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # lengthscales
      ]
    elif kernel_type == 'matern':
      self.kernel = (Matern(length_scale=0.25, nu=2.5))
      self.hyperparameters_generators = [
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # lengthscales
      ]
    elif kernel_type == 'periodic_noise':
      self.kernel = (WhiteKernel(1e-2) + ExpSineSquared(length_scale=0.5, periodicity=0.5))
      self.hyperparameters_generators = [
        lambda: 10**(-self.numpy_rng.uniform(5., 7.)), # white noise variance
        lambda: self.numpy_rng.lognormal(0.0, 1.0), # lengthscales
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # periodicity
      ]
    elif kernel_type == 'periodic':
      self.kernel = (ExpSineSquared(length_scale=0.5, periodicity=0.5))
      self.hyperparameters_generators = [
        lambda: self.numpy_rng.lognormal(0.0, 1.0), # lengthscales
        lambda: self.numpy_rng.lognormal(0.0, 1.0) # periodicity
      ]
    else:
      raise Exception("Unknown kernel type {}".format(kernel_type))

    
  def setup(self, stage=None):
    self.val_set = GaussianProcessDataset(x_dim=self.x_dim, 
                                          min_max=self.x_min_max,
                                          kernel=self.kernel, 
                                          is_vary_kernel_hyp=self.is_vary_kernel_hyp, 
                                          hyperparameters_generators=self.hyperparameters_generators, 
                                          n_same_samples=self.n_sample_samples, 
                                          n_samples=self.val_set_size, 
                                          n_points=self.n_total_points)
    self.test_set = GaussianProcessDataset(x_dim=self.x_dim, 
                                           min_max=self.x_min_max,
                                           kernel=self.kernel, 
                                           is_vary_kernel_hyp=self.is_vary_kernel_hyp,
                                           hyperparameters_generators=self.hyperparameters_generators, 
                                           n_same_samples=self.n_sample_samples, 
                                           n_samples=self.test_set_size, 
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
    
    self.collate_fn_train = collate_fn_train
    self.collate_fn_eval = collate_fn_eval
    
  def train_dataloader(self):
    self.train_set = GaussianProcessDataset(x_dim=self.x_dim, 
                                            min_max=self.x_min_max,
                                            kernel=self.kernel, 
                                            is_vary_kernel_hyp=self.is_vary_kernel_hyp, 
                                            hyperparameters_generators=self.hyperparameters_generators, 
                                            n_same_samples=self.n_sample_samples, 
                                            n_samples=self.train_set_size, 
                                            n_points=self.n_total_points)
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.batch_size,
                      shuffle=self.shuffle, 
                      collate_fn=self.collate_fn_train,
                      num_workers=self.num_workers,
                      pin_memory=True)
  
    



    
    

  
  


