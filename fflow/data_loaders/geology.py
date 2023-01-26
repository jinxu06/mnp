import os
import glob
import sys
import math
import numpy as np 
from PIL import Image, ImageFilter
import torch
import torchvision
import torch.distributions as D
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    get_all_indcs,
)
from .base import DataModule

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.ndimage


def signed_distance_transform(img_tensor):
  """Converts a binary image tensor to a signed distance function.
  
  Args:
      img_tensor (torch.Tensor): Image tensor of shape (1, height, width). Must be
          binary, i.e. have values in {0, 1} only.
  
  Notes:
      This function is based on the MetaSDF implementation:
      https://github.com/vsitzmann/metasdf/blob/master/dataio.py#L200
  """
  # Convert tensor to numpy so we can use scipy
  img_np = img_tensor.numpy()
  # This function calculates the distance of "foreground" points (i.e. points with 
  # value 1) to the nearest "background" point (i.e. points with value 0). As 
  # foreground points correspond to the inside of the river, this will correspond 
  # to the negative distances of the signed distance function
  negative_dists = scipy.ndimage.distance_transform_edt(img_np)
  # Flip image, so background becomes foreground and vice versa
  img_flipped = img_np - 1
  # Calculate distances on flipped image to get positive distances
  positive_dists = scipy.ndimage.distance_transform_edt(img_flipped)
  # Compute signed distance function
  signed_dists = positive_dists - negative_dists 
  # Distances are given in units of pixels, so normalize by image size to ensure
  # distances lie in [-1, 1]
  signed_dists /= img_np.shape[1]
  return signed_dists


class GeoFluvialDataset(torch.utils.data.Dataset):
  
  def __init__(self, 
               n_samples,
               resolution=128,
               data_augmentation=False,
               which_set='train',
               datadir=""):
    super().__init__()
    self.n_samples = n_samples
    self.resolution = resolution
    self.which_set = which_set
    folder = "{2}/{0}/{1}_{0}".format(128 if resolution <= 128 else 512, which_set, datadir)
    fs = glob.glob(folder+"/*.png")
    imgs = []
    for f in fs[:n_samples]:
      img = Image.open(f)
      # img = img.filter(ImageFilter.MaxFilter(3))
      # img = img.filter(ImageFilter.GaussianBlur(radius=3))
      img = img.resize((self.resolution, self.resolution))
      img = torch.tensor(np.array(img)).to(dtype=torch.float)
      img = (img / 255. >= 0.5).to(torch.float32)
      img = signed_distance_transform(img.unsqueeze(dim=0))
      img = torch.tensor(img).to(dtype=torch.float)
      # img = img - 0.5 #img / 255. - 0.5
      imgs.append(img)
    imgs = torch.stack(imgs, dim=0).unsqueeze(dim=-1)
    
    # data augmentation
    if data_augmentation:
      all_imgs = [torch.rot90(imgs, 2*k, dims=(2,3)) for k in range(2)]
      all_imgs = torch.cat(all_imgs)
      imgs = all_imgs[torch.randperm(all_imgs.shape[0])]
    ###################
    
    grid_x, grid_y = torch.meshgrid(torch.arange(resolution), torch.arange(resolution))
    X = torch.stack([grid_x, grid_y], dim=-1)
    Y = imgs
    self.X = X.reshape([resolution ** 2, 2])
    self.X = self.X / float(resolution-1) * 2 - 1.
    self.Y = Y.reshape([-1, resolution ** 2, 1])
    
    self.Y = (self.Y < 0) * torch.tanh(100 * self.Y) + (self.Y >= 0) * torch.tanh(20 * self.Y)
    
  def __len__(self):
      return self.n_samples
  
  def __getitem__(self, idx):
    return self.X, self.Y[idx]
  
      
class GeoFluvialDataModule(DataModule):
  
  """
  Pytorch lightning wrapper class
  """
  
  def __init__(self, 
               n_context_points,
               n_total_points,
               batch_size,
               resolution=128,
               data_augmentation=False,
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
    
    super().__init__(x_dim=2,
                    y_dim=1,
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
    self.random_n_target_points = random_n_target_points
    self.resolution = resolution 
    self.data_augmentation = data_augmentation
    
  def vis_dataloader(self):
    return torch.utils.data.DataLoader(self.vis_set, 
                      batch_size=self.batch_size, 
                      shuffle=False, 
                      collate_fn=self.collate_fn_vis,
                      num_workers=self.num_workers,
                      pin_memory=True)
    
  def setup(self, stage=None):
    self.train_set = GeoFluvialDataset(n_samples=self.train_set_size, 
                                       resolution=self.resolution,
                                       data_augmentation=self.data_augmentation,
                                       which_set='train',
                                       datadir=self.datadir)
    self.val_set = GeoFluvialDataset(n_samples=self.val_set_size, 
                                       resolution=self.resolution,
                                       which_set='val',
                                       datadir=self.datadir)
    self.test_set = GeoFluvialDataset(n_samples=self.test_set_size, 
                                       resolution=self.resolution,
                                       which_set='test',
                                       datadir=self.datadir)
    self.vis_set = GeoFluvialDataset(n_samples=self.batch_size, 
                                       resolution=self.resolution,
                                       data_augmentation=self.data_augmentation,
                                       which_set='val',
                                       datadir=self.datadir)
    
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
      Y_trgt = Y_trgt + 0.05 * torch.randn(Y_trgt.shape) 
      return X_cntxt, Y_cntxt, X_trgt, Y_trgt
    
    def collate_fn_vis(samples):
      xs, ys = [], []
      for x, y in samples:
        xs.append(x), ys.append(y)
      X, Y = torch.stack(xs), torch.stack(ys)
      X_cntxt, Y_cntxt, X_trgt, Y_trgt = cntxt_trgt_getter(X, Y)
      # Y_trgt = Y_trgt + 1.0 * torch.rand(Y_trgt.shape) - 0.5
      # Y_trgt = Y_trgt + 0.1 * torch.rand(Y_trgt.shape) - 0.05
      # Y_trgt = Y_trgt + 0.05 * torch.randn(Y_trgt.shape) 
      Y_trgt = Y_trgt + 0.05 * torch.randn(Y_trgt.shape) 
      return X_cntxt, Y_cntxt, X_trgt, Y_trgt
      #return CustomBatch((X_cntxt, Y_cntxt, X_trgt, Y_trgt)) 
    
    self.collate_fn_train = collate_fn_train
    self.collate_fn_eval = collate_fn_vis # collate_fn_train
    self.collate_fn_vis = collate_fn_vis
    
    
  def visualize(self, save_to=None):
    dataloader = self.vis_dataloader()
    X_c, Y_c, X, Y = next(iter(dataloader)) # Only use the target points
    X_c = (X_c + 1.) / 2. * (self.resolution-1)
    w = int(Y.shape[1] ** 0.5)
    Y = Y.view(-1, w, w)
    # Y = (Y >= 0.5).to(torch.float)
    # subplots:
    n_plots = X.shape[0]
    n_cols = int(n_plots**0.5) 
    n_rows = int(math.ceil(n_plots / n_cols))
    
    fig_sdf = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    for i in range(X.shape[0]):
      ax = fig_sdf.add_subplot(n_rows, n_cols, i+1)
      ax.imshow(Y[i], cmap='Greys',  interpolation='nearest')  #, vmin=0., vmax=1.)
      Xs = torch.index_select(X_c[i], dim=0, index=torch.argwhere(Y_c[i, :, 0] < 0.0)[..., 0])
      ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
      Xs = torch.index_select(X_c[i], dim=0, index=torch.argwhere(Y_c[i, :, 0] >= 0.0)[..., 0])
      ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    fig_sdf.tight_layout()
    
    fig_bn = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    for i in range(X.shape[0]):
      ax = fig_bn.add_subplot(n_rows, n_cols, i+1)
      ax.imshow(Y[i] >= 0, cmap='Greys',  interpolation='nearest')  
      Xs = torch.index_select(X_c[i], dim=0, index=torch.argwhere(Y_c[i, :, 0] < 0.0)[..., 0])
      ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
      Xs = torch.index_select(X_c[i], dim=0, index=torch.argwhere(Y_c[i, :, 0] >= 0.0)[..., 0])
      ax.scatter(x=Xs[:, 1], y=Xs[:, 0])
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    fig_bn.tight_layout()
    
    if save_to is not None:
      fig_sdf.savefig(save_to + "_sdf.png")
      fig_bn.savefig(save_to + "_bn.png")
    else:
      raise ValueError

    
