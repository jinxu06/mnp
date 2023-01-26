import colorsys
import copy
import numpy as np 
import torch 
import torchvision
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('off')
from PIL import Image 

def rescale(arr, vmin, vmax):
  arr = np.clip(arr, vmin, vmax)
  arr = (arr - vmin) / (vmax - vmin)
  return arr

def batch_image_display(imgs, gap=2, vmin=-1.0, vmax=1.0, figname="test.png"):
  """
  imgs: [X, Y, H, W, C]
  """
  X, Y, H, W, C = imgs.shape
  imgs = rescale(imgs, vmin, vmax)
  imgs = np.uint8(np.round(imgs *255))
  img_wall = np.ones([H * X + gap * (X-1), W * Y + gap * (Y-1), C], dtype=np.uint8) * 255
  for i in range(X):
    for j in range(Y):
      img_wall[i*(H+gap):(i+1)*(H+gap)-gap, j*(W+gap):(j+1)*(W+gap)-gap, :] = imgs[i, j]
  im = Image.fromarray(img_wall)
  return im.save(figname)

class Canvas(object):
  
  def __init__(self,
               figsize):
    super().__init__()
    self.figsize = figsize 
    self.fig = plt.figure(figsize=figsize)
    self.ax = self.fig.add_subplot(111)
    
  def save(self, filename=None):
    if filename is not None:
      self.fig.tight_layout()
      self.fig.savefig(filename)
      
  def plot_points(self, X, Y):
    """
    Args:
      X: [batch_size, set_size]
      Y: [batch_size, set_size]
    """
    if torch.is_tensor(X):
      X, Y = X.detach().numpy(), Y.detach().numpy()
    for i in range(X.shape[0]):
      sns.scatterplot(x=X[i], y=Y[i], ax=self.ax)
      
  def plot_lines(self, X, Y):
    """
    Args:
      X: [batch_size, set_size]
      Y: [batch_size, set_size]
    """
    if torch.is_tensor(X):
      X, Y = X.detach().numpy(), Y.detach().numpy()
    for i in range(X.shape[0]):
      sns.lineplot(x=X[i], y=Y[i], ax=self.ax)
    