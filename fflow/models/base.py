import os
import sys
from absl import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR


class FunctionFlowModule(pl.LightningModule, torch.nn.Module):
  
  def __init__(self,
               has_context_set=False,
               optim_lr=0.0001, 
               weight_decay=0,
               profiler=None):
    super().__init__()
    self.has_context_set = has_context_set
    self.optim_lr = optim_lr 
    self.weight_decay = weight_decay
    if profiler is None:
      profiler = pl.profiler.PassThroughProfiler()
    self.profiler = profiler
    self.params = None 
    self.automatic_optimization = True
  
  def compute_loss_and_metrics(self, x_set, y_set):
    raise NotImplementedError
  
  def training_step(self, batch, batch_idx):
    X_c, Y_c, X_t, Y_t = batch
    #X_c, Y_c, X_t, Y_t = batch.X_c.to('cuda'), batch.Y_c.to('cuda'), batch.X_t.to('cuda'), batch.Y_t.to('cuda')
    if self.has_context_set:
      loss, logs = self.compute_loss_and_metrics(X_t, Y_t, X_c, Y_c)
    else:
      loss, logs = self.compute_loss_and_metrics(X_c, Y_c)
    self.log_dict({f"tr_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=X_c.shape[0])
    return loss 

  def validation_step(self, batch, batch_idx):
    X_c, Y_c, X_t, Y_t = batch
    #X_c, Y_c, X_t, Y_t = batch.X_c.to('cuda'), batch.Y_c.to('cuda'), batch.X_t.to('cuda'), batch.Y_t.to('cuda')
    with self.profiler.profile("compute loss"):
      if self.has_context_set:
        loss, logs = self.compute_loss_and_metrics(X_t, Y_t, X_c, Y_c)
        eval_logs = self.evaluate(X_t, Y_t, X_c, Y_c)
      else:
        loss, logs = self.compute_loss_and_metrics(X_c, Y_c)
        eval_logs = self.evaluate(X_c, Y_c)
    logs = {**logs, **eval_logs}
    self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=X_c.shape[0])
    return loss 

  
  def test_step(self, batch, batch_idx):
    
    self.eval()
    X_c, Y_c, X_t, Y_t = batch
    with self.profiler.profile("compute loss"):
      if self.has_context_set:
        loss, logs = self.compute_loss_and_metrics(X_t, Y_t, X_c, Y_c)
        #loss, logs = 0.0, {}
        #eval_logs = self.evaluate_masks(X_t, Y_t, X_c, Y_c)
        eval_logs = self.evaluate(X_t, Y_t, X_c, Y_c)
      else:
        loss, logs = self.compute_loss_and_metrics(X_c, Y_c)
        eval_logs = self.evaluate(X_c, Y_c)
    logs = {**logs, **eval_logs}
    self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=X_c.shape[0])
    return loss 
  
  def configure_optimizers(self):
    assert self.params is not None, "self.params is None"
    logging.debug("Optim Algo: ADAM, Weight Decay: {}".format(self.weight_decay))
    optimizer = torch.optim.Adam(self.params, lr=self.optim_lr, weight_decay=self.weight_decay)
    return optimizer

    # lr_scheduler = {
    #   'scheduler': MultiStepLR(optimizer, [int(self.trainer.max_epochs*m) for m in [0.8, 0.9, 0.95]], gamma=0.2),
    #   'interval': 'epoch'
    # }
    # return [optimizer], [lr_scheduler]
  

  
  
  

  
    
  
    

