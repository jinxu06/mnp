import os
import sys
import random
import numpy as np
from absl import logging
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import pytorch_lightning as pl

# torch.use_deterministic_algorithms(True)

sys.path.insert(1, "third_party/neural_process_family")
sys.path.insert(1, "third_party/tsalib")
sys.path.insert(1, "third_party")

from experiments.manager import TaskManager

torch.backends.cudnn.allow_tf32 = True

@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg):
  
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)
  
  # random seeds
  random.seed(cfg.run.random_seed)
  torch.manual_seed(cfg.run.random_seed)
  np.random.seed(cfg.run.random_seed)
  
  # logging
  if cfg.run.mode == 'train':
    logging.set_verbosity(logging.DEBUG)
    logging.info(cfg.run.exp_name)
  else:
    logging.set_verbosity(logging.INFO)
    
  # torch.backends.cudnn.enabled = True
  # torch.backends.cudnn.allow_tf32 = True
  
  # running mode 
  task_manager = TaskManager(cfg)

  # task_manager.data_module.setup()
  # dataloader = task_manager.data_module.val_dataloader()
  # cfg = task_manager.config
  
  # quit()
  
  # for X_c, Y_c, X_t, Y_t in dataloader:
    # z_seq = torch.normal(0., 1., size=(5, cfg.data.train_batch_size, cfg.model.z_dim))
    # elbo = task_manager.model.elbo(X_t, Y_t, X_c, Y_c)
    # print(elbo)
    # samples = task_manager.model.sample(X_t, X_c, Y_c)
    # print(samples)
    # quit()
  
  if cfg.run.mode == 'train':
    task_manager.run_training()
  elif cfg.run.mode == 'eval':
    task_manager.run_evaluation()
  elif cfg.run.mode == 'visualize':
    task_manager.run_visualization()
  elif cfg.run.mode == 'visualize_data':
    task_manager.run_data_visualization()
  elif cfg.run.mode == 'debug':
    task_manager.run_debug()
  elif cfg.run.mode == 'infer_hyp':
    task_manager.infer_hyperparameters()
  else:
    raise Exception("unknow running mode {}".format(cfg.run.mode))
  
if __name__ == "__main__":
    app()