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

  
  if cfg.run.mode == 'train':
    task_manager.run_training()
  elif cfg.run.mode == 'eval':
    task_manager.run_evaluation()
  elif cfg.run.mode == 'visualize':
    task_manager.run_visualization()
  elif cfg.run.mode == 'visualize_data':
    task_manager.run_data_visualization()
  else:
    raise Exception("unknow running mode {}".format(cfg.run.mode))
  
if __name__ == "__main__":
    app()