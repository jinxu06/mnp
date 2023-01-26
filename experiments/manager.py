import os
import sys
import json
import pickle
import shutil
import numpy as np
import time 
from absl import logging
from fflow.models.gp import GaussianProcess
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from fflow.data_loaders import (
    GaussianProcessDataModule, 
    ConvexFunctionDataModule, 
    RandomGaussianMixtureCDFDataModule,
    MonotonicFunctionDataModule,
    GeoFluvialDataModule,
    MNISTDataModule,
    LearningCurveDataModule,
    StratonovichSDEDataModule,
    WheelDataModule,
)
from fflow.models import (
    DiscreteFunctionFlowModule, 
    NeuralProcess, 
    GaussianProcess, 
    BRUNO,
    GaussianCopulaProcess
)
from fflow.utils import infer_optimal_gp_hyperparameters

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.axis('on')

class TaskManager(object):
  
  def __init__(self, config):
    self.config = config 
    
    # Setting full experiment name for logging
    self.model_name = self.config.model.name
    self.exp_name = "[{0}]-[model:{1}]-[data:{2}]-[train_size:{3}]-[seed:{4}]".format(
                                                            self.config.run.exp_name, 
                                                            self.model_name, 
                                                            self.config.data.name, 
                                                            self.config.data.train_set_size, 
                                                            self.config.run.random_seed)
    self._load_data()
    self._create_model()
    
    overwrite = self.config.run.mode == 'train' and (not self.config.run.restore)
    logger, checkpoint_callbacks = self._setup_logging_and_checkpointing(overwrite=overwrite)
    if torch.cuda.device_count() >=1:
      gpus = 1 
      self.device = 'cuda'
    else:
      gpus = None
      self.device = 'cpu'
    
    # Create ProgressBar theme
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
        
    # configure the trainer
    self.trainer = pl.Trainer(accelerator=self.config.run.device,
                              # overfit_batches=1 if self.config.data.name == 'wheel' else 0,
                              gpus=gpus,
                              logger=logger, 
                              default_root_dir=os.path.join(self.config.run.logdir, self.exp_name),
                              reload_dataloaders_every_n_epochs=1 if self.config.data.reload else 0, 
                              check_val_every_n_epoch=self.config.train.check_val_every_n_epoch,
                              enable_checkpointing=True,
                              gradient_clip_val=self.config.train.gradient_clip_val,
                              callbacks=checkpoint_callbacks + [progress_bar],
                              precision=self.config.train.precision, 
                              max_epochs=self.config.run.max_epochs+1)
    
    if self.config.run.restore:
      self._restore_model()
    
    # Configure the output directory
    self.results_dir = self.exp_name
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)
      

      
  def _setup_logging_and_checkpointing(self, overwrite):
    dir_path = os.path.join(self.config.run.logdir, self.exp_name)
    if overwrite:
      if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
      os.makedirs(dir_path)
      os.makedirs(os.path.join(dir_path, "checkpoints"))
      os.makedirs(os.path.join(dir_path, "wandb_logs"))
    
    
    if 'debug' in self.config.run.exp_name or self.config.run.mode != 'train':
      logger = False
    else:
      if self.config.run.logger == "wandb":
        logger = WandbLogger(name=self.exp_name,
                            save_dir=os.path.join(dir_path, "wandb_logs"),
                            prefix='',
                            project=self.config.run.wandb_project,
                            entity=self.config.run.wandb_username)
      elif self.config.run.logger == "tensorboard":
        logger = TensorBoardLogger(os.path.join(dir_path, "tb_logs"), 
                                name="", 
                                version=self.config.run.version)
                               
    checkpoint_callbacks = [ModelCheckpoint(
        # monitor='val_loss_epoch',
        save_last=True,
        dirpath=os.path.join(dir_path, "checkpoints"),
        filename=self.exp_name
    )]

    return logger, checkpoint_callbacks
  

      
  
  def _load_data(self):
    if self.config.data.name == 'gp':
      self.data_module = GaussianProcessDataModule(x_dim=self.config.data.x_dim,
                                                   y_dim=self.config.data.y_dim,
                                                   n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                   n_total_points=self.config.data.n_total_points,
                                                   batch_size=self.config.data.batch_size,
                                                   x_min_max=self.config.data.x_min_max,
                                                   kernel_type= self.config.data.kernel_type,
                                                   is_vary_kernel_hyp=self.config.data.is_vary_kernel_hyp,
                                                   random_n_target_points=self.config.data.random_n_target_points,
                                                   n_sample_samples=self.config.data.n_sample_samples,
                                                   train_set_size=self.config.data.train_set_size,
                                                   val_set_size=self.config.data.val_set_size,
                                                   test_set_size=self.config.data.test_set_size,
                                                   shuffle=self.config.data.shuffle, 
                                                   datadir=self.config.data.datadir,
                                                   random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'cdf':
      self.data_module = RandomGaussianMixtureCDFDataModule(x_dim=self.config.data.x_dim,
                                                            y_dim=self.config.data.y_dim,
                                                            n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                            n_total_points=self.config.data.n_total_points,
                                                            batch_size=self.config.data.batch_size,
                                                            x_min_max=self.config.data.x_min_max,
                                                            poisson_rate=self.config.data.poisson_rate,
                                                            dirichlet_concentration=self.config.data.dirichlet_concentration,
                                                            random_n_target_points=self.config.data.random_n_target_points,
                                                            train_set_size=self.config.data.train_set_size,
                                                            val_set_size=self.config.data.val_set_size,
                                                            test_set_size=self.config.data.test_set_size,
                                                            shuffle=self.config.data.shuffle, 
                                                            datadir=self.config.data.datadir,
                                                            random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'learning_curve':
      self.data_module = LearningCurveDataModule(x_dim=self.config.data.x_dim,
                                                            y_dim=self.config.data.y_dim,
                                                            n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                            n_total_points=self.config.data.n_total_points,
                                                            batch_size=self.config.data.batch_size,
                                                            x_min_max=self.config.data.x_min_max,
                                                            random_n_target_points=self.config.data.random_n_target_points,
                                                            train_set_size=self.config.data.train_set_size,
                                                            val_set_size=self.config.data.val_set_size,
                                                            test_set_size=self.config.data.test_set_size,
                                                            shuffle=self.config.data.shuffle, 
                                                            datadir=self.config.data.datadir,
                                                            random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'sde':
      self.data_module = StratonovichSDEDataModule(x_dim=self.config.data.x_dim,
                                                  y_dim=self.config.data.y_dim,
                                                  n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                  n_total_points=self.config.data.n_total_points,
                                                  batch_size=self.config.data.batch_size,
                                                  x_min_max=self.config.data.x_min_max,
                                                  random_n_target_points=self.config.data.random_n_target_points,
                                                  train_set_size=self.config.data.train_set_size,
                                                  val_set_size=self.config.data.val_set_size,
                                                  test_set_size=self.config.data.test_set_size,
                                                  shuffle=self.config.data.shuffle, 
                                                  datadir=self.config.data.datadir,
                                                  random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'monotonic':
      self.data_module = MonotonicFunctionDataModule(x_dim=self.config.data.x_dim,
                                                            y_dim=self.config.data.y_dim,
                                                            n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                            n_total_points=self.config.data.n_total_points,
                                                            batch_size=self.config.data.batch_size,
                                                            x_min_max=self.config.data.x_min_max,
                                                            random_n_target_points=self.config.data.random_n_target_points,
                                                            train_set_size=self.config.data.train_set_size,
                                                            val_set_size=self.config.data.val_set_size,
                                                            test_set_size=self.config.data.test_set_size,
                                                            shuffle=self.config.data.shuffle, 
                                                            datadir=self.config.data.datadir,
                                                            random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'convex':
      self.data_module = ConvexFunctionDataModule(x_dim=self.config.data.x_dim,
                                                  y_dim=self.config.data.y_dim,
                                                  n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                                  n_total_points=self.config.data.n_total_points,
                                                  batch_size=self.config.data.batch_size,
                                                  x_min_max=self.config.data.x_min_max,
                                                  random_n_target_points=self.config.data.random_n_target_points,
                                                  train_set_size=self.config.data.train_set_size,
                                                  val_set_size=self.config.data.val_set_size,
                                                  test_set_size=self.config.data.test_set_size,
                                                  shuffle=self.config.data.shuffle, 
                                                  datadir=self.config.data.datadir,
                                                  random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'wheel':
      self.data_module = WheelDataModule(x_dim=self.config.data.x_dim,
                                              y_dim=self.config.data.y_dim,
                                              n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                              n_total_points=self.config.data.n_total_points,
                                              batch_size=self.config.data.batch_size,
                                              random_n_target_points=self.config.data.random_n_target_points,
                                              train_set_size=self.config.data.train_set_size,
                                              val_set_size=self.config.data.val_set_size,
                                              test_set_size=self.config.data.test_set_size,
                                              shuffle=self.config.data.shuffle, 
                                              datadir=self.config.data.datadir,
                                              random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'geofluvial':
      self.config.data.n_total_points = self.config.data.resolution ** 2
      self.data_module = GeoFluvialDataModule(n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                              n_total_points=self.config.data.n_total_points,
                                              batch_size=self.config.data.batch_size,
                                              data_augmentation=self.config.data.data_augmentation,
                                              random_n_target_points=self.config.data.random_n_target_points,
                                              train_set_size=self.config.data.train_set_size,
                                              val_set_size=self.config.data.val_set_size,
                                              test_set_size=self.config.data.test_set_size,
                                              resolution=self.config.data.resolution,
                                              shuffle=self.config.data.shuffle, 
                                              datadir=self.config.data.datadir,
                                              random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'mnist':
      self.data_module = MNISTDataModule(n_context_points=self.config.data.eval_n_context_points if self.config.run.mode == 'visualize' else self.config.data.n_context_points,
                                              n_total_points=self.config.data.n_total_points,
                                              batch_size=self.config.data.batch_size,
                                              random_n_target_points=self.config.data.random_n_target_points,
                                              train_set_size=self.config.data.train_set_size,
                                              val_set_size=self.config.data.val_set_size,
                                              test_set_size=self.config.data.test_set_size,
                                              resolution=self.config.data.resolution,
                                              shuffle=self.config.data.shuffle, 
                                              datadir=self.config.data.datadir,
                                              random_seed=self.config.run.random_seed)
    else:
      raise Exception("Unknow dataset {}".format(self.config.data.name))
  
  def _create_model(self):
    if self.config.model.name == 'dfflow':
      self.model = DiscreteFunctionFlowModule(n_steps=self.config.model.n_steps, 
                                              x_dim=self.config.data.x_dim,
                                              y_dim=self.config.data.y_dim,
                                              r_dim=self.config.model.r_dim,
                                              z_dim=self.config.model.z_dim,
                                              batch_size=self.config.data.batch_size,
                                              base_flow=self.config.model.base_flow,
                                              set_encoder_type=self.config.model.set_encoder_type,
                                              n_hidden_units=self.config.model.n_hidden_units,
                                              n_hidden_units_cif=self.config.model.n_hidden_units_cif,
                                              n_hidden_layers=self.config.model.n_hidden_layers,         
                                              n_inds=self.config.model.n_inds, 
                                              has_context_set=self.config.data.has_context_set,
                                              use_fourier_features=self.config.model.use_fourier_features,
                                              num_frequencies=self.config.model.num_frequencies,
                                              rff_init_std=self.config.model.rff_init_std,
                                              training_objective=self.config.model.training_objective,
                                              n_latent_samples=self.config.model.n_latent_samples,
                                              optim_lr=self.config.train.learning_rate,
                                              weight_decay=self.config.train.weight_decay)
    elif self.config.model.name == 'np':
      self.model = NeuralProcess(x_dim=self.config.data.x_dim,
                                 y_dim=self.config.data.y_dim,
                                 r_dim=self.config.model.r_dim,
                                 z_dim=self.config.model.z_dim,
                                 batch_size=self.config.data.batch_size,
                                 n_hidden_units_enc=self.config.model.n_hidden_units_enc,
                                 n_hidden_units_dec=self.config.model.n_hidden_units_dec,
                                 n_hidden_layers=self.config.model.n_hidden_layers,
                                 n_inds=self.config.model.n_inds, 
                                 y_sigma_lb=self.config.model.y_sigma_lb,
                                 set_encoder_type=self.config.model.set_encoder_type,
                                 use_fourier_features=self.config.model.use_fourier_features,
                                 num_frequencies=self.config.model.num_frequencies,
                                 rff_init_std=self.config.model.rff_init_std,
                                 training_objective=self.config.model.training_objective,
                                 n_latent_samples=self.config.model.n_latent_samples,
                                 optim_lr=self.config.train.learning_rate,
                                 weight_decay=self.config.train.weight_decay)
    elif self.config.model.name == 'gp':
      self.model = GaussianProcess(x_dim=self.config.data.x_dim,
                                   y_dim=self.config.data.y_dim,
                                   batch_size=self.config.data.batch_size,
                                   initial_noise_std=self.config.model.initial_noise_std,
                                   kernel_cls=self.config.model.kernel_cls,
                                   is_oracle=self.config.model.is_oracle,
                                   optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'copula':
      self.model = GaussianCopulaProcess(x_dim=self.config.data.x_dim,
                                   y_dim=self.config.data.y_dim,
                                   n_hidden_layers=self.config.model.n_hidden_layers,
                                   n_hidden_units=self.config.model.n_hidden_units,
                                   n_flow_layers=self.config.model.n_flow_layers,
                                   batch_size=self.config.data.batch_size,
                                   initial_noise_std=self.config.model.initial_noise_std,
                                   kernel_cls=self.config.model.kernel_cls,
                                   optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'bruno':
      self.model = BRUNO(x_dim=self.config.data.x_dim,
                         y_dim=self.config.data.y_dim,
                         batch_size=self.config.data.batch_size,
                         optim_lr=self.config.train.learning_rate)
    else:
      raise Exception("Unknow model {}".format(self.config.model.name))
  
  
  def run_training(self):
    if self.config.run.restore:
       self._restore_model()
    self.trainer.fit(self.model, self.data_module)
  
  def run_evaluation(self):
    if os.path.exists(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt")):
      self._restore_model()
    else:
      logging.info("No checkpoint found, using untrained model")
    self.data_module.setup()
    if self.config.eval.which_set == 'val':
      dataloader = self.data_module.val_dataloader()
      num_examples = len(self.data_module.val_set)
    elif self.config.eval.which_set == 'test':
      dataloader = self.data_module.test_dataloader()
      num_examples = len(self.data_module.test_set)
      
    results = self.trainer.test(self.model, dataloaders=dataloader, verbose=False)
    logging.info("-------------------------------------")
    logging.info("EXP Name: " + self.exp_name)
    logging.info("Evaluation on {0} set, {1} examples".format(self.config.eval.which_set, num_examples))
    logging.info(str(results))
      
    return results
  
  def run_data_visualization(self):
    self.data_module.setup()
    dataloader = self.data_module.vis_dataloader()
    X_c, Y_c, X_t, Y_t = next(iter(dataloader))
    self.data_module.visualize(save_to="{0}/data_vis_{1}".format(self.exp_name, self.config.data.name))
  
  def run_visualization(self):
    if os.path.exists(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt")):
      self._restore_model()
    else:
      logging.info("No checkpoint found, using untrained model")
    self.data_module.setup()
    self.model.to(self.device)
    dataloader = self.data_module.vis_dataloader()
    logging.info("Run Visualization ...") 
    dataloader_iter = iter(dataloader)
    for k in range(10):
      logging.info("Generate {}-th batch ...".format(k)) 
      X_c, Y_c, X_t, Y_t = next(dataloader_iter)
      self.model.visualize(X_t, Y_t, X_c, Y_c, 
                          save_to="{0}/vis_{1}_{2}".format(self.exp_name, self.config.model.name, k), 
                          resolution=None if 'resolution' not in self.config.data else  self.config.data.resolution)
    
  def run_debug(self):
    if os.path.exists(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt")):
      self._restore_model()
    else:
      logging.info("No checkpoint found, using untrained model")
    self.data_module.setup()
    dataloader = self.data_module.vis_dataloader()
    X_c, Y_c, X_t, Y_t = next(iter(dataloader))
    
    index = torch.argwhere(Y_t[0, :, 0] < 0.0)[..., 0]
    indices = index[torch.randperm(index.shape[0])[:25]].to(self.device)  # index[:25].to(self.device)  #torch.randperm(768)[:25].to(self.device)
    X_c, Y_c, X_t, Y_t = X_c.to(self.device), Y_c.to(self.device), X_t.to(self.device), Y_t.to(self.device)
    X_t = torch.index_select(X_t, dim=1, index=indices)
    Y_t = torch.index_select(Y_t, dim=1, index=indices)
    self.model.to(self.device)
    
    X_t_ff, X_c_ff, Y_c_ff = X_t, X_c, Y_c
    if self.config.model.use_fourier_features:
      X_c_ff = self.model.fourier_features(X_c_ff)
      X_t_ff = self.model.fourier_features(X_t_ff)

    z_seq = self.model.infer_net.sample(X_c_ff, Y_c_ff)
    # prior = self.infer(X_c, Y_c)
    # z = self._sample_with_reparameterization(prior)
    
    log_probs = []
    # for i in np.linspace(-0.6, 0.6, num=1001):
    for i in np.linspace(-1.1, 1.1, num=1001):
      _Y_t = i * torch.ones_like(Y_t)
      log_prob = self.model.cond_cif.log_prob(z_seq, X_t_ff, _Y_t, aggregate=False)[0]
      # log_prob = self.model.cond_cif.log_prob(z, X_t, _Y_t, aggregate=False)[0]
      log_prob = log_prob.detach().cpu().numpy()
      log_probs.append(log_prob)
    log_prob = np.stack(log_probs)
    
    log_prob_gt = self.model.cond_cif.log_prob(z_seq, X_t_ff, Y_t, aggregate=False)[0].detach().cpu().numpy()
    Y_t = Y_t.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(5*5,5*5))
    
    for i in range(5):
      for j in range(5):
        ax = fig.add_subplot(5,5,i*5+j+1)
        sns.lineplot(x=np.linspace(-1.1, 1.1, num=1001), y=log_prob[:, i*5+j], ax=ax)
        ax.scatter(x=Y_t[:, i*5+j, 0], y=log_prob_gt[i*5+j])
        
    fig.savefig("dfflow_out_geofluvia_negative.png")
    # quit()
    
  def infer_hyperparameters(self):
    self._restore_model()
    self.data_module.setup()
    dataloader = self.data_module.val_dataloader()
    X_c, Y_c, X_t, Y_t = next(iter(dataloader))
    logging.info("Infer Hyperparameters ...") 
    Y_t_pred, _ = self.model(X_t, X_c, Y_c)
    hyps = infer_optimal_gp_hyperparameters(self.data_module.kernel, X_t, Y_t)
    for hyp in hyps:
      logging.info(str(hyp))
  
  def _restore_model(self):
    if self.config.model.name == 'dfflow':
      self.model = DiscreteFunctionFlowModule.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                              n_steps=self.config.model.n_steps, 
                                              x_dim=self.config.data.x_dim,
                                              y_dim=self.config.data.y_dim,
                                              r_dim=self.config.model.r_dim,
                                              z_dim=self.config.model.z_dim,
                                              batch_size=self.config.data.batch_size,
                                              base_flow=self.config.model.base_flow,
                                              set_encoder_type=self.config.model.set_encoder_type,
                                              n_hidden_units=self.config.model.n_hidden_units,
                                              n_hidden_units_cif=self.config.model.n_hidden_units_cif,
                                              n_hidden_layers=self.config.model.n_hidden_layers,          
                                              n_inds=self.config.model.n_inds, 
                                              has_context_set=self.config.data.has_context_set,
                                              use_fourier_features=self.config.model.use_fourier_features,
                                              num_frequencies=self.config.model.num_frequencies,
                                              rff_init_std=self.config.model.rff_init_std,
                                              training_objective=self.config.model.training_objective,
                                              n_latent_samples=self.config.model.n_latent_samples,
                                              optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'np':
      self.model = NeuralProcess.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                      x_dim=self.config.data.x_dim,
                                      y_dim=self.config.data.y_dim,
                                      r_dim=self.config.model.r_dim,
                                      z_dim=self.config.model.z_dim,
                                      batch_size=self.config.data.batch_size,
                                      n_hidden_units_enc=self.config.model.n_hidden_units_enc,
                                      n_hidden_units_dec=self.config.model.n_hidden_units_dec,
                                      n_hidden_layers=self.config.model.n_hidden_layers,
                                      n_inds=self.config.model.n_inds, 
                                      y_sigma_lb=self.config.model.y_sigma_lb,
                                      set_encoder_type=self.config.model.set_encoder_type,
                                      use_fourier_features=self.config.model.use_fourier_features,
                                      num_frequencies=self.config.model.num_frequencies,
                                      rff_init_std=self.config.model.rff_init_std,
                                      training_objective=self.config.model.training_objective,
                                      n_latent_samples=self.config.model.n_latent_samples,
                                      optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'gp':
      self.model = GaussianProcess.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                   x_dim=self.config.data.x_dim,
                                   y_dim=self.config.data.y_dim,
                                   batch_size=self.config.data.batch_size,
                                   initial_noise_std=self.config.model.initial_noise_std,
                                   kernel_cls=self.config.model.kernel_cls,
                                   is_oracle=self.config.model.is_oracle,
                                   optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'copula':
      self.model = GaussianCopulaProcess.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                   x_dim=self.config.data.x_dim,
                                   y_dim=self.config.data.y_dim,
                                   n_hidden_layers=self.config.model.n_hidden_layers,
                                   n_hidden_units=self.config.model.n_hidden_units,
                                   n_flow_layers=self.config.model.n_flow_layers,
                                   batch_size=self.config.data.batch_size,
                                   initial_noise_std=self.config.model.initial_noise_std,
                                   kernel_cls=self.config.model.kernel_cls,
                                   optim_lr=self.config.train.learning_rate)
    elif self.config.model.name == 'bruno':
      self.model = BRUNO.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                   x_dim=self.config.data.x_dim,
                                   y_dim=self.config.data.y_dim,
                                   batch_size=self.config.data.batch_size,
                                   optim_lr=self.config.train.learning_rate)                                            
    else:
      raise Exception("Unknow model {}".format(self.config.model.name))
    logging.info("Restore from {}".format(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt")))
  
  def _restore_session(self):
    pass 
  
  
