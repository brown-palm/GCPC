import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from .policylm import PolicyLM

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, goal_dim: int, config: DictConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.embed_dim = config.embed_dim

        self.obs_goal_embed = nn.Linear(self.obs_dim + self.goal_dim, self.embed_dim)
        self.hidden_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.action_decoder = nn.Linear(self.embed_dim, self.action_dim)
        self.dropout_activation = nn.Sequential(nn.ReLU(), nn.Dropout(config.pdrop))

        self.mlp_policy = nn.Sequential(
            self.obs_goal_embed,
            self.dropout_activation,
            self.hidden_layer,
            self.dropout_activation,
            self.action_decoder
        )

    def forward(self, observations: Tensor, goal: Tensor):
        '''
        obs_mask: boolean tensor, True means masked
        '''
        obs_goal = torch.cat([observations, goal], dim=-1)
        pred_a = self.mlp_policy(obs_goal)

        return pred_a
    

class MLPPolicyLM(PolicyLM):
    def __init__(self, seed: int, env_name: str, obs_dim: int, action_dim: int, goal_dim: int, lr: float, use_scheduler: bool, epochs: int, ctx_size: int, future_horizon: int, num_eval_rollouts: int, eval_last_k: int, goal_type: str, model_config: DictConfig, **kwargs):
        super().__init__(seed, env_name, obs_dim, action_dim, goal_dim, lr, use_scheduler, epochs, ctx_size, future_horizon, num_eval_rollouts, eval_last_k, goal_type, model_config)

        self.model = MLPPolicy(obs_dim, action_dim, goal_dim, model_config)
        
        self.save_hyperparameters()
    
    def forward(self, observations: Tensor, goal: Tensor):
        return self.model.forward(observations, goal)
    
    def loss(self, target_a: Tensor, pred_a: Tensor):
        loss = F.mse_loss(pred_a, target_a)

        return loss
    
    def training_step(self, batch, batch_idx):
        observations, actions, goal, _, _ = batch
        
        pred_a = self(observations, goal)
        loss = self.loss(actions, pred_a)

        self.log_dict({
            'train/train_loss': loss
            },  
        sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        observations, actions, goal, _, _ = batch

        pred_a = self(observations, goal)
        loss = self.loss(actions, pred_a)

        self.log_dict({
            'val/val_loss': loss
            },  
        sync_dist=True)
    
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=self.num_epochs // 2)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
            }
        return {
                'optimizer': optimizer,
        }

    def init_eval(self, ini_obs: Tensor, goal: Tensor, obs_dim: int, action_dim: int, goal_dim: int):
        # ini_obs: (obs_dim, )
        observations = ini_obs.view(1, -1, self.model.obs_dim)
        # goal: (goal_dim, )
        goal = goal.view(1, -1, goal_dim)
        actions = None
        return observations, actions, goal

    def ar_step(self, timestep: int, observations: Tensor, actions: Tensor, goal: Tensor):
        pred_a = self(observations, goal)
        action = pred_a[0, 0]
        return action
    
    def ar_step_end(self, timestep: int, next_obs: Tensor, action: Tensor, obs_seq: Tensor, action_seq: Tensor):
        actions = None
        observations = next_obs.view(1, 1, self.model.obs_dim)
        return observations, actions

