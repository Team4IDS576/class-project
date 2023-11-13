import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from ray.tune.registry import register_env
from torch import nn

import RoadNetEnv
from NguyenNetwork import traffic

class nnModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU()
        )
        
        self.policy_fn = nn.Linear(64, num_outputs)
        self.value_fn = nn.Linear(64, 1)
        
    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"])
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state
    
    def value_function(self):
        return self._value_out
    
def env_creator(args):
    env = RoadNetEnv.parallel_env()
    return env
    
if __name__ == "__main__":
    ray.init()

    env_name = "NguyenNet"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("simpleNN", nnModel)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 500},
        checkpoint_freq=10,
        local_dir="ray_results/" + env_name,
        config=config.to_dict(),
    )

        
    