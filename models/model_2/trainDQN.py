import os

import ray
from gymnasium.spaces import Discrete
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env

import models.model_2.OLD_RoadNetEnvEncoded as rne
from NguyenNetwork import traffic

torch, nn = try_import_torch()

class TorchMaskedActions(DQNTorchModel):
    
    def __init__(
        self,
        obs_space: Discrete,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )
        
        obs_len = obs_space.n
        
        orig_obs_space = Discrete(28)
        
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
    
if __name__ == "__main__":
    
    traffic = traffic()
    agents = traffic["agents"]
    
    ray.init()
    
    
    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    
    def env_creator():
        env = rne.parallel_env()
        return env
    
    env_name = "NguyenNetwork"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))
    
    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[64],
            dueling=False,
            model={"custom_model": "pa_model"}
            
        )
        .multi_agent(
            policies = {agent: (None, obs_space, act_space, {}) for agent in agents},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id)
        )
        .resources(num_gpus=0)
        .debugging(
            log_level="DEBUG"
        )
        .framework(framework="torch")
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timestpes": 100
            }
        )
    )
    
    #config.environment(disable_env_checking=True)
    
    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000},
        checkpoint_freq=20,
        config=config.to_dict()
    )
    