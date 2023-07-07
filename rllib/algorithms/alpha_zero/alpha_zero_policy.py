import numpy as np
from typing import Dict, List, Type, Union, Tuple, Optional

import gymnasium as gym

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.alpha_zero.mcts import Node, RootParentNode
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import ValueNetworkMixin, LearningRateSchedule
from ray.rllib.algorithms.alpha_zero.mcts import MCTS
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch
)

from ray.rllib.utils.torch_utils import (
    explained_variance,
    apply_grad_clipping
)


from ray.rllib.algorithms.alpha_zero.ranked_rewards import get_r2_env_wrapper


torch, nn = try_import_torch()

class AlphaZeroPolicy(ValueNetworkMixin, LearningRateSchedule, TorchPolicyV2):
    
    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        
        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        
        _, env_creator = Algorithm._get_env_id_and_creator(config["env"], config)
        
        if self.config['ranked_rewards'].get('enabled', True):
            self.env = get_r2_env_wrapper(env_creator, self.config['ranked_rewards'])(config['env_config'])
        else:
            self.env = env_creator(config['env_config'])
        
        # Specific MCTS Settings
        self.mcts : MCTS =  MCTS(self.model, self.env, config["mcts_config"])
        
        self.view_requirements[SampleBatch.VF_PREDS] = ViewRequirement(
            space=gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            used_for_compute_actions=False
        )
        
        self.view_requirements['mcts_policies'] = ViewRequirement(
            space=gym.spaces.Box(-np.inf, np.inf, shape=(42,), dtype=np.float32),
            used_for_compute_actions=False
        )
        
        self.view_requirements[SampleBatch.VALUES_BOOTSTRAPPED] = ViewRequirement(
            used_for_compute_actions=False
        )
        
        self.view_requirements[Postprocessing.ADVANTAGES] = ViewRequirement(
            used_for_compute_actions=False
        )
        
        self.view_requirements[Postprocessing.VALUE_TARGETS] = ViewRequirement(
            used_for_compute_actions=False
        )
    
    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        input_dict = restore_original_dimensions(
            train_batch["obs"], self.observation_space, "torch"
        )
        
        # get inputs unflattened inputs
        logits, _ = model(input_dict, None, [])
        priors = nn.LogSoftmax(dim=-1)(torch.squeeze(logits))
        
        # Compute Policy Loss
        policy_loss = torch.mean(
            -torch.sum(train_batch["mcts_policies"] * priors, dim=-1)
        )
        
        # Compute Value Loss
        value_fn_out = model.value_function()
        value_loss = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
        )
        
        if "vf_clip_param" in self.config:
            value_loss = torch.clamp(value_loss, 0, self.config["vf_clip_param"])
        value_loss = torch.mean(value_loss)
        
        # compute total loss
        total_loss = policy_loss + self.config["vf_coeff"] * value_loss
        
        # Log Stats
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = policy_loss
        model.tower_stats["mean_vf_loss"] = value_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        
        return total_loss
    
    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_lr": self.cur_lr,
                "node_expansions": self.mcts.nodes_expanded,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
            }
        )
    
    def extra_grad_process(
        self, optimizer: "torch.optim.Optimizer", loss: TensorType
    ) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def compute_actions_from_input_dict(
        self,
        input_dict: Dict[str, TensorType],
        explore: bool = None,
        timestep: Optional[int] = None,
        prior_actions: Optional[int] = [],
        episodes = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(input_dict) 
            input_dict.set_training(True)
            
            # Switch to eval mode.
            if self.model:
                self.model.eval()
            
            if episodes is None or len(episodes) == 0:
                return self._compute_action_helper(prior_actions, input_dict)
                
            if len(episodes) == 1:
                actions = episodes[0].user_data.get("actions", [])
                return self._compute_action_helper(actions, input_dict, episodes[0])
            else:
                raise ValueError("MCTS Policy can only handle one episode at a time")

    def _compute_action_helper(self, prior_actions, input_dict, episode=None, **kwargs):
        pre_expansions = self.mcts.nodes_expanded
        node = self.mcts.get_node(prior_actions)
        mcts_policy, action = self.mcts.compute_action(node)
        num_expansions = self.mcts.nodes_expanded - pre_expansions
        
        extra_fetches = self.extra_action_out(
            input_dict, kwargs.get("state_batches", []), self.model, None
        )
        extra_fetches["mcts_policies"] = mcts_policy
        #extra_fetches['num_expansions'] = np.array(num_expansions)
        
        if episode is not None:
            episode.user_data["actions"] = prior_actions + [action]
            
            # TODO: Temporary until I can figure out how to store mcts policies in extra_fetches
            prior_mcts_policies = episode.user_data.get("mcts_policies", [])
            episode.user_data["mcts_policies"] = prior_mcts_policies + [mcts_policy]
        
        return convert_to_numpy(([action], [], extra_fetches))
    
    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            if episode is not None: 
                sample_batch["mcts_policies"] = np.array(
                    episode.user_data["mcts_policies"]
                )[sample_batch["t"]]
            
            
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
        
    
    