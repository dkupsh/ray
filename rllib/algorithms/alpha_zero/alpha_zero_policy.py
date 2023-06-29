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
        # Specific MCTS Settings
        self.env = env_creator(config['env_config'])
        self.mcts =  MCTS(self.model, config["mcts_config"])
        
        self.view_requirements[SampleBatch.VF_PREDS] = ViewRequirement(
            space=gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            used_for_compute_actions=False
        )
        
        self.view_requirements['mcts_policies'] = ViewRequirement(
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
        model_out, _ = model(input_dict, None, [1])
        logits = torch.squeeze(model_out)
        priors = nn.Softmax(dim=-1)(logits)
        
        '''
        action_dist = dist_class(model_out, model)
        actions = train_batch[SampleBatch.ACTIONS]
        logprobs = action_dist.logp(actions)
        '''
        if priors.shape != train_batch["mcts_policies"].shape:
            raise ValueError("logprobs and mcts_policies must have the same shape", priors.shape, train_batch["mcts_policies"].shape)
        
        # Compute Policy Loss
        policy_loss = torch.mean(
            -torch.sum(train_batch["mcts_policies"] * torch.log(priors), dim=-1)
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
        episodes = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(input_dict)
            input_dict.set_training(True)
            
            if episodes is None:
                raise ValueError("Episodes must be passed in for AlphaZeroPolicy. Got None")
            
            actions = []
            for i, episode in enumerate(episodes):
                if episode.length == 0:
                    env_state = episode.user_data["initial_state"]
                    
                    # create tree root node
                    obs = self.env.set_state(env_state)
                    tree_node = Node(
                        state=env_state,
                        obs=obs,
                        reward=0,
                        done=False,
                        action=None,
                        parent=RootParentNode(env=self.env),
                        mcts=self.mcts,
                    )
                else:
                    # otherwise get last root node from previous time step
                    tree_node = episode.user_data["tree_node"]

                # run monte carlo simulations to compute the actions
                # and record the tree
                mcts_policy, action, tree_node = self.mcts.compute_action(tree_node)
                # record action
                actions.append(action)
                # store new node
                episode.user_data["tree_node"] = tree_node

                # store mcts policies vectors and current tree root node
                if episode.length == 0:
                    episode.user_data["mcts_policies"] = [mcts_policy]
                else:
                    episode.user_data["mcts_policies"].append(mcts_policy)
            
            extra_fetches = self.extra_action_out(
                input_dict, kwargs.get("state_batches", []), self.model, None
            )

            return (
                np.array(actions),
                [],
                convert_to_numpy(extra_fetches),
            )
    
    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        with torch.no_grad():
            sample_batch["mcts_policies"] = np.array(episode.user_data["mcts_policies"])[sample_batch["t"]]
            
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )