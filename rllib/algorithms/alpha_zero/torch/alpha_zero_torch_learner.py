"""Torch learner for AlphaZero.

Two losses on the standard new-API stack:

* policy: cross-entropy of the MCTS visit-count target against the
  current network's policy logits — `-sum(mcts_pi * log_softmax(logits))`.
  The mask is implicit in `mcts_pi` (illegal-action visit counts are 0)
  and the model's `forward_train` is expected to mask its own logits.
* value: MSE between the network value head and the discounted
  return-to-go target stashed on each sample.

Together: `total = policy_loss + vf_loss_coeff * value_loss`.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ray.rllib.algorithms.alpha_zero.alpha_zero import (
    AlphaZeroConfig,
    MCTS_POLICY,
    VALUE_TARGET,
)
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import TensorType


class AlphaZeroTorchLearner(TorchLearner):
    """Implements policy CE + value MSE loss on torch tensors."""

    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: str,
        config: Optional[AlphaZeroConfig] = None,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()

        # Policy loss: MCTS visit-distribution target vs network logits.
        # `fwd_out[ACTION_DIST_INPUTS]` is the model's *masked* logits.
        logits = fwd_out[Columns.ACTION_DIST_INPUTS]
        mcts_pi = batch[MCTS_POLICY]
        # Crop to a common width — defensive against the model's policy
        # head emitting fewer outputs than `max_nodes` mask width.
        action_n = min(logits.shape[-1], mcts_pi.shape[-1])
        log_probs = F.log_softmax(logits[..., :action_n], dim=-1)
        # MCTS-pi is already mask-implicit (illegal actions have 0
        # visits). Cross-entropy: -sum(target * log_pred).
        policy_loss = -(mcts_pi[..., :action_n] * log_probs).sum(dim=-1).mean()

        # Value loss: MSE against discounted return-to-go.
        value_pred = module.compute_values(
            batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
        ).reshape(-1)
        value_target = batch[VALUE_TARGET].reshape(-1)
        value_loss = F.mse_loss(value_pred, value_target)

        total_loss = policy_loss + config.vf_loss_coeff * value_loss

        # Value-head fit quality: 1 - Var(target - pred) / Var(target).
        # 0 = value head is no better than predicting the mean;
        # 1 = perfect. Negative means worse than the mean baseline,
        # which usually flags a target-scale mismatch (e.g.,
        # `value_target_scale` set wrong vs the value head's clamp).
        # Same key PPO emits, so dashboards line up.
        vf_explained = explained_variance(value_target, value_pred.detach())

        # Log per-module loss components — surfaced in `train()` result.
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: policy_loss,
                VF_LOSS_KEY: value_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: vf_explained,
            },
            key=module_id,
            window=1,
        )
        return total_loss
