"""AlphaZero / MapZero — MCTS self-play with the new RLlib Learner pipeline.

Pipeline:

1. **Self-play** runs on the driver process. The driver holds an
   RLModule copy (built from the user's `RLModuleSpec`) that supplies
   PUCT priors and value estimates to MCTS. The env is `MCTSSchedule`
   (or any env that exposes `get_state`/`set_state` for rollback).

2. **Training** uses the standard `LearnerGroup` + custom
   `AlphaZeroTorchLearner`. After each iteration the new module
   weights are pulled out of the learner group and copied back onto
   the driver-side module so the next self-play iteration uses an
   up-to-date policy.

Why not use `EnvRunner` for sampling? AlphaZero does
`num_simulations` model forward passes *per* env step with env state
rollback between them — that loop doesn't fit `EnvRunner.sample()`
cleanly. Driving self-play directly from `training_step` is shorter
and far easier to reason about; the SGD half still benefits from the
new pipeline (Learner, RLModule, connector, checkpointing).
"""

import asyncio
from collections import deque
from typing import Any, Optional, Type, Union
import logging
import time

import numpy as np
import torch
from typing_extensions import Self

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.callbacks.utils import make_callback
from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE, DEFAULT_MODULE_ID
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils import force_list
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    EPISODE_DURATION_SEC_MEAN,
    EPISODE_LEN_MAX,
    EPISODE_LEN_MEAN,
    EPISODE_LEN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)
# Algorithm + RLModule default IDs used by the lifetime counters.
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.core import DEFAULT_AGENT_ID
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from ray.rllib.algorithms.alpha_zero.mcts import MCTS, Node, RootParentNode


logger = logging.getLogger(__name__)


# Default MCTS hyperparameters; callers override via
# `AlphaZeroConfig.training(mcts_config=...)`.
_DEFAULT_MCTS_CONFIG = {
    'num_simulations': 50,
    'puct_coefficient': 1.0,
    'temperature': 1.0,
    'argmax_tree_policy': False,
    'add_dirichlet_noise': True,
    'dir_epsilon': 0.25,
    'dir_noise': 0.03,
    'dir_alpha_scale': 10.0,
    'normalize_q': True,
    'temperature_decay_moves': 20,
    'final_temperature': 0.1,
}


# Extra columns the AlphaZero learner expects on every batch.
MCTS_POLICY = 'mcts_policy'
VALUE_TARGET = 'value_target'


class AlphaZeroConfig(AlgorithmConfig):
    """Configuration for AlphaZero / MapZero.

    Example:
        config = (
            AlphaZeroConfig()
            .environment(MCTSSchedule, env_config={...})
            .framework('torch')
            .rl_module(rl_module_spec=RLModuleSpec(module_class=MapZeroModel,
                                                  model_config={...}))
            .training(
                lr=1e-4,
                num_simulations=64,
                mcts_config={'puct_coefficient': 2.0},
            )
        )
        algo = config.build()
        for _ in range(100):
            print(algo.train())
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or AlphaZero)
        # AlphaZero-specific knobs.
        self.num_simulations = 50
        self.mcts_config = dict(_DEFAULT_MCTS_CONFIG)
        # Eval-time MCTS overrides on top of `mcts_config`. Empty by
        # default; the evaluator (`mcts_eval`) forces deterministic
        # search (argmax tree policy, no Dirichlet noise) and reuses
        # `num_simulations`. Set e.g. `{'num_simulations': 256}` to
        # use a stronger search at evaluation than during self-play.
        self.eval_mcts_config: dict = {}
        self.vf_loss_coeff = 1.0
        # Scalar divisor applied to discounted return-to-go targets
        # before the value-MSE loss sees them. The network's value
        # output is multiplied by the same scale at MCTS leaves so
        # tree Q-values stay in env-reward units. Use >1.0 when raw
        # rewards push targets out of the value head's representable
        # range (e.g. when `compute_values` clamps to [-100, 100]).
        self.value_target_scale = 1.0
        # Batched MCTS: when >1, collect this many leaves per real
        # action with virtual loss, evaluate them as a single GPU
        # forward, then back up. 1 = original sequential MCTS.
        self.mcts_batch_size = 1
        # Virtual loss per pending visit. Standard AlphaZero practice
        # adds a small positive visit and a negative value to each
        # in-flight path so parallel selectors prefer different leaves.
        self.mcts_virtual_loss = 1.0
        # Parallel self-play. When >0, spawn that many Ray actors
        # (one CPU each) and dispatch one episode per actor per
        # training_step in parallel. 0 = run self-play sequentially
        # on the driver (legacy path, uses driver GPU).
        self.num_self_play_workers = 0
        # Shared GPU inference server pool. When True, spawn
        # `num_inference_servers` dedicated Ray actors that each
        # hold the model on GPU; actors ship their MCTS-leaf obs
        # batches to one of these servers for evaluation instead of
        # running a CPU model copy locally. CPU forwards at batch=32
        # are ~500x slower than GPU. The pool exists (vs a single
        # server) because each server processes requests sequentially
        # (the underlying model is not thread-safe), so N actors >
        # 1 server serialize on its queue — spread the load across K
        # servers and queue depth drops to ~N/K. Each server takes
        # `inference_server_num_gpus / num_inference_servers` GPU.
        # Only meaningful with `num_self_play_workers > 0`.
        self.use_inference_server = False
        self.num_inference_servers = 1
        # Total fraction of GPU reserved for the inference-server
        # pool (split evenly across `num_inference_servers`). The
        # driver/learner reserves the rest via `num_gpus_per_learner`.
        # With a single GPU and a small model, 0.5/0.5 fits.
        self.inference_server_num_gpus = 0.5
        # `torch.compile` the server-side inference module. Adds JIT
        # cost on the first ~few forwards (~10-30s) but per-call
        # latency drops 1.5-2× afterward. Off by default because the
        # model's stateful `_active_inst_queue` is fragile under
        # compile; opt in with care.
        self.inference_server_compile = False
        # Mixed-precision (fp16) inference. The scheduler's model
        # also enables autocast internally when `use_amp=True` (the
        # default), so this is mostly a belt-and-suspenders wrap at
        # the server level — leaving it on lets us force fp16 even
        # if a future model subclass forgets the internal autocast.
        self.inference_server_amp = True
        self.replay_buffer_size = 10_000
        self.episodes_per_iter = 1
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 4
        # Curriculum learning: a list of (iter_threshold, predicate)
        # tuples, sorted by `iter_threshold` ascending. When training
        # iteration >= `iter_threshold`, the predicate selects which
        # (ADG, DFG) listfiles the env may sample from. `None` disables
        # curriculum (env samples uniformly from the full corpus).
        #
        # Predicate forms accepted:
        #   - callable(listfile_path) -> bool
        #   - str / collection-of-str: substring(s) the listfile path
        #     must contain (e.g. '2x2_' admits all 2x2 ADGs).
        #
        # See `AlphaZero._curriculum_predicate_for_iter` for resolution.
        self.curriculum_schedule: Optional[list] = None
        # Standard knobs.
        self.lr = 1e-4
        self.gamma = 1.0
        self.train_batch_size = 1024
        self.grad_clip: Optional[float] = None

        # New API stack on; we drive self-play ourselves so there are no
        # env runners. is_online=False keeps Algorithm.setup from
        # spawning an EnvRunnerGroup we'd never use.
        self.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        self.num_env_runners = 0
        self.create_env_on_local_worker = False
        self._is_online = False

    @property
    def is_online(self) -> bool:
        return False

    @property
    def is_offline(self) -> bool:
        return False

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        num_simulations: Optional[int] = NotProvided,
        mcts_config: Optional[dict] = NotProvided,
        eval_mcts_config: Optional[dict] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        value_target_scale: Optional[float] = NotProvided,
        mcts_batch_size: Optional[int] = NotProvided,
        mcts_virtual_loss: Optional[float] = NotProvided,
        num_self_play_workers: Optional[int] = NotProvided,
        use_inference_server: Optional[bool] = NotProvided,
        num_inference_servers: Optional[int] = NotProvided,
        inference_server_num_gpus: Optional[float] = NotProvided,
        inference_server_compile: Optional[bool] = NotProvided,
        inference_server_amp: Optional[bool] = NotProvided,
        replay_buffer_size: Optional[int] = NotProvided,
        episodes_per_iter: Optional[int] = NotProvided,
        sgd_minibatch_size: Optional[int] = NotProvided,
        num_sgd_iter: Optional[int] = NotProvided,
        curriculum_schedule: Optional[list] = NotProvided,
        **kwargs,
    ) -> Self:
        super().training(**kwargs)
        if num_simulations is not NotProvided:
            self.num_simulations = num_simulations
            self.mcts_config['num_simulations'] = num_simulations
        if mcts_config is not NotProvided:
            # Merge so callers can override one knob at a time.
            self.mcts_config = {**self.mcts_config, **mcts_config}
            if 'num_simulations' in mcts_config:
                self.num_simulations = mcts_config['num_simulations']
            else:
                self.mcts_config['num_simulations'] = self.num_simulations
        if eval_mcts_config is not NotProvided:
            self.eval_mcts_config = {
                **self.eval_mcts_config, **eval_mcts_config}
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if value_target_scale is not NotProvided:
            self.value_target_scale = value_target_scale
        if mcts_batch_size is not NotProvided:
            self.mcts_batch_size = mcts_batch_size
        if mcts_virtual_loss is not NotProvided:
            self.mcts_virtual_loss = mcts_virtual_loss
        if num_self_play_workers is not NotProvided:
            self.num_self_play_workers = num_self_play_workers
        if use_inference_server is not NotProvided:
            self.use_inference_server = use_inference_server
        if num_inference_servers is not NotProvided:
            self.num_inference_servers = num_inference_servers
        if inference_server_num_gpus is not NotProvided:
            self.inference_server_num_gpus = inference_server_num_gpus
        if inference_server_compile is not NotProvided:
            self.inference_server_compile = inference_server_compile
        if inference_server_amp is not NotProvided:
            self.inference_server_amp = inference_server_amp
        if replay_buffer_size is not NotProvided:
            self.replay_buffer_size = replay_buffer_size
        if episodes_per_iter is not NotProvided:
            self.episodes_per_iter = episodes_per_iter
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if curriculum_schedule is not NotProvided:
            self.curriculum_schedule = curriculum_schedule
        return self

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type, str]:
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.alpha_zero.torch.alpha_zero_torch_learner import (
                AlphaZeroTorchLearner,
            )
            return AlphaZeroTorchLearner
        raise ValueError(
            f"Framework {self.framework_str!r} is not supported by AlphaZero "
            "— use framework='torch'.")

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpec:
        """AlphaZero has no canonical default model — the user supplies one.

        `Algorithm.setup` -> `get_multi_rl_module_spec()` unconditionally
        calls this method even when the user already provided a spec via
        `.rl_module(rl_module_spec=...)`. Return the user spec when
        present; otherwise raise a clear error.
        """
        if self._rl_module_spec is not None:
            spec = self._rl_module_spec
            # `_rl_module_spec` can be either RLModuleSpec or
            # MultiRLModuleSpec; unpack the single-agent case so the
            # caller (which expects either) doesn't recurse.
            if hasattr(spec, 'rl_module_specs') and isinstance(
                    spec.rl_module_specs, dict) and len(
                    spec.rl_module_specs) == 1:
                return next(iter(spec.rl_module_specs.values()))
            return spec
        raise ValueError(
            'AlphaZero requires an explicit RLModule. Call '
            'AlphaZeroConfig.rl_module(rl_module_spec=RLModuleSpec(...)) '
            'before .build().')

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        if self.num_simulations < 1:
            self._value_error('`num_simulations` must be >= 1.')
        if self.vf_loss_coeff < 0:
            self._value_error('`vf_loss_coeff` must be >= 0.')
        if self.value_target_scale <= 0:
            self._value_error('`value_target_scale` must be > 0.')
        if self.mcts_batch_size < 1:
            self._value_error('`mcts_batch_size` must be >= 1.')
        if self.num_self_play_workers < 0:
            self._value_error('`num_self_play_workers` must be >= 0.')


class _SelfPlayWorker:
    """Self-contained MCTS self-play unit.

    Owns one env + one RLModule + one MCTS instance + its callback
    objects. `run_episode()` rolls out one episode and returns the
    trajectory plus a reduced metrics dict.

    The driver builds one of these locally for the legacy
    sequential-self-play path (`num_self_play_workers=0`). When the
    pool is enabled, N Ray actors (`_SelfPlayActor`) each wrap one;
    the driver dispatches one `run_episode()` per actor per
    `training_step` and aggregates the returned trajectories +
    metrics. Driver and actors execute the exact same code path —
    only env+module ownership and the device differ.
    """

    def __init__(
        self,
        env,
        module,
        device,
        mcts_config: dict,
        value_target_scale: float,
        gamma: float,
        callbacks_class=None,
        callbacks_funcs: Optional[dict] = None,
        env_config: Optional[dict] = None,
        inference_server=None,
    ):
        self.env = env
        self.module = module
        self.device = device
        self.value_target_scale = float(value_target_scale)
        self.gamma = float(gamma)
        # Optional Ray actor handle to a shared GPU inference server.
        # When set, `_priors_and_value` / `_batch_priors_and_value`
        # ship the obs over RPC instead of running the local module.
        self._inference_server = inference_server

        # Materialize class-based callbacks (mirrors what an EnvRunner
        # does in its constructor). Tolerate ctors that require args
        # — those have to be wired functionally instead.
        cb_classes = force_list(callbacks_class) if callbacks_class else []
        self._callbacks_objs = []
        for cls in cb_classes:
            if isinstance(cls, type):
                try:
                    self._callbacks_objs.append(cls())
                except TypeError:
                    logger.warning(
                        'Skipped callback %r — ctor requires args.', cls)
            else:
                self._callbacks_objs.append(cls)
        # Functional callbacks: {'on_episode_end': [fn, ...], ...}
        self._callbacks_funcs = dict(callbacks_funcs or {})

        # Build the MCTS bound to this worker's inference functions.
        mcts_cfg = dict(mcts_config)
        self.mcts = MCTS(
            self._priors_and_value,
            mcts_cfg,
            batched_priors_and_value=self._batch_priors_and_value,
        )

        # Fire on_environment_created once so user callbacks can
        # inspect the env. Lifetime metrics emitted here are kept on
        # the worker but never aggregated back (purely informational).
        self._setup_metrics = MetricsLogger()
        self._fire_callback(
            'on_environment_created',
            env_runner=self,
            metrics_logger=self._setup_metrics,
            env=self.env,
            env_context=env_config,
        )

    # ----- weight + curriculum sync ----------------------------------

    def set_weights(self, state: Optional[dict]) -> None:
        """Load `state` onto the wrapped module (no-op if `state` is empty).

        When this worker delegates inference to a server, there is no
        local module to update — weights are pushed directly to the
        server by the caller. Same call is harmlessly a no-op here.
        """
        if self.module is None:
            return
        if state:
            self.module.set_state(state)
        self.module.eval()

    def set_train_specification(self, spec) -> None:
        """Forward a new train-spec to the env (driver curriculum push)."""
        target = self.env
        if hasattr(target, 'set_train_specification'):
            target.set_train_specification(spec)
        elif hasattr(target, 'env') and hasattr(
                target.env, 'set_train_specification'):
            target.env.set_train_specification(spec)

    # ----- callbacks --------------------------------------------------

    def _fire_callback(self, name: str, **kwargs) -> None:
        make_callback(
            name,
            callbacks_objects=self._callbacks_objs,
            callbacks_functions=self._callbacks_funcs.get(name),
            kwargs=kwargs,
        )

    # ----- MCTS oracle ------------------------------------------------

    def _priors_and_value(self, obs):
        # Route to the shared GPU inference server when configured.
        # Single-obs calls are wrapped in a length-1 batch since the
        # server's only entry point is `evaluate(obs_list)`.
        if self._inference_server is not None:
            return ray.get(
                self._inference_server.evaluate.remote([obs]))[0]

        torch_obs = convert_to_torch_tensor(obs, device=self.device)
        batched = {k: _add_batch_dim(v) for k, v in torch_obs.items()}
        input_batch = {Columns.OBS: batched}

        with torch.no_grad():
            fwd = self.module.forward_inference(input_batch)

        logits = fwd[Columns.ACTION_DIST_INPUTS].squeeze(0).cpu().numpy()
        # `_compute_embeddings` stashes embeddings under
        # `Columns.EMBEDDINGS` in the batch; passing them here lets
        # `compute_values` skip a second full GAT pass. ~2× speed-up
        # on the inference half of MCTS.
        try:
            value = self.module.compute_values(
                input_batch, embeddings=fwd.get(Columns.EMBEDDINGS))
            v = float(value.squeeze().cpu().item()) * self.value_target_scale
        except (AttributeError, NotImplementedError):
            v = 0.0

        mask = np.asarray(obs['action_mask'], dtype=bool)
        masked = np.where(mask, logits, -1e30)
        masked = masked - masked.max()
        exp = np.exp(masked) * mask
        s = exp.sum()
        if s > 0:
            priors = exp / s
        else:
            priors = mask.astype(np.float32) / max(mask.sum(), 1)
        return priors.astype(np.float32), v

    def _batch_priors_and_value(self, obs_list):
        if not obs_list:
            return []
        # Server path: one RPC per MCTS batch. Per-batch wall-time is
        # GPU forward (~5ms at batch=32) + Ray RPC (~1-2ms) + pickle
        # of the obs list. Massive win over per-actor CPU forwards.
        if self._inference_server is not None:
            return ray.get(self._inference_server.evaluate.remote(obs_list))

        if len(obs_list) == 1:
            p, v = self._priors_and_value(obs_list[0])
            return [(p, v)]

        try:
            stacked = _stack_obs(obs_list)
        except (ValueError, TypeError):
            return [self._priors_and_value(o) for o in obs_list]

        torch_obs = convert_to_torch_tensor(stacked, device=self.device)
        input_batch = {Columns.OBS: torch_obs}

        with torch.no_grad():
            fwd = self.module.forward_inference(input_batch)
            # Reuse the GAT embeddings stashed by `_compute_embeddings`
            # so `compute_values` doesn't re-run the encoder.
            try:
                values = self.module.compute_values(
                    input_batch,
                    embeddings=fwd.get(Columns.EMBEDDINGS),
                ).reshape(-1).cpu().numpy()
            except (AttributeError, NotImplementedError):
                values = np.zeros(len(obs_list), dtype=np.float32)

        scale = self.value_target_scale
        logits_batch = fwd[Columns.ACTION_DIST_INPUTS].cpu().numpy()

        out = []
        for i, obs in enumerate(obs_list):
            logits = logits_batch[i]
            mask = np.asarray(obs['action_mask'], dtype=bool)
            n = min(len(mask), logits.shape[-1])
            masked = np.where(mask[:n], logits[:n], -1e30)
            masked = masked - masked.max()
            exp = np.exp(masked) * mask[:n]
            s = exp.sum()
            if s > 0:
                priors = (exp / s).astype(np.float32)
            else:
                priors = (mask[:n].astype(np.float32)
                          / max(mask[:n].sum(), 1))
            if priors.shape[0] < mask.shape[0]:
                padded = np.zeros(mask.shape[0], dtype=np.float32)
                padded[:priors.shape[0]] = priors
                priors = padded
            v = float(values[i]) * scale
            out.append((priors, v))
        return out

    # ----- self-play --------------------------------------------------

    def run_episode(self) -> dict:
        """Roll out one MCTS-driven episode; return trajectory + stats + metrics.

        Returns a dict with:
            trajectory: list of {obs, mcts_pi, reward, value_target}
            episode_return / episode_length / duration_s: scalars
            metrics: reduced env-runner-scoped MetricsLogger dict — the
                caller merges these into its own `env_runner_metrics`
                so callbacks (EnvironmentLogger etc.) survive the
                actor boundary.
        """
        env_runner_metrics = MetricsLogger()
        episode_start_time = time.perf_counter()

        episode = SingleAgentEpisode(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        self._fire_callback(
            'on_episode_created',
            episode=episode, env_runner=self,
            metrics_logger=env_runner_metrics, env=self.env,
            rl_module=self.module, env_index=0,
        )

        obs, info = self.env.reset()
        episode.add_env_reset(observation=obs, infos=info)
        self._fire_callback(
            'on_episode_start',
            episode=episode, env_runner=self,
            metrics_logger=env_runner_metrics, env=self.env,
            rl_module=self.module, env_index=0,
        )

        state = self.env.get_state()
        root_parent = RootParentNode(self.env)
        node = Node(
            action=0, obs=obs, done=False, reward=0.0,
            state=state, mcts=self.mcts, parent=root_parent,
        )

        trajectory = []
        max_steps = int(self.env.action_space.n) * 4
        steps = 0

        while not node.done and steps < max_steps:
            mcts_pi, action, next_node = self.mcts.compute_action(node)

            trajectory.append({
                'obs': obs,
                MCTS_POLICY: mcts_pi.astype(np.float32),
                # The reward for leaving this state: the value target is the
                # return from here on, what the tree's backup expects of it.
                'reward': float(next_node.reward),
            })

            episode.add_env_step(
                observation=next_node.obs,
                action=int(action),
                reward=float(next_node.reward),
                infos=getattr(next_node, 'info', None) or {},
                terminated=bool(next_node.done),
                truncated=False,
                extra_model_outputs={MCTS_POLICY: mcts_pi.astype(np.float32)},
            )
            self._fire_callback(
                'on_episode_step',
                episode=episode, env_runner=self,
                metrics_logger=env_runner_metrics, env=self.env,
                rl_module=self.module, env_index=0,
            )

            obs = next_node.obs
            node = next_node
            steps += 1

        # Discounted return-to-go / `value_target_scale`. The MCTS
        # oracle multiplies the network's output back by the same
        # scale, so tree Q stays in env-reward units while the
        # value-MSE loss sees small targets.
        # A cutoff at `max_steps` is not an end: the return goes on past it.
        running = 0.0 if node.done else self._priors_and_value(node.obs)[1]
        for entry in reversed(trajectory):
            running = entry['reward'] + self.gamma * running
            entry[VALUE_TARGET] = float(running) / self.value_target_scale

        # MCTS expansions write checkpoints into the C++ backend; drop them.
        if hasattr(self.env, 'clear_checkpoints'):
            self.env.clear_checkpoints()

        self._fire_callback(
            'on_episode_end',
            episode=episode, env_runner=self,
            metrics_logger=env_runner_metrics, env=self.env,
            rl_module=self.module, env_index=0,
            prev_episode_chunks=[],
        )

        duration_s = time.perf_counter() - episode_start_time
        return {
            'trajectory': trajectory,
            'episode_return': float(episode.get_return()),
            'episode_length': int(len(episode)),
            'duration_s': duration_s,
            'metrics': env_runner_metrics.reduce(),
        }


def _stack_obs(obs_list):
    """Stack a list of obs dicts along axis 0 (batch dim).

    Handles nested dicts. Lists / tuples inside an obs (variable
    graph data) are kept as a per-batch list of length B.
    """
    if not obs_list:
        return {}

    def _is_graph_instance(x):
        return hasattr(x, 'nodes') and hasattr(x, 'edge_links')

    first = obs_list[0]
    out = {}
    for k in first:
        vals = [o[k] for o in obs_list]
        v0 = vals[0]
        if isinstance(v0, dict):
            out[k] = _stack_obs(vals)
        elif isinstance(v0, (list, tuple)) or _is_graph_instance(v0):
            out[k] = tuple(vals)
        elif isinstance(v0, np.ndarray):
            out[k] = np.stack(vals, axis=0)
        elif hasattr(v0, 'unsqueeze'):
            out[k] = torch.stack(
                [v.unsqueeze(0) if v.dim() == 0 else v for v in vals], dim=0)
        else:
            out[k] = np.asarray(vals)
    return out


def _add_batch_dim(v):
    """Add a leading batch dim to a (possibly-nested) tensor structure."""
    if isinstance(v, dict):
        return {k: _add_batch_dim(x) for k, x in v.items()}
    if hasattr(v, 'nodes') and hasattr(v, 'edge_links'):
        return v
    if isinstance(v, (tuple, list)) and not hasattr(v, '_fields'):
        return type(v)(_add_batch_dim(x) for x in v)
    if hasattr(v, 'unsqueeze'):
        return v.unsqueeze(0)
    return v


@ray.remote
class _SelfPlayActor:
    """Ray actor wrapping one `_SelfPlayWorker` on a single CPU.

    Built lazily inside `__init__` from picklable args (env class +
    config, RLModuleSpec, scalars) so the driver doesn't need to
    serialize a live env/module.
    """

    def __init__(
        self,
        env_class_or_id,
        env_config: dict,
        module_spec,
        mcts_config: dict,
        value_target_scale: float,
        gamma: float,
        callbacks_class=None,
        callbacks_funcs: Optional[dict] = None,
        inference_server=None,
    ):
        # Force single-threaded torch — Ray pins each actor to
        # `num_cpus=1`, but torch's default BLAS thread count is the
        # number of physical cores. With K actors each spawning ~N
        # threads we'd oversubscribe wildly and per-leaf inference
        # latency goes through the floor. MUST be called before any
        # torch op that touches the thread pool.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # set_num_interop_threads can only be called once per
            # process; safe to ignore on the second call.
            pass

        # Build env (mirror of Algorithm.setup's env construction).
        if callable(env_class_or_id) and not isinstance(env_class_or_id, str):
            env = env_class_or_id(env_config)
        else:
            from ray.tune.registry import _global_registry, ENV_CREATOR
            env = _global_registry.get(
                ENV_CREATOR, env_class_or_id)(env_config)

        # Model copy: only build a local CPU module when there is no
        # inference server. With the server, every leaf eval goes
        # over RPC and a local module would just waste ~100MB per
        # actor (× 20 actors = 2GB). The MCTS instance still needs a
        # `compute_priors_and_value` callable; the worker's methods
        # detect the server and route there instead.
        device = torch.device('cpu')
        if inference_server is None:
            module = module_spec.build()[DEFAULT_MODULE_ID]
            module = module.to(device).eval()
        else:
            module = None

        self.worker = _SelfPlayWorker(
            env=env,
            module=module,
            device=device,
            mcts_config=mcts_config,
            value_target_scale=value_target_scale,
            gamma=gamma,
            callbacks_class=callbacks_class,
            callbacks_funcs=callbacks_funcs,
            env_config=env_config,
            inference_server=inference_server,
        )

    def set_weights(self, state):
        self.worker.set_weights(state)

    def set_train_specification(self, spec):
        self.worker.set_train_specification(spec)

    def run_episode(self):
        return self.worker.run_episode()


@ray.remote
class _InferenceServer:
    """Shared GPU model server with cross-actor request coalescing.

    Holds one RLModule on GPU (or CPU if GPU unavailable) and serves
    batched leaf-eval requests from any number of `_SelfPlayActor`s.

    The server is an **async actor**: methods are coroutines. Many
    `evaluate(obs_list)` calls can be in-flight on the same event
    loop, where they enqueue their obs and await the next batch
    flush. A short coalesce window (`coalesce_window_s`) lets the
    server combine concurrent requests from N actors into a single
    GPU forward of effective batch = sum(per-actor batch sizes),
    turning N small forwards into one large one. Per-call latency
    goes up by at most `coalesce_window_s`, but GPU duty cycle goes
    way up (one big kernel beats N tiny ones).

    Asyncio is single-threaded by design — coroutines never run
    simultaneously — so the model's non-thread-safe stateful queues
    stay consistent. This is why we use async, not `max_concurrency>1`
    with real threads (which broke with "pop from empty list").
    """

    def __init__(
        self,
        module_spec,
        value_target_scale: float,
        prefer_gpu: bool = True,
        coalesce_window_s: float = 0.005,
        compile_module: bool = False,
        use_amp: bool = True,
    ):
        # Single-thread torch on the server: GPU work doesn't
        # benefit from many CPU threads, and we don't want this
        # process to contend with actors for cores.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        gpu_ok = prefer_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if gpu_ok else 'cpu')
        self.module = module_spec.build()[DEFAULT_MODULE_ID]
        self.module = self.module.to(self.device).eval()
        # Mixed-precision: wrap every forward in `torch.autocast`.
        # The model's `_shared_forward` already does this internally
        # (`_use_amp` defaults True under CUDA), but having a server-
        # level wrap belt-and-suspenders forces fp16 even if a model
        # subclass forgets it. CPU fallback uses fp32 — autocast
        # only kicks in on CUDA.
        self._use_amp = bool(use_amp) and gpu_ok
        if compile_module and gpu_ok:
            # `dynamic=True` because the server gets variable-size
            # obs batches (single-leaf eval up to coalesced mega-
            # batches). Compile failures fall back to the eager
            # module — better than crashing the server.
            try:
                self.module = torch.compile(self.module, dynamic=True)
                logger.info('Inference server module compiled')
            except Exception as e:
                logger.warning(
                    f'torch.compile failed, using eager: {e}')
        self.value_target_scale = float(value_target_scale)
        # Coalescing config + state.
        self._coalesce_window = float(coalesce_window_s)
        # Pending: list of (obs_list, asyncio.Future). Drained
        # together inside a single GPU forward.
        self._pending: list = []
        # One active batch task at a time — if the first request of
        # a batch arrives, it spawns the task; subsequent requests
        # in the coalesce window piggyback on the same task.
        self._batch_task = None

    async def set_weights(self, state):
        # Yield to the loop briefly so any in-flight evaluate can
        # finish on the old weights before we swap them.
        if state:
            self.module.set_state(state)
        self.module.eval()

    async def evaluate(self, obs_list):
        """Async entry point: enqueue and await the next batch flush.

        Returns a list of (priors, value) pairs aligned with `obs_list`.
        """
        if not obs_list:
            return []
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending.append((obs_list, fut))
        # Kick off a coalescing task if none is running.
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._coalesce_and_flush())
        return await fut

    async def _coalesce_and_flush(self):
        """Wait `coalesce_window_s`, then process all pending requests.

        Runs as a single asyncio task. Multiple `evaluate` calls
        arriving within the window all land in `self._pending`; we
        do exactly one GPU forward on the combined batch and slice
        the results back to each caller's future.
        """
        # Give peers a moment to enqueue too.
        await asyncio.sleep(self._coalesce_window)
        # Snapshot + reset the queue atomically (asyncio is
        # single-threaded so list assignment is atomic w.r.t.
        # other coroutines).
        batch = self._pending
        self._pending = []
        if not batch:
            return

        # Combine every caller's obs into one flat list. Remember
        # each caller's length so we can split results back.
        all_obs = []
        lengths = []
        for obs_list, _ in batch:
            all_obs.extend(obs_list)
            lengths.append(len(obs_list))

        try:
            results = self._evaluate_sync(all_obs)
        except Exception as e:
            # Fail every awaiting future with the same exception.
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
            return

        idx = 0
        for length, (_, fut) in zip(lengths, batch):
            if not fut.done():
                fut.set_result(results[idx:idx + length])
            idx += length

    def _evaluate_sync(self, obs_list):
        """Synchronous GPU forward over an arbitrary obs_list size.

        Same masking/scale logic as `_SelfPlayWorker._batch_priors_and_value`,
        operates on whatever obs the coalescer hands it (potentially
        hundreds of obs from many actors).
        """
        if not obs_list:
            return []

        try:
            stacked = _stack_obs(obs_list)
        except (ValueError, TypeError):
            return [self._evaluate_single(o) for o in obs_list]

        torch_obs = convert_to_torch_tensor(stacked, device=self.device)
        input_batch = {Columns.OBS: torch_obs}

        with torch.no_grad(), torch.amp.autocast(
                'cuda', enabled=self._use_amp):
            fwd = self.module.forward_inference(input_batch)
            try:
                values = self.module.compute_values(
                    input_batch,
                    embeddings=fwd.get(Columns.EMBEDDINGS),
                ).reshape(-1).cpu().numpy()
            except (AttributeError, NotImplementedError):
                values = np.zeros(len(obs_list), dtype=np.float32)

        scale = self.value_target_scale
        logits_batch = fwd[Columns.ACTION_DIST_INPUTS].cpu().numpy()
        if logits_batch.ndim == 1:
            logits_batch = logits_batch[None, :]

        out = []
        for i, obs in enumerate(obs_list):
            logits = logits_batch[i]
            mask = np.asarray(obs['action_mask'], dtype=bool)
            n = min(len(mask), logits.shape[-1])
            masked = np.where(mask[:n], logits[:n], -1e30)
            masked = masked - masked.max()
            exp = np.exp(masked) * mask[:n]
            s = exp.sum()
            if s > 0:
                priors = (exp / s).astype(np.float32)
            else:
                priors = (mask[:n].astype(np.float32)
                          / max(mask[:n].sum(), 1))
            if priors.shape[0] < mask.shape[0]:
                padded = np.zeros(mask.shape[0], dtype=np.float32)
                padded[:priors.shape[0]] = priors
                priors = padded
            v = float(values[i]) * scale
            out.append((priors, v))
        return out

    def _evaluate_single(self, obs):
        """Fallback for obs that can't be stacked with peers."""
        torch_obs = convert_to_torch_tensor(obs, device=self.device)
        batched = {k: _add_batch_dim(v) for k, v in torch_obs.items()}
        input_batch = {Columns.OBS: batched}
        with torch.no_grad(), torch.amp.autocast(
                'cuda', enabled=self._use_amp):
            fwd = self.module.forward_inference(input_batch)
            logits = fwd[Columns.ACTION_DIST_INPUTS].squeeze(0).cpu().numpy()
            try:
                value = self.module.compute_values(
                    input_batch, embeddings=fwd.get(Columns.EMBEDDINGS))
                v = float(value.squeeze().cpu().item()) * self.value_target_scale
            except (AttributeError, NotImplementedError):
                v = 0.0
        mask = np.asarray(obs['action_mask'], dtype=bool)
        masked = np.where(mask, logits, -1e30)
        masked = masked - masked.max()
        exp = np.exp(masked) * mask
        s = exp.sum()
        if s > 0:
            priors = (exp / s).astype(np.float32)
        else:
            priors = mask.astype(np.float32) / max(mask.sum(), 1)
        return priors, v


class AlphaZero(Algorithm):
    """AlphaZero / MapZero algorithm — driver-side self-play, new-API-stack SGD."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlphaZeroConfig:
        return AlphaZeroConfig()

    @classmethod
    @override(Algorithm)
    def _should_create_evaluation_env_runners(cls, eval_config) -> bool:
        """AlphaZero evaluates on the driver — no eval EnvRunners needed.

        Returning False here keeps `Algorithm.setup` from spinning up
        an `eval_env_runner_group` whose only job would be to sit idle
        (our `evaluate()` override never touches it). Otherwise the
        local eval EnvRunner alone tries to vectorize MCTSSchedule and
        then `sync_env_runner_states` chokes on the missing sampled-step
        count, since we don't actually sample from it.
        """
        return False

    @classmethod
    @override(Algorithm)
    def default_resource_request(cls, config):
        """Tally `num_self_play_workers` CPUs into Tune's reservation.

        The actor pool is spawned inside `setup()`, so the base
        `Algorithm.default_resource_request` doesn't see it — Tune
        ends up reporting "1/56 CPUs" while we secretly hold 1 + N.
        Adding one CPU bundle per actor here makes Tune's display
        accurate and, more importantly, makes multi-trial sweeps
        safe (otherwise each trial advertises 1 CPU and oversubscribes).
        """
        from ray.tune.execution.placement_groups import PlacementGroupFactory

        base_pgf = super().default_resource_request(config)

        # `config` may arrive as a dict (legacy) or an AlgorithmConfig.
        if isinstance(config, dict):
            resolved = cls.get_default_config().update_from_dict(config)
        else:
            resolved = config
        n_workers = int(getattr(resolved, 'num_self_play_workers', 0))
        use_server = bool(getattr(resolved, 'use_inference_server', False))
        server_gpu_total = float(getattr(
            resolved, 'inference_server_num_gpus', 0.0))
        n_servers = max(1, int(getattr(
            resolved, 'num_inference_servers', 1)))
        if n_workers <= 0 and not use_server:
            return base_pgf

        extra_bundles = []
        # One bundle per self-play actor. CPU-only.
        extra_bundles += [{'CPU': 1.0} for _ in range(n_workers)]
        # One bundle per inference server. Fractional GPU split N
        # ways from `inference_server_num_gpus`; they co-reside on
        # the same physical GPU as the learner via Ray PACK.
        if use_server and n_workers > 0:
            gpu_per_server = server_gpu_total / n_servers
            extra_bundles += [
                {'CPU': 1.0, 'GPU': gpu_per_server}
                for _ in range(n_servers)
            ]
        return PlacementGroupFactory(
            bundles=base_pgf.bundles + extra_bundles,
            strategy=base_pgf.strategy,
        )

    @override(Algorithm)
    def setup(self, config: AlgorithmConfig) -> None:
        # Initialize before `super().setup()` because the base setup
        # invokes `_sync_driver_module` (which checks
        # `self._self_play_actors`) before our own setup body runs.
        self._self_play_actors: list = []
        self._inference_servers: list = []

        # Inherit the standard setup — builds `learner_group` (since
        # `enable_rl_module_and_learner=True`), skips the env runners
        # because `is_online=False`.
        super().setup(config)

        # Build a driver-local env for self-play.
        env_id_or_class = self.config.env
        if env_id_or_class is None:
            raise ValueError(
                'AlphaZeroConfig.environment(env=...) must be set before build().')
        if callable(env_id_or_class) and not isinstance(env_id_or_class, str):
            self._driver_env = env_id_or_class(self.config.env_config)
        else:
            from ray.tune.registry import _global_registry, ENV_CREATOR
            self._driver_env = _global_registry.get(
                ENV_CREATOR, env_id_or_class)(self.config.env_config)

        # Build a driver-side RLModule copy for MCTS inference. Building
        # via the spec keeps the constructor path identical to what the
        # Learner used, so the state-dict shapes line up for sync.
        module_spec = self.config.get_multi_rl_module_spec(
            spaces={
                DEFAULT_MODULE_ID: (
                    self._driver_env.observation_space,
                    self._driver_env.action_space,
                ),
            },
            inference_only=True,
        )
        # Stash the spec so the actor-pool spawner can rebuild module
        # copies on each worker without re-deriving spaces.
        self._module_spec = module_spec
        self._driver_module = module_spec.build()[DEFAULT_MODULE_ID]
        self._driver_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._driver_module = self._driver_module.to(self._driver_device)
        self._driver_module.eval()
        self._sync_driver_module()

        # MCTS hyperparameters: merge config defaults with the
        # `num_simulations` lifted to the top level. `batch_size > 1`
        # routes leaf evals through `_batch_priors_and_value` (one GPU
        # forward per K leaves) instead of the sequential single-leaf
        # path; `virtual_loss` keeps parallel selectors from picking
        # the same leaf during a batch.
        mcts_cfg = dict(self.config.mcts_config)
        mcts_cfg['num_simulations'] = self.config.num_simulations
        mcts_cfg['batch_size'] = self.config.mcts_batch_size
        mcts_cfg['virtual_loss'] = self.config.mcts_virtual_loss
        mcts_cfg['gamma'] = self.config.gamma
        self._mcts_cfg = mcts_cfg

        # Collect functional callbacks once (config.callbacks_<name>
        # lists). We snapshot them so `_SelfPlayWorker` can replay the
        # same `make_callback` invocations regardless of where it
        # runs (driver or actor).
        self._callbacks_funcs_snapshot = {
            name: getattr(self.config, f'callbacks_{name}', None)
            for name in (
                'on_environment_created',
                'on_episode_created',
                'on_episode_start',
                'on_episode_step',
                'on_episode_end',
            )
        }

        # Driver-side worker. Owns the (env, module, mcts, callbacks)
        # tuple used when `num_self_play_workers=0` (legacy path).
        # When the actor pool is enabled we keep this around for
        # eval and for the on_environment_created lifecycle hook.
        self._driver_worker = _SelfPlayWorker(
            env=self._driver_env,
            module=self._driver_module,
            device=self._driver_device,
            mcts_config=mcts_cfg,
            value_target_scale=self.config.value_target_scale,
            gamma=self.config.gamma,
            callbacks_class=self.config.callbacks_class,
            callbacks_funcs=self._callbacks_funcs_snapshot,
            env_config=self.config.env_config,
        )
        # Backward-compat aliases.
        self.mcts = self._driver_worker.mcts
        self._episode_callbacks = self._driver_worker._callbacks_objs

        # Replay buffer — bounded FIFO so old (stale-policy) MCTS targets
        # don't dominate as training progresses.
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)

        # Curriculum bookkeeping. Snapshot the full (ADG, DFG) corpus at
        # setup time so we can restrict it per-iteration without losing
        # the original list — env.set_train_specification is destructive.
        # The current iteration index is read off `self.iteration` (set
        # by Algorithm.train()), so we don't need our own counter.
        self._curriculum_full_corpus: dict = dict(
            self._driver_env.env._train_groups)
        self._curriculum_active_idx: int = -1

        # Parallel self-play actor pool. Each actor runs one episode
        # per training_step on its own CPU; if `use_inference_server`
        # is set, leaf evaluations are RPC'd to a shared GPU server
        # (much faster than per-actor CPU forwards). The server is
        # spawned first so we can hand its handle to the actors.
        # (Both lists already initialized at top of setup so the
        # base `_sync_driver_module` call inside super().setup() is
        # safe.)
        if (self.config.num_self_play_workers > 0
                and self.config.use_inference_server):
            self._spawn_inference_servers(self._module_spec)
        if self.config.num_self_play_workers > 0:
            self._spawn_self_play_actors(self._module_spec)

        # Force a first-iter update so iter=0 lands on the right phase.
        self._apply_curriculum_for_iter(0)

        # Lifetime accumulators used to compute throughput in
        # `training_step`. Episodes themselves get logged through the
        # MetricsLogger so reduce/window semantics match PPO.
        self._lifetime_env_steps = 0
        self._lifetime_episodes = 0

    # ----- callbacks ------------------------------------------------

    def _fire_callback(self, name: str, **kwargs) -> None:
        """Fire both class-based and functional callbacks for `name`.

        Mirrors `SingleAgentEnvRunner._make_on_episode_callback`. The
        functional list lives at `self.config.callbacks_<name>` (e.g.
        `callbacks_on_episode_end`); we pull it via `getattr` so a
        missing list falls through cleanly.
        """
        make_callback(
            name,
            callbacks_objects=self._episode_callbacks,
            callbacks_functions=getattr(self.config, f'callbacks_{name}', None),
            kwargs=kwargs,
        )

    # ----- self-play actor pool -------------------------------------

    def _spawn_inference_servers(self, module_spec) -> None:
        """Create the shared GPU inference server pool.

        Each server is a Ray actor with its own process + GPU model
        copy. Actors are statically assigned one server (round-robin
        by index) and stick with it for the run. N>1 servers exist
        because each one processes requests sequentially (the model
        is not thread-safe), so 20 actors on 1 server serialize on
        its queue — splitting across K servers cuts queue depth ~K×.
        """
        total_gpu = float(self.config.inference_server_num_gpus)
        n_servers = max(1, int(self.config.num_inference_servers))
        gpu_per = total_gpu / n_servers
        prefer_gpu = gpu_per > 0.0

        self._inference_servers = [
            _InferenceServer.options(
                num_cpus=1,
                num_gpus=gpu_per,
            ).remote(
                module_spec=module_spec,
                value_target_scale=self.config.value_target_scale,
                prefer_gpu=prefer_gpu,
                compile_module=self.config.inference_server_compile,
                use_amp=self.config.inference_server_amp,
            )
            for _ in range(n_servers)
        ]
        state = self._extract_module_state()
        if state:
            for srv in self._inference_servers:
                srv.set_weights.remote(state)

    def _spawn_self_play_actors(self, module_spec) -> None:
        """Create `num_self_play_workers` `_SelfPlayActor` actors.

        Each actor is pinned to 1 CPU, builds its own env clone and a
        CPU-resident RLModule from `module_spec`, and shares no state
        with the driver other than what's pushed via `set_weights` /
        `set_train_specification`.

        Lazy initialization: we fire the actor constructors and the
        initial `set_weights` call but do NOT `ray.get` either one
        here. Ray actors process method calls in FIFO order, so the
        first `run_episode` call queued in `training_step` will wait
        for both — but `setup()` returns immediately. That keeps Tune
        from sitting in `PENDING` for several minutes while 20 actors
        sequentially build their envs + modules.
        """
        n = int(self.config.num_self_play_workers)
        env_class_or_id = self.config.env

        # Static round-robin server assignment: actor i talks to
        # server[i % len(servers)]. Even split keeps queue depths
        # balanced.
        servers = self._inference_servers
        self._self_play_actors = [
            _SelfPlayActor.options(num_cpus=1).remote(
                env_class_or_id=env_class_or_id,
                env_config=self.config.env_config,
                module_spec=module_spec,
                mcts_config=self._mcts_cfg,
                value_target_scale=self.config.value_target_scale,
                gamma=self.config.gamma,
                callbacks_class=self.config.callbacks_class,
                callbacks_funcs=self._callbacks_funcs_snapshot,
                inference_server=(servers[i % len(servers)]
                                  if servers else None),
            )
            for i in range(n)
        ]

        # Fire the initial weight sync but don't wait. The actors
        # haven't finished __init__ yet either; both the constructor
        # and this set_weights call resolve in the background. The
        # first `run_episode.remote()` in `training_step` queues
        # behind them and is the first place we actually block.
        state = self._extract_module_state()
        if state:
            for a in self._self_play_actors:
                a.set_weights.remote(state)

    def _extract_module_state(self) -> dict:
        """Pull the latest module state-dict from the learner group."""
        state = self.learner_group.get_state(
            components=COMPONENT_LEARNER + '/' + COMPONENT_RL_MODULE,
            inference_only=True,
        )[COMPONENT_LEARNER]
        if COMPONENT_RL_MODULE in state:
            return state[COMPONENT_RL_MODULE].get(DEFAULT_MODULE_ID, {})
        return state.get(DEFAULT_MODULE_ID, {})

    # ----- curriculum learning --------------------------------------

    @staticmethod
    def _normalize_predicate(predicate):
        """Coerce a curriculum predicate to a `callable(path) -> bool`.

        Accepts a callable as-is, a single string (substring match), or
        an iterable of strings (any-match).
        """
        if callable(predicate):
            return predicate
        if isinstance(predicate, str):
            needle = predicate
            return lambda path, n=needle: n in path
        try:
            needles = tuple(predicate)
        except TypeError as e:
            raise TypeError(
                f'curriculum predicate must be callable, str, or '
                f'iterable of str; got {type(predicate).__name__}') from e
        return lambda path, ns=needles: any(n in path for n in ns)

    def _curriculum_phase_index(self, iter_idx: int) -> int:
        """Largest schedule index `i` such that schedule[i][0] <= iter_idx.

        Returns -1 if no curriculum is configured or the first threshold
        is in the future (no filtering active yet).
        """
        schedule = self.config.curriculum_schedule
        if not schedule:
            return -1
        active = -1
        for i, (threshold, _) in enumerate(schedule):
            if iter_idx >= threshold:
                active = i
            else:
                break
        return active

    def _apply_curriculum_for_iter(self, iter_idx: int) -> None:
        """Refresh env's train_specification if the curriculum phase changed.

        Pushes the new spec to every self-play actor (if the pool is
        active) so all envs sample from the same corpus subset.
        """
        idx = self._curriculum_phase_index(iter_idx)
        if idx == self._curriculum_active_idx:
            return  # phase unchanged — env spec already correct
        self._curriculum_active_idx = idx

        if idx < 0:
            spec = dict(self._curriculum_full_corpus)
        else:
            predicate = self._normalize_predicate(
                self.config.curriculum_schedule[idx][1])
            spec = {}
            for adg_path, listfiles in self._curriculum_full_corpus.items():
                keep = [lf for lf in listfiles
                        if predicate(lf) or predicate(adg_path)]
                if keep:
                    spec[adg_path] = keep
            if not spec:
                # Predicate matched nothing — fall back to full corpus
                # so training doesn't deadlock on `swap_file`.
                spec = dict(self._curriculum_full_corpus)

        self._driver_env.set_train_specification(spec)
        if self._self_play_actors:
            # Fire-and-forget; the run_episode calls in this same
            # training_step queue behind set_train_specification on
            # each actor (FIFO), so they're guaranteed to use the
            # new spec without the driver round-tripping 20 times.
            for a in self._self_play_actors:
                a.set_train_specification.remote(spec)

    # ----- weight sync from learner_group → driver module -----------

    def _sync_driver_module(self) -> None:
        """Pull the latest RLModule state from `learner_group`; broadcast.

        After SGD the driver and any self-play actors all need the
        new weights. The driver gets them synchronously (its module
        is used by `mcts_eval`). For actors we fire `set_weights`
        without `ray.get` — Ray actors are FIFO, so the next iter's
        `run_episode.remote()` queues behind the set_weights and
        is guaranteed to see the new weights, but the driver doesn't
        block waiting for 20 round-trips per iter.
        """
        mod_state = self._extract_module_state()
        if mod_state:
            self._driver_module.set_state(mod_state)
        self._driver_module.eval()
        if mod_state:
            # Inference servers first — actor `evaluate.remote()`
            # calls FIFO-queue behind set_weights on each server,
            # then FIFO-queue behind each actor's set_weights too
            # (the actor's set_weights is a no-op when servers are
            # configured, but kept for symmetry).
            for srv in self._inference_servers:
                srv.set_weights.remote(mod_state)
            for a in self._self_play_actors:
                a.set_weights.remote(mod_state)

    # ----- MCTS oracle (driver-side; eval uses these) ---------------
    #
    # Kept as thin delegations to `_driver_worker` so external callers
    # (notably `mcts_eval`) keep working. The actor pool uses the same
    # code path inside each actor.

    def _priors_and_value(self, obs):
        return self._driver_worker._priors_and_value(obs)

    def _batch_priors_and_value(self, obs_list):
        return self._driver_worker._batch_priors_and_value(obs_list)

    # ----- training loop -------------------------------------------

    @override(Algorithm)
    def training_step(self) -> None:
        # 0) Update the curriculum phase if the schedule advanced.
        # `self.iteration` is set by the Algorithm base before each call.
        # No-op when `curriculum_schedule=None`.
        self._apply_curriculum_for_iter(self.iteration)

        # 1) Pipelined self-play.
        #
        # Order:
        #   a) Kick off this iter's self-play (non-blocking) using the
        #      current weights on each actor.
        #   b) Run SGD on the existing replay buffer (data from
        #      previous iter's self-play) — runs concurrently with (a)
        #      because the actors live in their own processes.
        #   c) Wait for self-play to finish, add trajectories to the
        #      buffer, log metrics.
        #   d) Sync the post-SGD weights to actors (fire-and-forget;
        #      next iter's self-play queues FIFO behind set_weights).
        #
        # Wall time per iter ≈ max(self_play, SGD) instead of sum.
        # Trade-off: actors are 1 iter stale on weights (they used
        # iter N-1's post-SGD weights for iter N's self-play). For
        # replay-buffer algos, 1-iter staleness is fine.
        #
        # First iter is special: no replay data yet, so we just
        # synchronously fire+wait, no overlap.
        env_runner_metrics = MetricsLogger()
        episode_returns: list = []
        episode_lengths: list = []
        episode_durations: list = []
        per_episode_metric_dicts: list = []

        sp_start = time.perf_counter()
        if self._self_play_actors:
            # (a) Fire this iter's self-play asynchronously.
            inflight = [a.run_episode.remote()
                        for a in self._self_play_actors]
        else:
            # Sequential driver path has no overlap opportunity —
            # the driver is the only worker.
            inflight = None
            results = [self._driver_worker.run_episode()
                       for _ in range(self.config.episodes_per_iter)]

        # (b) SGD on existing buffer data, runs in parallel with (a)
        # when actors are configured. `_maybe_run_sgd` handles the
        # `buffer >= sgd_minibatch_size` gate and weight broadcast.
        self._maybe_run_sgd()

        # (c) Wait for self-play to finish and add to buffer. Time
        # the whole fire→wait span as `env_runner_sampling_timer`
        # (matches the legacy key, preserves PPO-comparable dash):
        # with overlap this equals max(SP_time, SGD_time elapsed
        # before wait starts), so iter wall ≈ this value rather
        # than SP+SGD as in the legacy non-overlap path.
        if inflight is not None:
            results = ray.get(inflight)
        self.metrics.log_value(
            (TIMERS, ENV_RUNNER_SAMPLING_TIMER),
            time.perf_counter() - sp_start)
        for r in results:
                self.replay_buffer.extend(r['trajectory'])
                episode_returns.append(r['episode_return'])
                episode_lengths.append(r['episode_length'])
                episode_durations.append(r['duration_s'])
                if r.get('metrics'):
                    per_episode_metric_dicts.append(r['metrics'])

        # Per-episode callback metrics (one reduced dict per worker)
        # aggregate into the per-iter env-runner logger so user
        # callbacks (EnvironmentLogger etc.) survive the actor
        # boundary. Top-level keys here land at `env_runners/<key>`
        # after the final aggregate below.
        if per_episode_metric_dicts:
            env_runner_metrics.aggregate(per_episode_metric_dicts)

        # Per-iter + per-episode env-runner metrics. Mirror the set that
        # `SingleAgentEnvRunner._log_episode_metrics` /
        # `_increase_sampled_metrics` emit so Tuner / W&B see the same
        # keys as a normal PPO run. All keys go directly into
        # `env_runner_metrics` (top-level there) — the aggregate step
        # at the end nests everything under `env_runners/`.
        if episode_returns:
            win = max(
                1, int(self.config.metrics_num_episodes_for_smoothing))

            for ret, length, sec in zip(
                    episode_returns, episode_lengths, episode_durations):
                env_runner_metrics.log_value(
                    EPISODE_RETURN_MEAN, ret, window=win)
                env_runner_metrics.log_value(
                    EPISODE_LEN_MEAN, length, window=win)
                env_runner_metrics.log_value(
                    EPISODE_DURATION_SEC_MEAN, sec, window=win)
                env_runner_metrics.log_value(
                    EPISODE_RETURN_MIN, ret, reduce='min', window=win)
                env_runner_metrics.log_value(
                    EPISODE_RETURN_MAX, ret, reduce='max', window=win)
                env_runner_metrics.log_value(
                    EPISODE_LEN_MIN, length, reduce='min', window=win)
                env_runner_metrics.log_value(
                    EPISODE_LEN_MAX, length, reduce='max', window=win)
                env_runner_metrics.log_value(
                    ('agent_episode_return_mean', DEFAULT_AGENT_ID),
                    ret, window=win)
                env_runner_metrics.log_value(
                    ('module_episode_return_mean', DEFAULT_MODULE_ID),
                    ret, window=win)

            num_steps_total = int(sum(episode_lengths))
            num_episodes_total = len(episode_returns)

            env_runner_metrics.log_value(
                NUM_ENV_STEPS_SAMPLED, num_steps_total, reduce='sum')
            env_runner_metrics.log_value(
                (NUM_AGENT_STEPS_SAMPLED, DEFAULT_AGENT_ID),
                num_steps_total, reduce='sum')
            env_runner_metrics.log_value(
                (NUM_MODULE_STEPS_SAMPLED, DEFAULT_MODULE_ID),
                num_steps_total, reduce='sum')
            env_runner_metrics.log_value(
                NUM_EPISODES, num_episodes_total, reduce='sum')

            env_runner_metrics.log_value(
                NUM_ENV_STEPS_SAMPLED_LIFETIME, num_steps_total,
                reduce='sum', with_throughput=True)
            env_runner_metrics.log_value(
                (NUM_AGENT_STEPS_SAMPLED_LIFETIME, DEFAULT_AGENT_ID),
                num_steps_total, reduce='sum')
            env_runner_metrics.log_value(
                (NUM_MODULE_STEPS_SAMPLED_LIFETIME, DEFAULT_MODULE_ID),
                num_steps_total, reduce='sum')
            env_runner_metrics.log_value(
                NUM_EPISODES_LIFETIME, num_episodes_total, reduce='sum')

        # Aggregate the env-runner-scoped logger into `self.metrics`
        # under `env_runners/` — same nesting PPO produces. Whatever
        # `EnvironmentLogger` (and any other user callback) wrote at the
        # top of `env_runner_metrics` lands at `env_runners/<key>` here.
        env_runner_results = env_runner_metrics.reduce()
        if env_runner_results:
            self.metrics.aggregate([env_runner_results], key=ENV_RUNNER_RESULTS)

        # Curriculum + buffer state — useful for live-debugging runs.
        self.metrics.log_value(
            'curriculum_phase', self._curriculum_active_idx,
            reduce=None, window=1)
        self.metrics.log_value(
            'replay_buffer_size', len(self.replay_buffer),
            reduce=None, window=1)
        self.metrics.log_value(
            'mcts_num_simulations', self.mcts.num_simulations,
            reduce=None, window=1)

        # SGD already ran earlier (overlapped with self-play). The
        # bookkeeping above logs replay_buffer_size AFTER (c) added
        # the new trajectories, so downstream dashboards see the
        # post-iter buffer size, matching the legacy non-overlap
        # iteration semantics.

    def _maybe_run_sgd(self) -> None:
        """SGD step on the replay buffer + broadcast updated weights.

        Runs while self-play is in flight on actors (pipelined). Skips
        until the buffer holds at least `sgd_minibatch_size` samples.
        """
        if len(self.replay_buffer) < self.config.sgd_minibatch_size:
            return
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            for _ in range(self.config.num_sgd_iter):
                batch = self._build_batch(self.config.train_batch_size)
                learner_results = self.learner_group.update(
                    batch=batch,
                    minibatch_size=self.config.sgd_minibatch_size,
                )
        # Push learner stats (per-module policy/value loss, etc.)
        # under `learners/...` — same convention as MARWIL/PPO.
        self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)

        # Sync new weights to the driver module and (fire-and-forget)
        # to all actors + inference servers. The in-flight self-play
        # episodes for this iter still use the pre-SGD weights they
        # were given at .remote() time; the NEXT iter's actor
        # `run_episode` will FIFO-queue behind `set_weights` and pick
        # up the new weights. That's the source of the 1-iter
        # weight-staleness in the pipelined path — acceptable for
        # replay-buffer algorithms.
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self._sync_driver_module()

    @override(Algorithm)
    def evaluate(self, parallel_train_future=None):
        """AlphaZero evaluation — driver-side MCTS, no env runners.

        The base `Algorithm.evaluate` syncs weights to eval EnvRunners
        and calls their `sample()` method (pure policy rollout). For
        AlphaZero that path is both unnecessary (driver already has the
        module + env) and unfair vs. PPO (PPO's `sample()` IS its full
        inference mode; AlphaZero's `forward_inference` is just the
        prior — its real inference adds MCTS on top).

        Instead, dispatch directly to `config.custom_evaluation_function`,
        which the scheduler wires to `mcts_eval`. The function runs
        deterministic MCTS rollouts on the driver and pushes results
        into `self.metrics` under `EVALUATION_RESULTS/ENV_RUNNER_RESULTS`
        — same layout PPO produces, so dashboards line up.
        """
        custom_fn = self.config.custom_evaluation_function
        if custom_fn is None:
            raise ValueError(
                'AlphaZero.evaluate() requires `custom_evaluation_function`. '
                'Use `mcts_eval` from `scheduler.callbacks.evaluation.mcts_eval`.')

        self._before_evaluate()
        with self.metrics.log_time((TIMERS, 'evaluation_iteration_timer')):
            eval_results, env_steps, agent_steps = custom_fn(
                self, self.eval_env_runner_group,
            )
        # `_run_one_evaluation` re-wraps our return under EVALUATION_RESULTS, so
        # we only add the inner ENV_RUNNER_RESULTS layer — otherwise metrics
        # land at `evaluation/evaluation/env_runners/...`. Matches what the
        # base `Algorithm.evaluate()` returns (contents of EVALUATION_RESULTS).
        return {ENV_RUNNER_RESULTS: eval_results}

    def _build_batch(self, n: int) -> MultiAgentBatch:
        """Sample `n` entries from the replay buffer into a MultiAgentBatch.

        Each entry contributes one row in `Columns.OBS`, `mcts_policy`,
        and `value_target`. Observations are kept as-is (the model's
        forward path is responsible for batch handling).
        """
        if len(self.replay_buffer) < n:
            idx = np.random.choice(len(self.replay_buffer), size=n, replace=True)
        else:
            idx = np.random.choice(len(self.replay_buffer), size=n, replace=False)
        entries = [self.replay_buffer[i] for i in idx]

        # Stack observation dict leaves. Keep numpy throughout — the
        # learner connector / TorchLearner converts to torch as needed.
        obs_dict = self._stack_obs([e['obs'] for e in entries])
        mcts_pi = np.stack([e[MCTS_POLICY] for e in entries]).astype(np.float32)
        value_tgt = np.array([e[VALUE_TARGET] for e in entries], dtype=np.float32)

        sample_batch = SampleBatch({
            Columns.OBS: obs_dict,
            MCTS_POLICY: mcts_pi,
            VALUE_TARGET: value_tgt,
        })
        return MultiAgentBatch({DEFAULT_MODULE_ID: sample_batch}, env_steps=n)

    @staticmethod
    def _stack_obs(obs_list):
        """Stack a list of obs dicts along axis 0 (batch dim).

        Handles nested dicts. Lists / tuples inside an obs (variable
        graph data) are kept as a per-batch list of length B.
        """
        if not obs_list:
            return {}

        def _is_graph_instance(x):
            # Variable-length graph entries arrive as objects (e.g.
            # gymnasium GraphInstance) or as lists of tensors; either
            # way they don't stack to a uniform array.
            return hasattr(x, 'nodes') and hasattr(x, 'edge_links')

        first = obs_list[0]
        out = {}
        for k in first:
            vals = [o[k] for o in obs_list]
            v0 = vals[0]
            if isinstance(v0, dict):
                out[k] = AlphaZero._stack_obs(vals)
            elif isinstance(v0, (list, tuple)) or _is_graph_instance(v0):
                # Per-batch graph entries — `SchedulerLearner._process_graph_tuple`
                # expects an actual tuple, not a list.
                out[k] = tuple(vals)
            elif isinstance(v0, np.ndarray):
                out[k] = np.stack(vals, axis=0)
            elif hasattr(v0, 'unsqueeze'):
                # Already a tensor — stack along dim 0.
                out[k] = torch.stack(
                    [v.unsqueeze(0) if v.dim() == 0 else v for v in vals], dim=0)
            else:
                out[k] = np.asarray(vals)
        return out
