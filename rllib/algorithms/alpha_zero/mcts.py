"""Monte Carlo Tree Search for AlphaZero-style training.

Ported from the AlphaZero implementation that lived in earlier Ray
versions (`ray.rllib.algorithms.alpha_zero.mcts`) and the in-repo
`scheduler/misc/mcts_search.py`. Generalized so it works against any
gym env that exposes `get_state` / `set_state` for rollback.

The search tree stores statistics on the *child edges* of each node
(not the children themselves) so visit counts and Q-values are
addressable by action index — the canonical PUCT formulation. Each
`Node` owns a slot in `parent.child_*` arrays for its own statistics.
"""

import math

import numpy as np

# Sentinel large negative number; logits below this are treated as
# disallowed for prior computation. Mirrors `FLOAT_MIN` in
# `ray.rllib.utils.torch_utils` without forcing torch import here.
_LARGE_NEG = -1e30


class Node:
    """One node in the PUCT search tree.

    Attributes:
        action: The action taken from `parent` to reach this node.
        env: Reference to the shared `gym.Env` (with `get_state` /
            `set_state` rollback).
        state: Opaque handle returned by `env.get_state()` at this node.
        obs: Observation dict at this node (includes `action_mask`).
        reward: Immediate reward received when stepping into this node.
        done: Whether this node is terminal.
        valid_actions: Boolean mask over the discrete action space.
        child_priors: Network prior over actions, length `action_space.n`.
        child_total_value: Sum of backed-up values per child action.
        child_number_visits: Visit count per child action.
        children: action -> Node, lazily populated on selection.
    """

    def __init__(self, action, obs, done, reward, state, mcts, parent=None,
                 info=None):
        self.env = parent.env
        self.action = action
        self.is_expanded = False
        # In batched mode a leaf may be selected but still awaiting its
        # network evaluation. Track that separately from `is_expanded`
        # so simultaneous selectors skip it instead of re-walking through
        # an un-expanded leaf and choosing its zero-prior children.
        self.is_pending_eval = False
        self.parent = parent
        # Real moves from the episode's start. Survives the detach that makes a
        # chosen child the next search's root (see `_build_tree_policy_and_action`).
        self.depth = parent.depth + 1 if isinstance(parent, Node) else 0
        self.children = {}

        self.action_space_size = self.env.action_space.n

        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32)
        self.child_priors = np.zeros(
            [self.action_space_size], dtype=np.float32)
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)
        # Virtual-loss visits keyed by child action. Applied before leaf
        # evaluation and reverted after the real value backs up.
        self.child_virtual_loss_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs
        # Info dict from the env.step that produced this node. Kept so
        # callers (e.g. AlphaZero's self-play episode builder) can feed
        # it back into `SingleAgentEpisode.add_env_step(infos=...)` —
        # callbacks like `EnvironmentLogger` read it via
        # `episode.get_infos(-1)`.
        self.info = info or {}
        self.mcts = mcts
        # Allow priors at illegal actions to be masked out.
        self.valid_actions = np.asarray(obs['action_mask'], dtype=bool)

    # --- statistics owned by `parent.child_*` slots --------------------

    @property
    def number_visits(self) -> float:
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value: float) -> None:
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self) -> float:
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value: float) -> None:
        self.parent.child_total_value[self.action] = value

    # --- PUCT scoring --------------------------------------------------

    def _effective_total_value(self) -> np.ndarray:
        # Real backed-up totals, with each in-flight visit counted as the worst
        # value this search has seen, so a pending path only looks worse
        # whatever the reward's scale and sign.
        return (self.child_total_value
                + self.child_virtual_loss_visits * self.mcts.pessimistic_q())

    def _effective_visits(self) -> np.ndarray:
        return self.child_number_visits + self.child_virtual_loss_visits

    @property
    def effective_number_visits(self) -> float:
        """This node's visit count including any in-flight virtual loss.

        Real visits live on `parent.child_number_visits[action]`; pending
        virtual losses on `parent.child_virtual_loss_visits[action]`.
        For the root (parent is `RootParentNode`) the latter is zero.
        """
        if self.parent is None:
            return float(self.number_visits)
        return float(
            self.parent.child_number_visits[self.action]
            + self.parent.child_virtual_loss_visits[self.action]
        )

    def child_Q(self) -> np.ndarray:
        # Mean action-value (Q) per child edge, including pending
        # virtual-loss contributions so concurrent selectors discount
        # paths that already have an in-flight evaluation. An unvisited
        # edge scores 0 after normalization, as in MuZero.
        visits = self._effective_visits()
        visited = visits > 0
        mean = np.divide(self._effective_total_value(), visits,
                         out=np.zeros_like(visits), where=visited)
        return np.where(visited, self.mcts.normalized_q(mean), 0.0)

    def child_U(self) -> np.ndarray:
        # Exploration bonus: c_puct * prior * sqrt(N) / (1 + n_i),
        # with N and n_i including in-flight virtual-loss visits.
        return (math.sqrt(max(self.effective_number_visits, 1))
                * self.child_priors
                / (1 + self._effective_visits()))

    def best_action(self) -> int:
        """Argmax of `Q + c * U`, restricted to legal actions."""
        score = (self.child_Q()
                 + self.mcts.c_puct * self.child_U())
        # Mask illegal actions with a large negative score.
        masked = np.where(self.valid_actions, score, _LARGE_NEG)
        return int(np.argmax(masked))

    def select(self):
        """Walk down the tree by `best_action` until a leaf is hit.

        In batched mode the returned leaf may already be pending an
        evaluation; the caller is responsible for checking
        `node.is_pending_eval` and falling back to a fresh simulation
        if the same leaf would be re-picked.
        """
        node = self
        while node.is_expanded:
            action = node.best_action()
            node = node.get_child(action)
        return node

    def expand(self, child_priors: np.ndarray) -> None:
        self.is_expanded = True
        # Mask priors to legal actions and renormalize.
        priors = np.where(self.valid_actions, child_priors, 0.0)
        s = priors.sum()
        if s > 0:
            priors = priors / s
        else:
            # Degenerate: model emitted zero mass on legal actions. Fall
            # back to uniform over legal moves so MCTS can still explore.
            legal = self.valid_actions.astype(np.float32)
            priors = legal / max(legal.sum(), 1.0)
        self.child_priors = priors

    def get_child(self, action: int) -> 'Node':
        if action not in self.children:
            # Restore env to *this* node's state, then step.
            self.env.set_state(self.state)
            obs, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.env.get_state()
            done = bool(terminated or truncated)
            self.children[action] = Node(
                action=action,
                obs=obs,
                done=done,
                reward=reward,
                state=next_state,
                mcts=self.mcts,
                parent=self,
                info=info,
            )
        return self.children[action]

    def backup(self, value: float) -> None:
        """Walk parents from leaf to root, accumulating visit/value. `value` is
        the return from this node on, excluding the reward for arriving here;
        each edge's Q is that reward plus the discounted value beyond it."""
        node = self
        while node.parent is not None:
            value = node.reward + self.mcts.gamma * value
            node.number_visits += 1
            node.total_value += value
            # The root's own edge scores nothing, and its arrival reward (any
            # size, from a move already made) must not set the scale.
            if not isinstance(node.parent, RootParentNode):
                self.mcts.observe_q(node.total_value / node.number_visits)
            node = node.parent

    def apply_virtual_loss(self, loss: float) -> None:
        """Mark this path as in-flight: bump visits, debit value.

        Walks from this node up to (but not including) the root parent,
        adding `loss` virtual visits on each edge, each valued at the
        search's worst Q (`MCTS.pessimistic_q`). Concurrent selectors then
        see Q lowered and N raised on this path, so they prefer different
        leaves. Reverted in `revert_virtual_loss` once the real value backs up.
        """
        node = self
        while node.parent is not None:
            node.parent.child_virtual_loss_visits[node.action] += loss
            node = node.parent

    def revert_virtual_loss(self, loss: float) -> None:
        """Undo `apply_virtual_loss(loss)` along the same path."""
        node = self
        while node.parent is not None:
            node.parent.child_virtual_loss_visits[node.action] -= loss
            node = node.parent


class RootParentNode:
    """Stub parent for the search root.

    Holds the root's `child_*` statistics arrays. Has no `parent` of
    its own so `backup` knows when to stop walking.
    """

    def __init__(self, env):
        self.env = env
        self.parent = None
        self.child_total_value = np.zeros(
            [env.action_space.n], dtype=np.float32)
        self.child_number_visits = np.zeros(
            [env.action_space.n], dtype=np.float32)
        # Mirror of Node's per-edge virtual-loss buffers so the root
        # (whose parent is this stub) can be queried uniformly by
        # `Node.effective_number_visits`.
        self.child_virtual_loss_visits = np.zeros(
            [env.action_space.n], dtype=np.float32)


class MCTS:
    """PUCT-based Monte Carlo Tree Search.

    Args:
        compute_priors_and_value: callable `obs_dict -> (priors, value)`.
            `priors` is a length-`action_space.n` numpy array (the
            network's softmaxed policy); `value` is a scalar (the
            network's value estimate).
        mcts_param: dict of hyperparameters:
            num_simulations (int): simulations per real action selection.
            gamma (float): discount applied while backing values up.
            puct_coefficient (float): PUCT exploration constant `c`.
            temperature (float): sample temperature on visit counts when
                `argmax_tree_policy=False`. 0 collapses to argmax.
            argmax_tree_policy (bool): pick the most-visited child
                deterministically (True) or sample by visit-count
                distribution (False, paper-faithful for self-play).
            add_dirichlet_noise (bool): mix Dirichlet noise into the
                root prior to ensure self-play exploration.
            dir_epsilon (float): mix coefficient `(1-eps)*prior + eps*noise`.
            dir_noise (float): Dirichlet concentration α.
    """

    def __init__(self, compute_priors_and_value, mcts_param,
                 batched_priors_and_value=None):
        self.compute_priors_and_value = compute_priors_and_value
        # Optional batched evaluator: `obs_list -> [(priors, value), ...]`.
        # Used when `batch_size > 1` to amortize the GPU forward across
        # multiple MCTS leaves.
        self.batched_priors_and_value = batched_priors_and_value
        self.num_simulations = mcts_param['num_simulations']
        self.c_puct = mcts_param.get('puct_coefficient', 1.0)
        self.gamma = float(mcts_param.get('gamma', 1.0))
        self.temperature = mcts_param.get('temperature', 1.0)
        self.argmax_tree_policy = mcts_param.get('argmax_tree_policy', True)
        self.add_dirichlet_noise = mcts_param.get('add_dirichlet_noise', False)
        self.dir_epsilon = mcts_param.get('dir_epsilon', 0.25)
        self.dir_noise = mcts_param.get('dir_noise', 0.03)
        # Dirichlet concentration per legal action: alpha = scale / legal, so
        # the noise is equally spread on 5 legal actions and on 500. Replaces
        # `dir_noise` when set.
        self.dir_alpha_scale = mcts_param.get('dir_alpha_scale')
        # Min-max normalize Q to [0, 1] over this search (MuZero), so `c_puct`
        # means the same whatever scale the reward has.
        self.normalize_q = mcts_param.get('normalize_q', True)
        self._q_min = float('inf')
        self._q_max = float('-inf')
        # Anneal the sampling temperature linearly from `temperature` to
        # `final_temperature` over this many real moves of an episode. Early
        # moves explore, late ones commit. 0 keeps it constant.
        self.temperature_decay_moves = int(mcts_param.get('temperature_decay_moves', 0))
        self.final_temperature = float(mcts_param.get('final_temperature', 0.0))
        self.batch_size = int(mcts_param.get('batch_size', 1))
        self.virtual_loss = float(mcts_param.get('virtual_loss', 1.0))

    def observe_q(self, q: float) -> None:
        self._q_min = min(self._q_min, q)
        self._q_max = max(self._q_max, q)

    def pessimistic_q(self) -> float:
        """The value of an in-flight visit: the worst Q this search has
        backed up, or `-virtual_loss` before any."""
        return self._q_min if np.isfinite(self._q_min) else -self.virtual_loss

    def normalized_q(self, q: np.ndarray) -> np.ndarray:
        # Until two distinct values were backed up there is no scale to apply.
        if not self.normalize_q or self._q_max <= self._q_min:
            return q
        return np.clip((q - self._q_min) / (self._q_max - self._q_min), 0.0, 1.0)

    def temperature_at(self, move: int) -> float:
        if self.temperature_decay_moves <= 0:
            return self.temperature
        progress = min(move / self.temperature_decay_moves, 1.0)
        return self.temperature + progress * (self.final_temperature - self.temperature)

    def compute_action(self, node: 'Node'):
        """Run `num_simulations` rollouts from `node`; return the chosen action.

        When `batch_size > 1` and a batched evaluator was supplied,
        leaves are gathered with virtual loss and evaluated together
        (one GPU forward per flush). Otherwise the original
        sequential per-leaf evaluation is used.

        Returns:
            tree_policy (np.ndarray): length-`action_space.n` visit-count
                distribution over the root's children — the training
                target for the policy head.
            action (int): the action selected to actually take in the env.
            next_node (Node): the child node corresponding to `action`,
                detached from the parent so the next `compute_action`
                can reuse it as the new root.
        """
        self._q_min = float('inf')
        self._q_max = float('-inf')
        use_batched = (self.batch_size > 1
                       and self.batched_priors_and_value is not None)
        if use_batched:
            self._run_batched_simulations(node)
        else:
            self._run_sequential_simulations(node)

        return self._build_tree_policy_and_action(node)

    def _run_sequential_simulations(self, node: 'Node') -> None:
        """Original AlphaZero MCTS loop: one leaf eval per simulation."""
        self._run_sequential_simulations_remaining(node, self.num_simulations)

    def _run_sequential_simulations_remaining(self, node, n: int) -> None:
        """Helper: run `n` sequential simulations from `node`."""
        for _ in range(n):
            leaf = node.select()
            if leaf.done:
                value = 0.0
            else:
                child_priors, value = self.compute_priors_and_value(leaf.obs)
                child_priors = self._maybe_mix_dirichlet(
                    leaf, node, child_priors)
                leaf.expand(child_priors)
            leaf.backup(float(value))

    def _run_batched_simulations(self, node: 'Node') -> None:
        """Batched MCTS: gather `batch_size` leaves with virtual loss,
        evaluate them in one network forward, then back up.

        Terminal leaves are backed up immediately (no model call) and
        count as one simulation. Non-terminal leaves are reserved
        (`is_pending_eval=True`) with virtual loss along their path so
        the next `select()` prefers a different leaf; they count as
        one simulation only after the model evaluates them. If a
        select keeps returning a pending leaf (very narrow tree),
        we cap retries per attempted-fill to avoid deadlocking and
        flush whatever batch we've gathered.
        """
        loss = self.virtual_loss
        sims_done = 0
        while sims_done < self.num_simulations:
            pending_leaves = []
            fresh_count = 0
            target = min(self.batch_size,
                         self.num_simulations - sims_done)
            # Cap select retries when the only available leaf is
            # already pending — narrow trees can have no alternative.
            # Once we hit the cap we flush whatever's in the batch.
            retries_remaining = target * 4
            # A reserved leaf is a simulation still to count, so the budget
            # stops gathering, not just the batch size.
            while (len(pending_leaves) < target and retries_remaining > 0
                   and sims_done + fresh_count < self.num_simulations):
                leaf = node.select()
                if leaf.done:
                    leaf.backup(0.0)
                    sims_done += 1
                    continue
                if leaf.is_pending_eval:
                    # Bump virtual loss along this path again so the
                    # next select pushes elsewhere. Doesn't count.
                    leaf.apply_virtual_loss(loss)
                    pending_leaves.append((leaf, True))  # extra-vloss marker
                    retries_remaining -= 1
                    continue
                # Reserve this leaf for evaluation.
                leaf.is_pending_eval = True
                leaf.apply_virtual_loss(loss)
                pending_leaves.append((leaf, False))
                fresh_count += 1

            # Separate fresh leaves (need eval) from duplicates (just
            # got extra virtual loss). Only fresh ones consume sims.
            fresh = [(l, dup) for (l, dup) in pending_leaves if not dup]
            duplicates = [l for (l, dup) in pending_leaves if dup]

            if fresh:
                obs_list = [leaf.obs for (leaf, _) in fresh]
                results = self.batched_priors_and_value(obs_list)
                for (leaf, _), (child_priors, value) in zip(fresh, results):
                    child_priors = self._maybe_mix_dirichlet(
                        leaf, node, child_priors)
                    leaf.expand(child_priors)
                    leaf.is_pending_eval = False
                    leaf.revert_virtual_loss(loss)
                    leaf.backup(float(value))
                    sims_done += 1

            # Revert the extra virtual loss we put on duplicates;
            # they didn't get a real backup, so the path stats must
            # return to pre-batch state.
            for leaf in duplicates:
                leaf.revert_virtual_loss(loss)

            # Couldn't fill the batch AND made no progress — bail to
            # the sequential fallback for the remaining sims so we
            # don't loop forever on a tree with no expandable leaves.
            if not fresh and not duplicates:
                self._run_sequential_simulations_remaining(
                    node, self.num_simulations - sims_done)
                return

    def _maybe_mix_dirichlet(self, leaf, root, child_priors):
        """Mix Dirichlet noise into priors only when `leaf` is the root."""
        if leaf is not root or not self.add_dirichlet_noise:
            return child_priors
        legal_idx = np.flatnonzero(leaf.valid_actions)
        if legal_idx.size == 0:
            return child_priors
        alpha = (self.dir_noise if self.dir_alpha_scale is None
                 else self.dir_alpha_scale / legal_idx.size)
        noise = np.random.dirichlet([alpha] * legal_idx.size)
        child_priors = np.array(child_priors, dtype=np.float32)
        child_priors[legal_idx] = (
            (1 - self.dir_epsilon) * child_priors[legal_idx]
            + self.dir_epsilon * noise
        )
        return child_priors

    def _build_tree_policy_and_action(self, node: 'Node'):
        """Convert root visit counts into the policy target + chosen action."""
        tree_policy = node.child_number_visits / max(node.number_visits, 1)
        # Zero out illegal actions (paranoia — they shouldn't have visits).
        tree_policy = np.where(node.valid_actions, tree_policy, 0.0)
        s = tree_policy.sum()
        if s > 0:
            tree_policy = tree_policy / s

        if self.argmax_tree_policy:
            action = int(np.argmax(tree_policy))
        else:
            # Sample with temperature; T=0 collapses to argmax.
            temperature = self.temperature_at(node.depth)
            if temperature <= 0:
                action = int(np.argmax(tree_policy))
            else:
                logits = np.log(tree_policy + 1e-12) / temperature
                exp_logits = np.exp(logits - logits.max())
                # Mask illegal (their `tree_policy` is 0 so log is -inf).
                exp_logits = np.where(node.valid_actions, exp_logits, 0.0)
                probs = exp_logits / max(exp_logits.sum(), 1e-12)
                action = int(np.random.choice(len(probs), p=probs))

        next_node = node.get_child(action)
        # Detach so the rest of the tree (and its checkpoints) can be GC'd.
        next_node.parent = RootParentNode(node.env)
        next_node.parent.child_number_visits[action] = node.child_number_visits[action]
        next_node.parent.child_total_value[action] = node.child_total_value[action]
        return tree_policy, action, next_node
