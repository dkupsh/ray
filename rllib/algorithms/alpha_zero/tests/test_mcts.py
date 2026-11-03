"""Standalone MCTS smoke tests against a toy env.

These don't depend on the scheduler env or any model — they use a
hand-rolled deterministic gym env so the assertions stay tight.
"""
import unittest

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete

from ray.rllib.algorithms.alpha_zero.mcts import MCTS, Node, RootParentNode


class _ToyEnv(gym.Env):
    """1-step env: action 0 yields reward 0, action 1 yields reward 1.

    Terminal-only reward — the standard AlphaZero regime, where MCTS
    backs up `leaf.reward` for done leaves and the network value head
    for non-terminal leaves.
    """

    action_space = Discrete(2)
    observation_space = Dict({
        'action_mask': Box(low=0, high=1, shape=(2,), dtype=np.int8),
        'depth': Box(low=0, high=10, shape=(1,), dtype=np.int32),
    })

    def __init__(self):
        self._depth = 0
        self._snapshots = {}
        self._next_handle = 0

    def _obs(self):
        return {
            'action_mask': np.ones(2, dtype=np.int8),
            'depth': np.array([self._depth], dtype=np.int32),
        }

    def reset(self, *, seed=None, options=None):
        self._depth = 0
        return self._obs(), {}

    def step(self, action):
        self._depth += 1
        reward = float(action)
        done = self._depth >= 1
        return self._obs(), reward, done, False, {}

    def get_state(self) -> str:
        h = f'h{self._next_handle}'
        self._next_handle += 1
        self._snapshots[h] = self._depth
        return h

    def set_state(self, h: str):
        self._depth = self._snapshots[h]
        return self._obs()


def _uniform_priors_zero_value(_obs):
    return np.array([0.5, 0.5], dtype=np.float32), 0.0


def _action_1_preferred(_obs):
    return np.array([0.1, 0.9], dtype=np.float32), 0.0


class TestMCTS(unittest.TestCase):
    """PUCT visit counts and action selection."""

    def _make_root(self, env, mcts):
        obs, _ = env.reset()
        state = env.get_state()
        return Node(
            action=0, obs=obs, done=False, reward=0,
            state=state, mcts=mcts, parent=RootParentNode(env),
        )

    def test_visit_counts_sum_to_num_simulations_minus_one(self):
        """The very first simulation expands the root without descending,
        so children accumulate `num_simulations - 1` visits in total."""
        env = _ToyEnv()
        mcts = MCTS(_uniform_priors_zero_value, {
            'num_simulations': 32,
            'puct_coefficient': 1.0,
            'argmax_tree_policy': True,
        })
        root = self._make_root(env, mcts)
        tree_policy, action, _ = mcts.compute_action(root)
        total_visits = root.child_number_visits.sum()
        self.assertEqual(int(total_visits), 31)
        # Tree policy normalizes visits to a distribution over legal actions.
        self.assertAlmostEqual(float(tree_policy.sum()), 1.0, places=5)
        self.assertIn(action, [0, 1])

    def test_action_follows_terminal_reward(self):
        """In a one-step env with terminal reward = action, MCTS picks 1.

        With uniform priors and zero value head, the only signal MCTS
        gets is the immediate reward at terminal leaves (`leaf.reward`).
        After enough simulations PUCT must concentrate on action 1.
        """
        env = _ToyEnv()
        mcts = MCTS(_uniform_priors_zero_value, {
            'num_simulations': 16,
            'puct_coefficient': 1.0,
            'argmax_tree_policy': True,
        })
        root = self._make_root(env, mcts)
        _, action, _ = mcts.compute_action(root)
        self.assertEqual(action, 1)

    def test_priors_bias_search(self):
        """Action-1-preferring priors must increase its visit share over uniform."""
        env = _ToyEnv()
        mcts_uniform = MCTS(_uniform_priors_zero_value, {
            'num_simulations': 32,
            'puct_coefficient': 1.0,
            'argmax_tree_policy': True,
        })
        mcts_biased = MCTS(_action_1_preferred, {
            'num_simulations': 32,
            'puct_coefficient': 1.0,
            'argmax_tree_policy': True,
        })
        # Compare visit fractions for action 1 across the two priors.
        root_u = self._make_root(_ToyEnv(), mcts_uniform)
        mcts_uniform.compute_action(root_u)
        root_b = self._make_root(_ToyEnv(), mcts_biased)
        mcts_biased.compute_action(root_b)
        self.assertGreaterEqual(
            root_b.child_number_visits[1],
            root_u.child_number_visits[1],
        )

    def test_action_mask_respected(self):
        """An MCTS run on an env that masks action 0 never visits action 0."""

        class _MaskedEnv(_ToyEnv):
            def _obs(self):
                # Mask out action 0 — only action 1 is legal.
                return {
                    'action_mask': np.array([0, 1], dtype=np.int8),
                    'depth': np.array([self._depth], dtype=np.int32),
                }

        env = _MaskedEnv()
        mcts = MCTS(_uniform_priors_zero_value, {
            'num_simulations': 16,
            'puct_coefficient': 1.0,
            'argmax_tree_policy': True,
        })
        root = self._make_root(env, mcts)
        _, action, _ = mcts.compute_action(root)
        self.assertEqual(action, 1)
        self.assertEqual(int(root.child_number_visits[0]), 0)


class TestBackup(unittest.TestCase):

    def test_every_edge_gets_its_rewards_and_the_discounted_value_beyond(self):
        mcts = MCTS(None, {'num_simulations': 1, 'gamma': 0.5})

        class Env:
            class action_space:
                n = 2

        root_parent = RootParentNode(Env())
        mask = {'action_mask': np.array([True, True])}
        root = Node(0, mask, False, 0.0, None, mcts, parent=root_parent)
        middle = Node(1, mask, False, -100.0, None, mcts, parent=root)
        leaf = Node(0, mask, True, 8.0, None, mcts, parent=middle)
        root.children[1] = middle
        middle.children[0] = leaf

        leaf.backup(0.0)
        self.assertEqual(middle.child_total_value[0], 8.0)
        self.assertEqual(root.child_total_value[1], -100.0 + 0.5 * 8.0)


class TestRootEdge(unittest.TestCase):

    def test_root_arrival_reward_does_not_set_the_q_scale(self):
        mcts = MCTS(None, {'num_simulations': 1, 'gamma': 1.0, 'normalize_q': True})

        class Env:
            class action_space:
                n = 2

        mask = {'action_mask': np.array([True, True])}
        root = Node(0, mask, False, -1000.0, None, mcts,
                    parent=RootParentNode(Env()))
        good = Node(0, mask, True, 0.0, None, mcts, parent=root)
        bad = Node(1, mask, True, -1.0, None, mcts, parent=root)
        good.backup(0.0)
        bad.backup(0.0)
        self.assertEqual(mcts._q_min, -1.0)
        self.assertEqual(mcts._q_max, 0.0)


class TestSearchScaling(unittest.TestCase):
    """Settings that must hold whatever scale the reward has."""

    def _mcts(self, **param):
        return MCTS(None, {'num_simulations': 1, **param})

    def test_q_is_unscaled_until_two_values_are_seen(self):
        mcts = self._mcts(normalize_q=True)
        q = np.array([-50.0, 3.0])
        np.testing.assert_array_equal(mcts.normalized_q(q), q)
        mcts.observe_q(-50.0)
        np.testing.assert_array_equal(mcts.normalized_q(q), q)

    def test_normalization_cancels_reward_scale(self):
        small, large = self._mcts(normalize_q=True), self._mcts(normalize_q=True)
        values = np.array([0.0, 1.0, 4.0])
        for value in values:
            small.observe_q(value)
            large.observe_q(value * 100.0)
        np.testing.assert_allclose(
            small.normalized_q(values), large.normalized_q(values * 100.0))

    def test_temperature_reaches_final_and_stays(self):
        mcts = self._mcts(temperature=1.0, temperature_decay_moves=10,
                          final_temperature=0.1)
        self.assertEqual(mcts.temperature_at(0), 1.0)
        self.assertAlmostEqual(mcts.temperature_at(10), 0.1)
        self.assertAlmostEqual(mcts.temperature_at(99), 0.1)


class _ChainEnv(gym.Env):
    """Two forced moves: -100 for the first, a terminal 8 for the second."""

    action_space = Discrete(2)
    observation_space = _ToyEnv.observation_space
    rewards = (-100.0, 8.0)

    def __init__(self):
        self._depth = 0
        self._snapshots = {}

    def _obs(self):
        return {
            'action_mask': np.array([1, 0], dtype=np.int8),
            'depth': np.array([self._depth], dtype=np.int32),
        }

    def reset(self, *, seed=None, options=None):
        self._depth = 0
        return self._obs(), {}

    def step(self, action):
        reward = self.rewards[self._depth]
        self._depth += 1
        return self._obs(), reward, self._depth == len(self.rewards), False, {}

    def get_state(self):
        handle = len(self._snapshots)
        self._snapshots[handle] = self._depth
        return handle

    def set_state(self, handle):
        self._depth = self._snapshots[handle]
        return self._obs()


class TestSearchModesAgree(unittest.TestCase):
    """Sequential and batched search back up the same path returns."""

    VALUE = 2.0
    GAMMA = 0.5

    def _search(self, batch_size):
        env = _ChainEnv()

        def evaluate(_obs):
            return np.array([1.0, 0.0], dtype=np.float32), self.VALUE

        mcts = MCTS(evaluate, {
            'num_simulations': 3, 'gamma': self.GAMMA, 'batch_size': batch_size,
            'normalize_q': False,
        }, batched_priors_and_value=lambda obs_list: [evaluate(o) for o in obs_list])
        obs, _ = env.reset()
        root = Node(0, obs, False, 0.0, env.get_state(), mcts,
                    parent=RootParentNode(env))
        if batch_size == 1:
            mcts._run_sequential_simulations(root)
        else:
            mcts._run_batched_simulations(root)
        return root

    def test_path_rewards_reach_the_root(self):
        # Expanding the middle node backs up -100 + gamma * V; reaching the
        # terminal leaf backs up 8 once and -100 + gamma * 8 above it.
        expected_root = (-100.0 + self.GAMMA * self.VALUE) + (-100.0 + self.GAMMA * 8.0)
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                root = self._search(batch_size)
                middle = root.children[0]
                self.assertEqual(root.child_number_visits[0], 2)
                self.assertAlmostEqual(root.child_total_value[0], expected_root)
                self.assertAlmostEqual(middle.child_total_value[0], 8.0)
                self.assertAlmostEqual(root.child_Q()[0], expected_root / 2)


class TestChildQ(unittest.TestCase):

    def test_q_is_the_mean_and_unvisited_edges_score_zero(self):
        mcts = MCTS(None, {'num_simulations': 1, 'normalize_q': False})

        class Env:
            class action_space:
                n = 2

        root = Node(0, {'action_mask': np.array([True, True])}, False, 0.0,
                    None, mcts, parent=RootParentNode(Env()))
        root.child_total_value[:] = [-6.0, 0.0]
        root.child_number_visits[:] = [3.0, 0.0]
        np.testing.assert_array_equal(root.child_Q(), [-2.0, 0.0])


class TestValueTarget(unittest.TestCase):

    def test_target_is_the_return_after_the_state(self):
        from ray.rllib.algorithms.alpha_zero.alpha_zero import (
            VALUE_TARGET, _SelfPlayWorker)

        class OneStep(_ToyEnv):
            def step(self, action):
                obs, _, done, truncated, info = super().step(action)
                return obs, 10.0, done, truncated, info

        worker = object.__new__(_SelfPlayWorker)
        worker.env = OneStep()
        worker.gamma = 0.99
        worker.value_target_scale = 1.0
        worker.module = None
        worker._callbacks_objs = []
        worker._callbacks_funcs = {}
        worker.mcts = MCTS(_uniform_priors_zero_value,
                           {'num_simulations': 4, 'gamma': worker.gamma})
        trajectory = worker.run_episode()['trajectory']
        self.assertEqual(len(trajectory), 1)
        self.assertAlmostEqual(trajectory[0][VALUE_TARGET], 10.0)


class TestBatchedSearch(unittest.TestCase):

    def test_backups_equal_the_simulation_budget(self):
        for budget in (7, 10, 33):
            env = _ToyEnv()
            mcts = MCTS(_uniform_priors_zero_value,
                        {'num_simulations': budget, 'batch_size': 4},
                        batched_priors_and_value=lambda obs_list: [
                            _uniform_priors_zero_value(o) for o in obs_list])
            obs, _ = env.reset()
            root = Node(0, obs, False, 0.0, env.get_state(), mcts,
                        parent=RootParentNode(env))
            mcts._run_batched_simulations(root)
            with self.subTest(budget=budget):
                self.assertEqual(int(root.number_visits), budget)

    def test_pending_visit_never_raises_q(self):
        mcts = MCTS(None, {'num_simulations': 1, 'normalize_q': True})

        class Env:
            class action_space:
                n = 2

        root = Node(0, {'action_mask': np.array([True, True])}, False, 0.0,
                    None, mcts, parent=RootParentNode(Env()))
        bad = Node(1, root.obs, False, -100.0, None, mcts, parent=root)
        root.children[1] = bad
        bad.backup(0.0)
        mcts.observe_q(0.0)
        before = root.child_Q()[1]
        bad.apply_virtual_loss(1.0)
        self.assertLessEqual(root.child_Q()[1], before)


if __name__ == '__main__':
    unittest.main()
