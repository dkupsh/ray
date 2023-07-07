"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np


class Node:
    def __init__(self, action, action_list, obs, done, reward, state, mcts, parent=None):
        self.env = parent.env
        self.action = action  # Action used to go to this state
        self.action_list = action_list # Actions used to go to this state from root

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n
        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # N
        self.valid_actions = obs["action_mask"].astype(np.bool_)

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts : MCTS = mcts
        
        self.mcts.nodes_expanded += 1

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return (
            math.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.mcts.c_puct * self.child_U()
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        
        return  np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors
    
    def has_child(self, action):
        return action in self.children

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
                
            next_state = self.env.get_state()
            
            self.children[action] = Node(
                state=next_state,
                action_list=self.action_list + [action],
                action=action,
                parent=self,
                reward=reward + self.reward,
                done=terminated,
                obs=obs,
                mcts=self.mcts,
            )
            
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self, model, env, mcts_param : dict):
        # Baseline Parameters
        self.model = model
        self.env = env
        
        self.nodes_expanded = 0
        
        # Calculate Root Node
        obs, _ = env.reset()
        state = env.get_state()
        self.root = Node(
            state=env.get_state(),
            obs=obs,
            reward=0,
            done=False,
            action=None,
            action_list=[],
            parent=RootParentNode(env=self.env),
            mcts=self,
        )
        
        self.temperature = mcts_param.get("temperature", 1.0)
        self.dir_epsilon = mcts_param.get("dirichlet_epsilon", 0.25)
        self.dir_noise = mcts_param.get("dirichlet_noise", 0.03)
        self.num_sims = mcts_param.get("num_simulations", 0)
        self.exploit = mcts_param.get("argmax_tree_policy", False)
        self.add_dirichlet_noise = mcts_param.get("add_dirichlet_noise", False)
        self.c_puct = mcts_param.get("puct_coefficient", 1.0)
    
    def get_node(self, actions):
        node = self.root
        for action in actions:
            node = node.get_child(action)
        if np.array_equal(node.action_list, actions) == False:
            raise Exception('Node action list', node.action_list, 'does not match action list', actions)
        return node

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(leaf.obs)
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    child_priors += self.dir_epsilon * np.random.dirichlet(
                        [self.dir_noise] * child_priors.size
                    )

                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(
            tree_policy
        )  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)
        else:
            # otherwise sample an action according to tree policy probabilities
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action