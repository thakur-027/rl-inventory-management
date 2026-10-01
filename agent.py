"""DQN Agent implementation with Target Network, Double DQN, and Adam optimizer.

Pure NumPy implementation, independent of heavy framework dependencies.

Features:
- MLP Q-network with customizable hidden layers (default: [32, 32])
- Target network with hard sync every `target_update_freq` steps
- Double DQN toggle for reduced Q-value overestimation
- Huber loss (smooth L1) with gradient norm clipping
- Full Adam optimizer with bias-corrected first and second moment estimates
- Linear epsilon decay schedule tracking total environment steps
- Experience replay buffer with capacity 20,000+
- Save/load weights using NumPy .npz
"""

from __future__ import annotations

import collections
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from config import AgentConfig, EnvConfig


# =====================================================================
# Neural Network with Adam Optimizer (Pure NumPy)
# =====================================================================

class MLPNetwork:
    """Multi-Layer Perceptron Q-Network with Adam optimizer."""

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.001,
        huber_delta: float = 1.0,
        gradient_clip: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.layer_sizes = list(layer_sizes)
        self.lr = learning_rate
        self.huber_delta = huber_delta
        self.gradient_clip = gradient_clip

        rng = np.random.default_rng(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # He / Xavier initialization
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            w = rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

        # Adam optimizer state
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.t = 0  # Adam timestep counter

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def forward(
        self, x: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through all layers returning activations and pre-activations."""
        activations = [x]
        z_values = []

        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:
                # Hidden layer: ReLU
                a = np.maximum(0.0, z)
            else:
                # Output layer: Linear
                a = z
            activations.append(a)

        return activations, z_values

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fast forward pass returning only the output Q-values."""
        a = x
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            a = np.maximum(0.0, z) if i < len(self.weights) - 1 else z
        return a

    def copy_weights_from(self, source: MLPNetwork) -> None:
        """Copy weights and biases from source network."""
        self.weights = [w.copy() for w in source.weights]
        self.biases = [b.copy() for b in source.biases]

    def train_on_targets(
        self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Perform one Adam gradient step for Huber loss on Q(s, a) vs targets.

        Parameters
        ----------
        states : (B, state_size)
        actions : (B,) int
        targets : (B,) float target Q-values for the chosen action
        """
        batch_size = states.shape[0]
        activations, z_values = self.forward(states)
        q_pred = activations[-1]  # (B, n_actions)

        # Extract predictions for the specific actions taken
        batch_indices = np.arange(batch_size)
        pred_actions = q_pred[batch_indices, actions]

        # TD Error
        error = pred_actions - targets  # (B,)

        # Huber loss calculation:
        # L(e) = 0.5 * e^2 if |e| <= delta else delta * (|e| - 0.5 * delta)
        abs_error = np.abs(error)
        huber_loss = np.where(
            abs_error <= self.huber_delta,
            0.5 * (error ** 2),
            self.huber_delta * (abs_error - 0.5 * self.huber_delta),
        ).mean()

        # Derivative of Huber loss w.r.t q_pred[i, a]:
        # dL/de = e if |e| <= delta else delta * sign(e)
        delta_val = np.where(
            abs_error <= self.huber_delta,
            error,
            self.huber_delta * np.sign(error),
        )

        # Gradient at output layer (B, n_actions)
        grad_out = np.zeros_like(q_pred)
        grad_out[batch_indices, actions] = delta_val / batch_size

        # Backpropagation
        delta = grad_out
        dw_list: List[np.ndarray] = []
        db_list: List[np.ndarray] = []

        for i in range(len(self.weights) - 1, -1, -1):
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # Backpropagate through ReLU: (z > 0)
                delta = (delta @ self.weights[i].T) * (z_values[i - 1] > 0.0).astype(np.float32)

            dw_list.insert(0, dw)
            db_list.insert(0, db)

        # Global gradient clipping by norm
        total_norm_sq = sum(np.sum(dw ** 2) for dw in dw_list) + sum(np.sum(db ** 2) for db in db_list)
        total_norm = np.sqrt(total_norm_sq)
        clip_coef = 1.0
        if total_norm > self.gradient_clip and total_norm > 0:
            clip_coef = self.gradient_clip / total_norm

        # Adam optimization update
        self.t += 1
        lr_t = self.lr * (np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))

        for i in range(len(self.weights)):
            g_w = dw_list[i] * clip_coef
            g_b = db_list[i] * clip_coef

            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * g_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * (g_w ** 2)
            self.weights[i] -= lr_t * self.m_w[i] / (np.sqrt(self.v_w[i]) + self.adam_eps)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * g_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (g_b ** 2)
            self.biases[i] -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.adam_eps)

        return float(huber_loss)


# =====================================================================
# Experience Replay Buffer
# =====================================================================

class ReplayBuffer:
    """Fast circular replay buffer storing transition tuples."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.idx = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


# =====================================================================
# DQN Agent
# =====================================================================

class DQNAgent:
    """Deep Q-Network Agent supporting Target Networks and Double DQN."""

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        agent_cfg: Optional[AgentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.env_cfg = env_cfg or EnvConfig()
        self.cfg = agent_cfg or AgentConfig()

        self.state_size = 1 if self.cfg.mask_pipeline else self.env_cfg.state_size
        self.n_actions = self.env_cfg.n_actions

        self.rng = np.random.default_rng(seed)

        layer_sizes = [self.state_size] + list(self.cfg.hidden_sizes) + [self.n_actions]

        # Primary online Q-network
        self.online_net = MLPNetwork(
            layer_sizes=layer_sizes,
            learning_rate=self.cfg.learning_rate,
            huber_delta=self.cfg.huber_delta,
            gradient_clip=self.cfg.gradient_clip,
            seed=seed,
        )

        # Target Q-network
        self.target_net = MLPNetwork(
            layer_sizes=layer_sizes,
            learning_rate=self.cfg.learning_rate,
            huber_delta=self.cfg.huber_delta,
            gradient_clip=self.cfg.gradient_clip,
            seed=None if seed is None else seed + 1000,
        )
        self.target_net.copy_weights_from(self.online_net)

        # Experience replay buffer
        self.buffer = ReplayBuffer(self.cfg.replay_buffer_size, self.state_size)

        # Counters and exploration state
        self.total_steps = 0
        self.epsilon = self.cfg.epsilon_start

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Choose action using epsilon-greedy policy.

        Parameters
        ----------
        obs : normalised observation vector (state_size,)
        explore : if True, use epsilon-greedy; if False, greedy action
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        _obs = obs[:1] if self.cfg.mask_pipeline else obs
        state_batch = _obs.reshape(1, -1)
        q_vals = self.online_net.predict(state_batch)[0]
        return int(np.argmax(q_vals))

    def update_epsilon(self) -> None:
        """Linear decay of epsilon over cfg.epsilon_decay_steps."""
        if self.total_steps >= self.cfg.epsilon_decay_steps:
            self.epsilon = self.cfg.epsilon_end
        else:
            fraction = self.total_steps / float(self.cfg.epsilon_decay_steps)
            self.epsilon = self.cfg.epsilon_start - fraction * (
                self.cfg.epsilon_start - self.cfg.epsilon_end
            )

    def step_learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition, update epsilon, step target sync, and train on batch.

        Returns loss if an update occurred, else None.
        """
        # Store scaled reward
        _state = state[:1] if self.cfg.mask_pipeline else state
        _next_state = next_state[:1] if self.cfg.mask_pipeline else next_state

        scaled_reward = reward * self.cfg.reward_scale
        self.buffer.add(_state, action, scaled_reward, _next_state, done)

        self.total_steps += 1
        self.update_epsilon()

        # Target network periodic hard sync
        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

        # Only train after warmup buffer is populated
        if len(self.buffer) < max(self.cfg.warmup_steps, self.cfg.batch_size):
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.cfg.batch_size, self.rng
        )

        # Calculate target Q-values
        if self.cfg.double_dqn:
            # Double DQN: action selected by online_net, evaluated by target_net
            next_q_online = self.online_net.predict(next_states)
            best_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_net.predict(next_states)
            max_next_q = next_q_target[np.arange(self.cfg.batch_size), best_actions]
        else:
            # Standard DQN: max over target_net
            next_q_target = self.target_net.predict(next_states)
            max_next_q = np.max(next_q_target, axis=1)

        target_q = rewards + self.cfg.gamma * (1.0 - dones.astype(np.float32)) * max_next_q

        # Perform gradient update
        loss = self.online_net.train_on_targets(states, actions, target_q)
        return loss

    def save(self, file_path: str) -> None:
        """Save model weights and metadata to .npz file."""
        save_dict = {
            f"w_{i}": w for i, w in enumerate(self.online_net.weights)
        }
        for i, b in enumerate(self.online_net.biases):
            save_dict[f"b_{i}"] = b
        save_dict["total_steps"] = np.array(self.total_steps)
        save_dict["epsilon"] = np.array(self.epsilon)
        np.savez(file_path, **save_dict)

    def load(self, file_path: str) -> None:
        """Load model weights from .npz file."""
        data = np.load(file_path)
        for i in range(len(self.online_net.weights)):
            self.online_net.weights[i] = data[f"w_{i}"].copy()
            self.online_net.biases[i] = data[f"b_{i}"].copy()
        self.target_net.copy_weights_from(self.online_net)
        if "total_steps" in data:
            self.total_steps = int(data["total_steps"])
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])
