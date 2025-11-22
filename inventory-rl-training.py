# Reinforcement Learning for Inventory Restocking Optimization

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import matplotlib.pyplot as plt
import collections
import random
import time

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# USD to INR conversion rate
USD_TO_INR = 83

print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Currency: Indian Rupees (INR)")


# PART 1: INVENTORY SIMULATION ENVIRONMENT


class InventoryEnv(gym.Env):
    """
    Custom Gym environment for single-product inventory management.
    
    State: Current inventory level (0-100 units)
    Actions: Discrete order quantities (0, 10, 20, 30, 40, 50 units)
    Reward: Revenue - Holding Cost - Stockout Cost (in INR)
    """
    
    def __init__(self):
        super(InventoryEnv, self).__init__()
        
        # Environment Parameters
        self.max_inventory = 100        # Maximum inventory capacity
        self.max_order_qty = 50         # Maximum order quantity
        self.n_actions = 6              # Number of discrete actions
        self.lead_time = 3              # Order lead time (days)
        
        # Cost Parameters (in INR)
        self.holding_cost = 0.1 * USD_TO_INR    # ₹8.30 per unit held per day
        self.stockout_cost = 1.0 * USD_TO_INR   # ₹83 per unit of unmet demand
        self.unit_price = 2.0 * USD_TO_INR      # ₹166 revenue per unit sold
        
        # Demand Parameters
        self.demand_mean = 20           # Average daily demand
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0, 
            high=self.max_inventory, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.inventory = 0
        self.pending_orders = collections.deque([0] * self.lead_time, maxlen=self.lead_time)
        self.day = 0
    
    def _get_action_value(self, action):
        """Convert discrete action index to order quantity"""
        return action * (self.max_order_qty // (self.n_actions - 1))
    
    def reset(self):
        """Reset environment to initial state"""
        self.inventory = np.random.randint(10, 30)
        self.pending_orders = collections.deque([0] * self.lead_time, maxlen=self.lead_time)
        self.day = 0
        return np.array([self.inventory], dtype=np.float32)
    
    def step(self, action):
        """Execute one time step"""
        self.day += 1
        
        # 1. Order arrives
        arrived_order = self.pending_orders.popleft()
        self.inventory = min(self.inventory + arrived_order, self.max_inventory)
        
        # 2. Place new order
        order_quantity = self._get_action_value(action)
        self.pending_orders.append(order_quantity)
        
        # 3. Simulate customer demand
        demand = np.random.poisson(self.demand_mean)
        
        # 4. Calculate sales and stockouts
        sales = min(self.inventory, demand)
        unmet_demand = demand - sales
        
        # 5. Update inventory
        self.inventory -= sales
        
        # 6. Calculate profit (positive values)
        revenue = sales * self.unit_price
        holding_cost_total = self.inventory * self.holding_cost
        stockout_cost_total = unmet_demand * self.stockout_cost
        profit = revenue - holding_cost_total - stockout_cost_total
        
        # Episode termination
        done = self.day >= 90  # 90-day episodes
        
        return (
            np.array([self.inventory], dtype=np.float32), 
            profit, 
            done, 
            {'unmet_demand': unmet_demand, 'revenue': revenue, 
             'holding_cost': holding_cost_total, 'stockout_cost': stockout_cost_total}
        )



# PART 2: DEEP Q-NETWORK (DQN) AGENT


class DQNAgent:
    """Deep Q-Network Agent for Inventory Management"""
    
    def __init__(self, state_shape, n_actions):
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Hyperparameters
        self.gamma = 0.95                    # Discount factor
        self.epsilon = 1.0                   # Exploration rate
        self.epsilon_min = 0.01              # Minimum exploration
        self.epsilon_decay = 0.995           # Exploration decay
        self.learning_rate = 0.001           # Learning rate
        self.batch_size = 64                 # Training batch size
        
        # Experience Replay Memory
        self.memory = collections.deque(maxlen=2000)
        
        # Neural Network Model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build neural network for Q-value approximation"""
        model = Sequential([
            Dense(24, input_shape=self.state_shape, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.n_actions, activation='linear')  # Q-values output
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        q_values = self.model(state, training=False)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = np.array([t[0] for t in minibatch]).reshape(-1, 1)
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch]).reshape(-1, 1)
        dones = np.array([t[4] for t in minibatch])
        
        # Compute target Q-values
        q_next = self.model.predict_on_batch(next_states)
        targets = rewards + self.gamma * np.amax(q_next, axis=1) * (1 - dones)
        
        # Update Q-values
        q_current = self.model.predict_on_batch(states)
        q_current[np.arange(self.batch_size), actions] = targets
        
        # Train model
        self.model.fit(states, q_current, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# PART 3: BASELINE POLICY


def fixed_reorder_policy(inventory, reorder_point=20, order_amount_idx=4):
    """
    Simple (s, S) inventory policy
    Reorder when inventory falls below reorder_point
    """
    if inventory < reorder_point:
        return order_amount_idx  # Order 40 units
    return 0  # No order



# PART 4: POLICY EVALUATION


def run_simulation(policy_func, env, episodes=100):
    """Run simulation and return performance metrics"""
    total_profits = []
    total_unmet_demands = []
    
    for e in range(episodes):
        state = env.reset()
        episode_profit = 0
        episode_unmet_demand = 0
        done = False
        
        while not done:
            action = policy_func(state[0])
            next_state, profit, done, info = env.step(action)
            state = next_state
            episode_profit += profit
            episode_unmet_demand += info['unmet_demand']
        
        total_profits.append(episode_profit)
        total_unmet_demands.append(episode_unmet_demand)
    
    avg_profit = np.mean(total_profits)
    avg_unmet_demand = np.mean(total_unmet_demands)
    
    return avg_profit, avg_unmet_demand



# PART 5: MAIN TRAINING AND EVALUATION


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" REINFORCEMENT LEARNING FOR INVENTORY MANAGEMENT")
    print(" Currency: Indian Rupees (₹)")
    print("="*70 + "\n")
    
    # Display Environment Parameters
    print("📦 ENVIRONMENT PARAMETERS:")
    print(f"   • Max Inventory Capacity: 100 units")
    print(f"   • Average Daily Demand: 20 units")
    print(f"   • Lead Time: 3 days")
    print(f"   • Unit Selling Price: ₹{2.0 * USD_TO_INR:.2f}")
    print(f"   • Holding Cost: ₹{0.1 * USD_TO_INR:.2f} per unit/day")
    print(f"   • Stockout Cost: ₹{1.0 * USD_TO_INR:.2f} per unmet demand")
    print(f"   • Episode Length: 90 days\n")
    
    # Setup
    env = InventoryEnv()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)
    episodes = 500
    
    # Training
    print("--- Starting DQN Agent Training ---\n")
    start_time = time.time()
    profits_history = []
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_shape[0]])
        total_profit = 0
        
        for step in range(100):  # Max steps per episode
            action = agent.act(state)
            next_state, profit, done, _ = env.step(action)
            total_profit += profit
            next_state = np.reshape(next_state, [1, state_shape[0]])
            
            # Store experience
            agent.remember(state[0][0], action, profit, next_state[0][0], done)
            state = next_state
            
            if done:
                break
        
        profits_history.append(total_profit)
        agent.replay()  # Train after each episode
        
        # Progress updates
        if (e + 1) % 50 == 0:
            avg_profit = np.mean(profits_history[-50:])
            print(f"Episode: {e + 1:4d}/{episodes} | "
                  f"Avg Profit (last 50): ₹{avg_profit:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    print(f"\n--- Training Completed in {training_time:.2f} seconds ---\n")
    
    # Evaluation
    print("--- Evaluating Policies ---\n")
    eval_episodes = 200
    
    # DQN Policy
    def dqn_policy(inventory):
        state = np.reshape(np.array([inventory]), [1, state_shape[0]])
        agent.epsilon = 0.0  # Pure exploitation
        return agent.act(state)
    
    dqn_profit, dqn_unmet = run_simulation(dqn_policy, env, eval_episodes)
    print(f"🤖 DQN Agent")
    print(f"   Average Profit: ₹{dqn_profit:,.2f} per 90-day cycle")
    print(f"   Unmet Demand: {dqn_unmet:.2f} units\n")
    
    # Fixed Policy
    fixed_profit, fixed_unmet = run_simulation(fixed_reorder_policy, env, eval_episodes)
    print(f"📋 Fixed Reorder Policy")
    print(f"   Average Profit: ₹{fixed_profit:,.2f} per 90-day cycle")
    print(f"   Unmet Demand: {fixed_unmet:.2f} units\n")
    
    # Performance Comparison
    improvement = ((dqn_profit - fixed_profit) / fixed_profit) * 100
    additional_profit = dqn_profit - fixed_profit
    service_level_dqn = (1 - (dqn_unmet / (20 * 90))) * 100
    service_level_fixed = (1 - (fixed_unmet / (20 * 90))) * 100
    
    print("=" * 70)
    print(" PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\n💰 Additional Profit (DQN vs Fixed): ₹{additional_profit:,.2f} per cycle")
    print(f"📈 Profit Improvement: {improvement:.2f}%")
    print(f"🎯 DQN Service Level: {service_level_dqn:.1f}%")
    print(f"🎯 Fixed Service Level: {service_level_fixed:.1f}%")
    print(f"✨ Service Level Improvement: {service_level_dqn - service_level_fixed:.1f}%\n")
    
    # Annual Projection
    cycles_per_year = 365 / 90
    annual_additional_profit = additional_profit * cycles_per_year
    print(f"💵 Projected Annual Additional Profit: ₹{annual_additional_profit:,.2f}\n")
    
    # Visualization
    print("--- Generating Visualizations ---\n")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Training Progress
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(profits_history, alpha=0.6, linewidth=1, label='Episode Profit', color='#3b82f6')
    moving_avg = [np.mean(profits_history[max(0, i-20):i+1]) 
                  for i in range(len(profits_history))]
    ax1.plot(moving_avg, color='red', linewidth=3, label='Moving Average (20)')
    ax1.set_title('DQN Training Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Profit (₹)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Format y-axis to show Indian Rupee symbol
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    # Plot 2: Profit Comparison
    ax2 = plt.subplot(1, 3, 2)
    policies = ['DQN Agent', 'Fixed Policy']
    profits = [dqn_profit, fixed_profit]
    colors = ['#10b981', '#ef4444']
    
    bars = ax2.bar(policies, profits, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_title('Profit Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Profit per 90-Day Cycle (₹)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:,.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    ax2.annotate(f'{improvement:.1f}% Better', 
                xy=(0.5, max(profits) * 0.95),
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Format y-axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    # Plot 3: Service Level Comparison
    ax3 = plt.subplot(1, 3, 3)
    service_levels = [service_level_dqn, service_level_fixed]
    colors_sl = ['#3b82f6', '#f59e0b']
    
    bars_sl = ax3.bar(policies, service_levels, color=colors_sl, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_title('Service Level Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Service Level (%)', fontsize=12)
    ax3.set_ylim([90, 100])
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels on bars
    for bar in bars_sl:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('inventory_rl_results_inr.png', dpi=300, bbox_inches='tight')
    print("✅ Results saved to 'inventory_rl_results_inr.png'\n")
    plt.show()
    
    # Summary Table
    print("=" * 70)
    print(" DETAILED COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} {'DQN Agent':<20} {'Fixed Policy':<20}")
    print("-" * 70)
    print(f"{'Average Profit (₹)':<30} {f'₹{dqn_profit:,.2f}':<20} {f'₹{fixed_profit:,.2f}':<20}")
    print(f"{'Unmet Demand (units)':<30} {f'{dqn_unmet:.2f}':<20} {f'{fixed_unmet:.2f}':<20}")
    print(f"{'Service Level (%)':<30} {f'{service_level_dqn:.2f}%':<20} {f'{service_level_fixed:.2f}%':<20}")
    print(f"{'Profit Improvement':<30} {f'+{improvement:.2f}%':<20} {'Baseline':<20}")
    print(f"{'Additional Profit (₹)':<30} {f'+₹{additional_profit:,.2f}':<20} {'-':<20}")
    print("=" * 70)
    
    print("\n🎉 TRAINING AND EVALUATION COMPLETE!")
    print(f"💡 The DQN agent generates ₹{additional_profit:,.2f} more profit per cycle")
    print(f"💡 Projected annual additional profit: ₹{annual_additional_profit:,.2f}")
    print("=" * 70)
