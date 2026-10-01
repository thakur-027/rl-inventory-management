# Superseded legacy prototype. Use the top-level train.py, evaluate.py, and reproduce.py pipeline.
# Reinforcement Learning for Inventory Restocking Optimization
# Pure NumPy implementation (no TensorFlow dependency)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import collections
import random
import time

# USD to INR conversion rate
USD_TO_INR = 83

print(f"NumPy Version: {np.__version__}")
print(f"Currency: Indian Rupees (INR)")


# PART 1: INVENTORY SIMULATION ENVIRONMENT


class InventoryEnv:
    """
    Custom environment for single-product inventory management.
    
    State: Current inventory level (0-100 units)
    Actions: Discrete order quantities (0, 10, 20, 30, 40, 50 units)
    Reward: Revenue - Holding Cost - Stockout Cost (in INR)
    """
    
    def __init__(self):
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
             'holding_cost': holding_cost_total, 'stockout_cost': stockout_cost_total,
             'demand': demand, 'sales': sales}
        )



# PART 2: PURE NUMPY NEURAL NETWORK


class NumpyNeuralNetwork:
    """Simple feedforward neural network using pure NumPy."""
    
    def __init__(self, layer_sizes, learning_rate=0.001):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)
    
    def forward(self, x):
        """Forward pass, returns all layer activations for backprop."""
        activations = [x]
        z_values = []
        
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            if i < len(self.weights) - 1:  # ReLU for hidden layers
                activations.append(self.relu(z))
            else:  # Linear for output layer
                activations.append(z)
        
        return activations, z_values
    
    def predict(self, x):
        """Forward pass, returns only output."""
        a = x
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z
        return a
    
    def copy_from(self, other):
        """Copy weights from another network (for target network)."""
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]
    
    def train_batch(self, x, y):
        """Train on a batch using Huber loss and backpropagation."""
        batch_size = x.shape[0]
        activations, z_values = self.forward(x)
        
        # Huber loss gradient (more stable than MSE for large rewards)
        error = activations[-1] - y
        delta = np.where(np.abs(error) <= 1.0, error, np.sign(error)) / batch_size
        
        # Backpropagation
        for i in range(len(self.weights) - 1, -1, -1):
            dw = activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(z_values[i-1])
            
            # Gradient clipping by norm
            dw_norm = np.linalg.norm(dw)
            if dw_norm > 10.0:
                dw = dw * (10.0 / dw_norm)
            db_norm = np.linalg.norm(db)
            if db_norm > 10.0:
                db = db * (10.0 / db_norm)
            
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db



# PART 3: DQN AGENT (NumPy-based)


class DQNAgent:
    """Deep Q-Network Agent for Inventory Management (Pure NumPy)"""
    
    def __init__(self, state_size, n_actions):
        self.state_size = state_size
        self.n_actions = n_actions
        
        # Hyperparameters
        self.gamma = 0.95                    # Discount factor
        self.epsilon = 1.0                   # Exploration rate
        self.epsilon_min = 0.01              # Minimum exploration
        self.epsilon_decay = 0.995           # Exploration decay
        self.learning_rate = 0.001           # Learning rate
        self.batch_size = 64                 # Training batch size
        self.reward_scale = 1.0 / USD_TO_INR # Normalize INR rewards to ~USD scale
        self.target_update_freq = 10         # Update target network every N episodes
        self.train_step = 0
        
        # Experience Replay Memory
        self.memory = collections.deque(maxlen=5000)
        
        # Neural Network Model (input -> 32 -> 32 -> n_actions)
        self.model = NumpyNeuralNetwork(
            [state_size, 32, 32, n_actions], 
            learning_rate=self.learning_rate
        )
        # Target network for stable Q-value bootstrapping
        self.target_model = NumpyNeuralNetwork(
            [state_size, 32, 32, n_actions],
            learning_rate=self.learning_rate
        )
        self.target_model.copy_from(self.model)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        # Normalize state input to [0, 1] range
        state_input = np.array([[state / 100.0]])
        q_values = self.model.predict(state_input)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train model using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        self.train_step += 1
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data with normalized states and rewards
        states = np.array([t[0] / 100.0 for t in minibatch]).reshape(-1, 1)
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] * self.reward_scale for t in minibatch])  # Normalize rewards
        next_states = np.array([t[3] / 100.0 for t in minibatch]).reshape(-1, 1)
        dones = np.array([t[4] for t in minibatch])
        
        # Compute target Q-values using TARGET network (more stable)
        q_next = self.target_model.predict(next_states)
        targets = rewards + self.gamma * np.amax(q_next, axis=1) * (1 - dones)
        
        # Update Q-values for the taken actions using ONLINE network
        q_current = self.model.predict(states)
        q_current[np.arange(self.batch_size), actions] = targets
        
        # Train online model
        self.model.train_batch(states, q_current)
        
        # Periodically update target network
        if self.train_step % self.target_update_freq == 0:
            self.target_model.copy_from(self.model)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# PART 4: BASELINE POLICY


def fixed_reorder_policy(inventory, reorder_point=20, order_amount_idx=4):
    """
    Simple (s, S) inventory policy
    Reorder when inventory falls below reorder_point
    """
    if inventory < reorder_point:
        return order_amount_idx  # Order 40 units
    return 0  # No order



# PART 5: POLICY EVALUATION


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


def run_detailed_simulation(policy_func, env, episodes=50):
    """
    Run simulation and collect detailed per-step metrics for graphing.
    Returns aggregated daily averages across episodes.
    """
    all_inventories = []
    all_revenues = []
    all_holding_costs = []
    all_stockout_costs = []
    all_demands = []
    all_sales = []
    all_profits = []
    episode_profits = []

    for e in range(episodes):
        state = env.reset()
        ep_inv, ep_rev, ep_hc, ep_sc, ep_dem, ep_sal, ep_prof = [], [], [], [], [], [], []
        done = False
        ep_total_profit = 0

        while not done:
            action = policy_func(state[0])
            next_state, profit, done, info = env.step(action)
            state = next_state
            ep_inv.append(next_state[0])
            ep_rev.append(info['revenue'])
            ep_hc.append(info['holding_cost'])
            ep_sc.append(info['stockout_cost'])
            ep_dem.append(info['demand'])
            ep_sal.append(info['sales'])
            ep_prof.append(profit)
            ep_total_profit += profit

        all_inventories.append(ep_inv)
        all_revenues.append(ep_rev)
        all_holding_costs.append(ep_hc)
        all_stockout_costs.append(ep_sc)
        all_demands.append(ep_dem)
        all_sales.append(ep_sal)
        all_profits.append(ep_prof)
        episode_profits.append(ep_total_profit)

    # Average across episodes for each day
    n_days = 90
    avg_inv = [np.mean([ep[d] for ep in all_inventories if d < len(ep)]) for d in range(n_days)]
    avg_rev = [np.mean([ep[d] for ep in all_revenues if d < len(ep)]) for d in range(n_days)]
    avg_hc = [np.mean([ep[d] for ep in all_holding_costs if d < len(ep)]) for d in range(n_days)]
    avg_sc = [np.mean([ep[d] for ep in all_stockout_costs if d < len(ep)]) for d in range(n_days)]
    avg_dem = [np.mean([ep[d] for ep in all_demands if d < len(ep)]) for d in range(n_days)]
    avg_sal = [np.mean([ep[d] for ep in all_sales if d < len(ep)]) for d in range(n_days)]
    avg_prof = [np.mean([ep[d] for ep in all_profits if d < len(ep)]) for d in range(n_days)]

    return {
        'avg_inventory': avg_inv,
        'avg_revenue': avg_rev,
        'avg_holding_cost': avg_hc,
        'avg_stockout_cost': avg_sc,
        'avg_demand': avg_dem,
        'avg_sales': avg_sal,
        'avg_daily_profit': avg_prof,
        'episode_profits': episode_profits,
        'total_revenue': np.mean([sum(ep) for ep in all_revenues]),
        'total_holding_cost': np.mean([sum(ep) for ep in all_holding_costs]),
        'total_stockout_cost': np.mean([sum(ep) for ep in all_stockout_costs]),
    }



# PART 6: MAIN TRAINING AND EVALUATION


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" REINFORCEMENT LEARNING FOR INVENTORY MANAGEMENT")
    print(" Currency: Indian Rupees (INR)")
    print(" Implementation: Pure NumPy DQN")
    print("="*70 + "\n")
    
    # Display Environment Parameters
    print("ENVIRONMENT PARAMETERS:")
    print(f"   Max Inventory Capacity: 100 units")
    print(f"   Average Daily Demand: 20 units")
    print(f"   Lead Time: 3 days")
    print(f"   Unit Selling Price: Rs.{2.0 * USD_TO_INR:.2f}")
    print(f"   Holding Cost: Rs.{0.1 * USD_TO_INR:.2f} per unit/day")
    print(f"   Stockout Cost: Rs.{1.0 * USD_TO_INR:.2f} per unmet demand")
    print(f"   Episode Length: 90 days\n")
    
    # Setup
    env = InventoryEnv()
    state_size = 1
    n_actions = env.n_actions
    agent = DQNAgent(state_size, n_actions)
    episodes = 500
    
    # Training
    print("--- Starting DQN Agent Training ---\n")
    start_time = time.time()
    profits_history = []
    epsilon_history = []
    
    for e in range(episodes):
        state = env.reset()
        total_profit = 0
        
        for step in range(100):  # Max steps per episode
            action = agent.act(state[0])
            next_state, profit, done, _ = env.step(action)
            total_profit += profit
            
            # Store experience
            agent.remember(state[0], action, profit, next_state[0], done)
            state = next_state
            
            if done:
                break
        
        profits_history.append(total_profit)
        epsilon_history.append(agent.epsilon)
        agent.replay()  # Train after each episode
        
        # Progress updates
        if (e + 1) % 50 == 0:
            avg_profit = np.mean(profits_history[-50:])
            print(f"Episode: {e + 1:4d}/{episodes} | "
                  f"Avg Profit (last 50): Rs.{avg_profit:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    print(f"\n--- Training Completed in {training_time:.2f} seconds ---\n")
    
    # Evaluation
    print("--- Evaluating Policies ---\n")
    eval_episodes = 200
    
    # DQN Policy
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    
    def dqn_policy(inventory):
        return agent.act(inventory)
    
    dqn_profit, dqn_unmet = run_simulation(dqn_policy, env, eval_episodes)
    print(f"DQN Agent")
    print(f"   Average Profit: Rs.{dqn_profit:,.2f} per 90-day cycle")
    print(f"   Unmet Demand: {dqn_unmet:.2f} units\n")
    
    # Fixed Policy
    fixed_profit, fixed_unmet = run_simulation(fixed_reorder_policy, env, eval_episodes)
    print(f"Fixed Reorder Policy")
    print(f"   Average Profit: Rs.{fixed_profit:,.2f} per 90-day cycle")
    print(f"   Unmet Demand: {fixed_unmet:.2f} units\n")
    
    # Detailed simulations for graphing
    print("--- Running detailed simulations for graphs ---\n")
    dqn_details = run_detailed_simulation(dqn_policy, env, episodes=50)
    fixed_details = run_detailed_simulation(fixed_reorder_policy, env, episodes=50)
    
    # Performance Comparison
    improvement = ((dqn_profit - fixed_profit) / abs(fixed_profit)) * 100 if fixed_profit != 0 else 0
    additional_profit = dqn_profit - fixed_profit
    service_level_dqn = (1 - (dqn_unmet / (20 * 90))) * 100
    service_level_fixed = (1 - (fixed_unmet / (20 * 90))) * 100
    
    print("=" * 70)
    print(" PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\n Additional Profit (DQN vs Fixed): Rs.{additional_profit:,.2f} per cycle")
    print(f" Profit Improvement: {improvement:.2f}%")
    print(f" DQN Service Level: {service_level_dqn:.1f}%")
    print(f" Fixed Service Level: {service_level_fixed:.1f}%")
    print(f" Service Level Improvement: {service_level_dqn - service_level_fixed:.1f}%\n")
    
    # Annual Projection
    cycles_per_year = 365 / 90
    annual_additional_profit = additional_profit * cycles_per_year
    print(f" Projected Annual Additional Profit: Rs.{annual_additional_profit:,.2f}\n")
    
    # =====================================================================
    # PART 7: COMPREHENSIVE MATPLOTLIB VISUALIZATIONS
    # =====================================================================
    print("--- Generating Comprehensive Visualizations ---\n")
    
    # Set global style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'grid.alpha': 0.3,
    })
    
    days = np.arange(1, 91)
    policies_labels = ['DQN Agent', 'Fixed Policy']
    color_dqn = '#2563eb'       # Blue
    color_fixed = '#dc2626'     # Red
    color_dqn_light = '#93c5fd'
    color_fixed_light = '#fca5a5'
    
    # =====================================================================
    # FIGURE 1: Training Progress with Epsilon Decay (2 subplots)
    # =====================================================================
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle('Figure 1: DQN Training Dynamics', fontsize=16, fontweight='bold', y=1.02)
    
    # 1a: Training Profit Curve
    ax1a.plot(profits_history, alpha=0.3, linewidth=0.8, color=color_dqn, label='Episode Profit')
    moving_avg = [np.mean(profits_history[max(0, i-20):i+1]) for i in range(len(profits_history))]
    ax1a.plot(moving_avg, color='#dc2626', linewidth=2.5, label='Moving Avg (20 episodes)')
    ax1a.axhline(y=np.mean(profits_history[-50:]), color='#16a34a', linestyle='--',
                 linewidth=1.5, label=f'Final Avg: Rs.{np.mean(profits_history[-50:]):,.0f}')
    ax1a.set_title('Training Progress - Episode Profits')
    ax1a.set_xlabel('Episode')
    ax1a.set_ylabel('Total Profit (Rs.)')
    ax1a.legend(loc='lower right', fontsize=10)
    ax1a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    
    # 1b: Epsilon Decay
    ax1b.plot(epsilon_history, color='#7c3aed', linewidth=2)
    ax1b.axhline(y=0.01, color='gray', linestyle=':', linewidth=1, label='epsilon min = 0.01')
    ax1b.fill_between(range(len(epsilon_history)), epsilon_history, alpha=0.15, color='#7c3aed')
    ax1b.set_title('Exploration Rate (Epsilon) Decay')
    ax1b.set_xlabel('Episode')
    ax1b.set_ylabel('Epsilon')
    ax1b.legend(fontsize=10)
    ax1b.set_ylim(-0.05, 1.05)
    
    fig1.tight_layout()
    fig1.savefig('fig1_training_dynamics.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig1_training_dynamics.png")
    
    # =====================================================================
    # FIGURE 2: Profit & Service Level Comparison (2 bar charts)
    # =====================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Figure 2: DQN vs Fixed Policy - Key Metrics Comparison',
                  fontsize=16, fontweight='bold', y=1.02)
    
    # 2a: Profit Comparison
    profits_vals = [dqn_profit, fixed_profit]
    bars = ax2a.bar(policies_labels, profits_vals, color=[color_dqn, color_fixed],
                    alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, profits_vals):
        ax2a.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(abs(v) for v in profits_vals)*0.01,
                  f'Rs.{val:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2a.set_title('Average Profit per 90-Day Cycle')
    ax2a.set_ylabel('Profit (Rs.)')
    ax2a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    # Improvement annotation
    ax2a.annotate(f'+{improvement:.1f}%', xy=(0.5, max(profits_vals)*0.5),
                  fontsize=18, fontweight='bold', ha='center', color='#16a34a',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#dcfce7', edgecolor='#16a34a'))
    
    # 2b: Service Level Comparison
    sl_vals = [service_level_dqn, service_level_fixed]
    bars_sl = ax2b.bar(policies_labels, sl_vals, color=['#0ea5e9', '#f59e0b'],
                       alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
    for bar, val in zip(bars_sl, sl_vals):
        ax2b.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2b.set_title('Service Level (Demand Fulfillment)')
    ax2b.set_ylabel('Service Level (%)')
    ax2b.set_ylim([min(sl_vals) - 5, 102])
    ax2b.axhline(y=95, color='gray', linestyle=':', linewidth=1, label='95% Target')
    ax2b.legend(fontsize=10)
    
    fig2.tight_layout()
    fig2.savefig('fig2_profit_service_comparison.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig2_profit_service_comparison.png")
    
    # =====================================================================
    # FIGURE 3: Cost Breakdown - Stacked Bar & Pie Charts
    # =====================================================================
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle('Figure 3: Revenue & Cost Breakdown (Avg per 90-Day Cycle)',
                  fontsize=16, fontweight='bold', y=1.02)
    
    # Data for cost breakdown
    dqn_rev_total = dqn_details['total_revenue']
    dqn_hc_total = dqn_details['total_holding_cost']
    dqn_sc_total = dqn_details['total_stockout_cost']
    fixed_rev_total = fixed_details['total_revenue']
    fixed_hc_total = fixed_details['total_holding_cost']
    fixed_sc_total = fixed_details['total_stockout_cost']
    
    # 3a: Grouped bar - Revenue vs Costs
    x_pos = np.arange(2)
    width = 0.25
    ax3a.bar(x_pos - width, [dqn_rev_total, fixed_rev_total], width, label='Revenue',
             color='#22c55e', edgecolor='black', linewidth=1)
    ax3a.bar(x_pos, [dqn_hc_total, fixed_hc_total], width, label='Holding Cost',
             color='#f97316', edgecolor='black', linewidth=1)
    ax3a.bar(x_pos + width, [dqn_sc_total, fixed_sc_total], width, label='Stockout Cost',
             color='#ef4444', edgecolor='black', linewidth=1)
    ax3a.set_xticks(x_pos)
    ax3a.set_xticklabels(policies_labels)
    ax3a.set_title('Revenue vs Costs')
    ax3a.set_ylabel('Amount (Rs.)')
    ax3a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    ax3a.legend(fontsize=9)
    
    # 3b: Pie - DQN cost breakdown
    dqn_net_profit = dqn_rev_total - dqn_hc_total - dqn_sc_total
    sizes_dqn = [max(0, dqn_net_profit), dqn_hc_total, dqn_sc_total]
    labels_dqn = [f'Net Profit\nRs.{dqn_net_profit:,.0f}',
                  f'Holding Cost\nRs.{dqn_hc_total:,.0f}',
                  f'Stockout Cost\nRs.{dqn_sc_total:,.0f}']
    colors_pie = ['#22c55e', '#f97316', '#ef4444']
    ax3b.pie(sizes_dqn, labels=labels_dqn, colors=colors_pie, autopct='%1.1f%%',
             startangle=90, textprops={'fontsize': 9})
    ax3b.set_title('DQN Agent - Cost Split')
    
    # 3c: Pie - Fixed policy cost breakdown
    fixed_net_profit = fixed_rev_total - fixed_hc_total - fixed_sc_total
    sizes_fixed = [max(0, fixed_net_profit), fixed_hc_total, fixed_sc_total]
    labels_fixed = [f'Net Profit\nRs.{fixed_net_profit:,.0f}',
                    f'Holding Cost\nRs.{fixed_hc_total:,.0f}',
                    f'Stockout Cost\nRs.{fixed_sc_total:,.0f}']
    ax3c.pie(sizes_fixed, labels=labels_fixed, colors=colors_pie, autopct='%1.1f%%',
             startangle=90, textprops={'fontsize': 9})
    ax3c.set_title('Fixed Policy - Cost Split')
    
    fig3.tight_layout()
    fig3.savefig('fig3_cost_breakdown.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig3_cost_breakdown.png")
    
    # =====================================================================
    # FIGURE 4: Inventory Levels Over Time (side-by-side)
    # =====================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig4.suptitle('Figure 4: Average Daily Inventory Levels (DQN vs Fixed)',
                  fontsize=16, fontweight='bold', y=1.02)
    
    # 4a: DQN Inventory
    ax4a.fill_between(days, dqn_details['avg_inventory'], alpha=0.3, color=color_dqn)
    ax4a.plot(days, dqn_details['avg_inventory'], color=color_dqn, linewidth=2, label='DQN Inventory')
    ax4a.axhline(y=np.mean(dqn_details['avg_inventory']), color=color_dqn,
                 linestyle='--', linewidth=1.5,
                 label=f'Mean: {np.mean(dqn_details["avg_inventory"]):.1f} units')
    ax4a.axhline(y=20, color='red', linestyle=':', linewidth=1, label='Reorder Point (20)')
    ax4a.set_title('DQN Agent - Inventory')
    ax4a.set_xlabel('Day')
    ax4a.set_ylabel('Inventory Level (units)')
    ax4a.legend(fontsize=9)
    ax4a.set_ylim(0, 105)
    
    # 4b: Fixed Policy Inventory
    ax4b.fill_between(days, fixed_details['avg_inventory'], alpha=0.3, color=color_fixed)
    ax4b.plot(days, fixed_details['avg_inventory'], color=color_fixed, linewidth=2, label='Fixed Inventory')
    ax4b.axhline(y=np.mean(fixed_details['avg_inventory']), color=color_fixed,
                 linestyle='--', linewidth=1.5,
                 label=f'Mean: {np.mean(fixed_details["avg_inventory"]):.1f} units')
    ax4b.axhline(y=20, color='red', linestyle=':', linewidth=1, label='Reorder Point (20)')
    ax4b.set_title('Fixed Policy - Inventory')
    ax4b.set_xlabel('Day')
    ax4b.legend(fontsize=9)
    ax4b.set_ylim(0, 105)
    
    fig4.tight_layout()
    fig4.savefig('fig4_inventory_levels.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig4_inventory_levels.png")
    
    # =====================================================================
    # FIGURE 5: Demand vs Fulfillment Over Time
    # =====================================================================
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig5.suptitle('Figure 5: Demand vs Sales Fulfillment Over 90 Days',
                  fontsize=16, fontweight='bold', y=1.01)
    
    # 5a: DQN demand vs sales
    ax5a.bar(days, dqn_details['avg_demand'], alpha=0.4, color='#94a3b8', label='Demand', width=1)
    ax5a.bar(days, dqn_details['avg_sales'], alpha=0.7, color=color_dqn, label='Sales (Fulfilled)', width=1)
    dqn_fulfill_rate = [s/d*100 if d > 0 else 100 for s, d in
                        zip(dqn_details['avg_sales'], dqn_details['avg_demand'])]
    ax5a.set_title(f'DQN Agent - Avg Fulfillment Rate: {np.mean(dqn_fulfill_rate):.1f}%')
    ax5a.set_ylabel('Units')
    ax5a.legend(fontsize=10)
    
    # 5b: Fixed demand vs sales
    ax5b.bar(days, fixed_details['avg_demand'], alpha=0.4, color='#94a3b8', label='Demand', width=1)
    ax5b.bar(days, fixed_details['avg_sales'], alpha=0.7, color=color_fixed, label='Sales (Fulfilled)', width=1)
    fixed_fulfill_rate = [s/d*100 if d > 0 else 100 for s, d in
                          zip(fixed_details['avg_sales'], fixed_details['avg_demand'])]
    ax5b.set_title(f'Fixed Policy - Avg Fulfillment Rate: {np.mean(fixed_fulfill_rate):.1f}%')
    ax5b.set_xlabel('Day')
    ax5b.set_ylabel('Units')
    ax5b.legend(fontsize=10)
    
    fig5.tight_layout()
    fig5.savefig('fig5_demand_fulfillment.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig5_demand_fulfillment.png")
    
    # =====================================================================
    # FIGURE 6: Cumulative Profit Over Time
    # =====================================================================
    fig6, ax6 = plt.subplots(figsize=(14, 6))
    fig6.suptitle('Figure 6: Cumulative Daily Profit - DQN vs Fixed Policy',
                  fontsize=16, fontweight='bold', y=1.01)
    
    cum_dqn = np.cumsum(dqn_details['avg_daily_profit'])
    cum_fixed = np.cumsum(fixed_details['avg_daily_profit'])
    
    ax6.plot(days, cum_dqn, color=color_dqn, linewidth=2.5, label='DQN Agent')
    ax6.plot(days, cum_fixed, color=color_fixed, linewidth=2.5, label='Fixed Policy')
    ax6.fill_between(days, cum_fixed, cum_dqn,
                     where=(np.array(cum_dqn) >= np.array(cum_fixed)),
                     interpolate=True, alpha=0.2, color='#22c55e', label='DQN Advantage')
    ax6.fill_between(days, cum_fixed, cum_dqn,
                     where=(np.array(cum_dqn) < np.array(cum_fixed)),
                     interpolate=True, alpha=0.2, color='#ef4444', label='Fixed Advantage')
    
    ax6.set_xlabel('Day')
    ax6.set_ylabel('Cumulative Profit (Rs.)')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    ax6.legend(fontsize=11, loc='upper left')
    # Annotate final values
    ax6.annotate(f'DQN: Rs.{cum_dqn[-1]:,.0f}', xy=(90, cum_dqn[-1]),
                 fontsize=11, fontweight='bold', color=color_dqn,
                 xytext=(-80, 15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color=color_dqn))
    ax6.annotate(f'Fixed: Rs.{cum_fixed[-1]:,.0f}', xy=(90, cum_fixed[-1]),
                 fontsize=11, fontweight='bold', color=color_fixed,
                 xytext=(-80, -25), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color=color_fixed))
    
    fig6.tight_layout()
    fig6.savefig('fig6_cumulative_profit.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig6_cumulative_profit.png")
    
    # =====================================================================
    # FIGURE 7: Daily Profit Distribution (Box + Violin)
    # =====================================================================
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))
    fig7.suptitle('Figure 7: Profit Distribution Comparison',
                  fontsize=16, fontweight='bold', y=1.02)
    
    # 7a: Episode profit distribution (box plot)
    bp = ax7a.boxplot([dqn_details['episode_profits'], fixed_details['episode_profits']],
                      tick_labels=policies_labels, patch_artist=True, widths=0.5,
                      medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(color_dqn_light)
    bp['boxes'][1].set_facecolor(color_fixed_light)
    bp['boxes'][0].set_edgecolor(color_dqn)
    bp['boxes'][1].set_edgecolor(color_fixed)
    ax7a.set_title('Episode Profit Distribution (Box Plot)')
    ax7a.set_ylabel('Total Profit per Episode (Rs.)')
    ax7a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    
    # 7b: Daily profit violin plot
    vp = ax7b.violinplot([dqn_details['avg_daily_profit'], fixed_details['avg_daily_profit']],
                         positions=[1, 2], showmeans=True, showmedians=True)
    vp['bodies'][0].set_facecolor(color_dqn_light)
    vp['bodies'][0].set_edgecolor(color_dqn)
    vp['bodies'][1].set_facecolor(color_fixed_light)
    vp['bodies'][1].set_edgecolor(color_fixed)
    ax7b.set_xticks([1, 2])
    ax7b.set_xticklabels(policies_labels)
    ax7b.set_title('Average Daily Profit Distribution (Violin Plot)')
    ax7b.set_ylabel('Daily Profit (Rs.)')
    ax7b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    
    fig7.tight_layout()
    fig7.savefig('fig7_profit_distribution.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig7_profit_distribution.png")
    
    # =====================================================================
    # FIGURE 8: Comprehensive Dashboard Summary
    # =====================================================================
    fig8, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig8.suptitle('Figure 8: Comprehensive Performance Dashboard - DQN vs Fixed Policy',
                  fontsize=18, fontweight='bold', y=1.01)
    
    # 8a: Overlay Inventory Levels
    ax8a = axes[0, 0]
    ax8a.plot(days, dqn_details['avg_inventory'], color=color_dqn, linewidth=2,
              label='DQN Agent', alpha=0.9)
    ax8a.plot(days, fixed_details['avg_inventory'], color=color_fixed, linewidth=2,
              label='Fixed Policy', alpha=0.9)
    ax8a.axhline(y=20, color='#f59e0b', linestyle='--', linewidth=1.5,
                 label='Reorder Point')
    ax8a.set_title('Inventory Levels Over Time')
    ax8a.set_xlabel('Day')
    ax8a.set_ylabel('Inventory (units)')
    ax8a.legend(fontsize=9)
    ax8a.set_ylim(0, 105)
    
    # 8b: Daily Holding Cost
    ax8b = axes[0, 1]
    ax8b.plot(days, dqn_details['avg_holding_cost'], color=color_dqn, linewidth=2,
              label='DQN Agent', alpha=0.9)
    ax8b.plot(days, fixed_details['avg_holding_cost'], color=color_fixed, linewidth=2,
              label='Fixed Policy', alpha=0.9)
    ax8b.set_title('Daily Holding Cost')
    ax8b.set_xlabel('Day')
    ax8b.set_ylabel('Holding Cost (Rs.)')
    ax8b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    ax8b.legend(fontsize=9)
    
    # 8c: Daily Stockout Cost
    ax8c = axes[1, 0]
    ax8c.plot(days, dqn_details['avg_stockout_cost'], color=color_dqn, linewidth=2,
              label='DQN Agent', alpha=0.9)
    ax8c.plot(days, fixed_details['avg_stockout_cost'], color=color_fixed, linewidth=2,
              label='Fixed Policy', alpha=0.9)
    ax8c.set_title('Daily Stockout Cost')
    ax8c.set_xlabel('Day')
    ax8c.set_ylabel('Stockout Cost (Rs.)')
    ax8c.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    ax8c.legend(fontsize=9)
    
    # 8d: Summary metrics table as text
    ax8d = axes[1, 1]
    ax8d.axis('off')
    table_data = [
        ['Metric', 'DQN Agent', 'Fixed Policy', 'Delta'],
        ['Avg Profit (Rs.)', f'Rs.{dqn_profit:,.0f}', f'Rs.{fixed_profit:,.0f}',
         f'+Rs.{additional_profit:,.0f}'],
        ['Service Level', f'{service_level_dqn:.1f}%', f'{service_level_fixed:.1f}%',
         f'+{service_level_dqn - service_level_fixed:.1f}%'],
        ['Unmet Demand', f'{dqn_unmet:.0f} units', f'{fixed_unmet:.0f} units',
         f'{dqn_unmet - fixed_unmet:+.0f} units'],
        ['Holding Cost', f'Rs.{dqn_hc_total:,.0f}', f'Rs.{fixed_hc_total:,.0f}',
         f'Rs.{dqn_hc_total - fixed_hc_total:+,.0f}'],
        ['Stockout Cost', f'Rs.{dqn_sc_total:,.0f}', f'Rs.{fixed_sc_total:,.0f}',
         f'Rs.{dqn_sc_total - fixed_sc_total:+,.0f}'],
        ['Annual Projection', f'Rs.{dqn_profit * cycles_per_year:,.0f}',
         f'Rs.{fixed_profit * cycles_per_year:,.0f}',
         f'+Rs.{annual_additional_profit:,.0f}'],
    ]
    
    table = ax8d.table(cellText=table_data[1:], colLabels=table_data[0],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#1e293b')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f9ff' if i % 2 == 0 else 'white'
        for j in range(4):
            table[i, j].set_facecolor(color)
    
    ax8d.set_title('Summary Metrics Table', fontsize=14, fontweight='bold', pad=20)
    
    fig8.tight_layout()
    fig8.savefig('fig8_dashboard.png', dpi=200, bbox_inches='tight')
    print("[OK] Saved fig8_dashboard.png")
    
    print("\n" + "=" * 70)
    print(" ALL 8 FIGURES SAVED SUCCESSFULLY!")
    print("=" * 70)
    print(" * fig1_training_dynamics.png")
    print(" * fig2_profit_service_comparison.png")
    print(" * fig3_cost_breakdown.png")
    print(" * fig4_inventory_levels.png")
    print(" * fig5_demand_fulfillment.png")
    print(" * fig6_cumulative_profit.png")
    print(" * fig7_profit_distribution.png")
    print(" * fig8_dashboard.png")
    print("=" * 70)
    
    # Summary Table (console)
    print("\n" + "=" * 70)
    print(" DETAILED COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} {'DQN Agent':<20} {'Fixed Policy':<20}")
    print("-" * 70)
    print(f"{'Average Profit (Rs.)':<30} {f'Rs.{dqn_profit:,.2f}':<20} {f'Rs.{fixed_profit:,.2f}':<20}")
    print(f"{'Unmet Demand (units)':<30} {f'{dqn_unmet:.2f}':<20} {f'{fixed_unmet:.2f}':<20}")
    print(f"{'Service Level (%)':<30} {f'{service_level_dqn:.2f}%':<20} {f'{service_level_fixed:.2f}%':<20}")
    print(f"{'Profit Improvement':<30} {f'+{improvement:.2f}%':<20} {'Baseline':<20}")
    print(f"{'Additional Profit (Rs.)':<30} {f'+Rs.{additional_profit:,.2f}':<20} {'-':<20}")
    print("=" * 70)
    
    print("\n TRAINING AND EVALUATION COMPLETE!")
    print(f" The DQN agent generates Rs.{additional_profit:,.2f} more profit per cycle")
    print(f" Projected annual additional profit: Rs.{annual_additional_profit:,.2f}")
    print("=" * 70)
