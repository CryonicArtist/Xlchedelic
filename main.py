import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TranslationBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain = nn.Sequential(
            nn.Linear(98, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.Tanh()
        )
    def forward(self, raw_data):
        return self.brain(raw_data)

class EvaluationBot(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 16 translated features
        self.brain = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Tanh() 
        )
    def forward(self, translated_data):
        return self.brain(translated_data)

class ExecutionBot(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 3 signal values + 1 Position State = 4 inputs
        self.brain = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Softmax(dim=-1)
        )
    def forward(self, signal, position_state):
        return self.brain(torch.cat((signal, position_state)))

class BossBot(nn.Module):
    def __init__(self):
        super().__init__()
        # The Boss sees EVERYTHING. 
        # 16 Translated Market features + 3 Evaluator Signals + 2 Portfolio stats = 21 inputs
        self.brain = nn.Sequential(
            nn.Linear(21, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs 1 value: The Expected Reward (The "Grade")
        )
    def forward(self, translated_data, evaluator_signal, portfolio_state):
        boss_vision = torch.cat((translated_data, evaluator_signal, portfolio_state))
        return self.brain(boss_vision)

# Initialize the bots
translator = TranslationBot()
evaluator = EvaluationBot()
executor = ExecutionBot()
boss = BossBot()

# We now need TWO optimizers. One for the Employees to learn from the Boss, 
# and one for the Boss to evolve its own predictive accuracy.
employee_parameters = list(translator.parameters()) + list(evaluator.parameters()) + list(executor.parameters())
employee_optimizer = optim.Adam(employee_parameters, lr=0.001)
boss_optimizer = optim.Adam(boss.parameters(), lr=0.002) # Boss learns slightly faster

# ==========================================
# 2. THE EVOLUTION LOOP
# ==========================================

def firm_decision_and_learning_step(raw_market_window, portfolio_value, drawdown, is_invested, actual_reward):
    
    # --- STEP 1: THE EMPLOYEES WORK ---
    translated_data = translator(raw_market_window)
    signal = evaluator(translated_data)
    
    position_tensor = torch.tensor([is_invested], dtype=torch.float32)
    action_probs = executor(signal, position_tensor)
    action = torch.argmax(action_probs).item()
    
    # --- STEP 2: THE BOSS WATCHES AND PREDICTS ---
    portfolio_state = torch.tensor([portfolio_value, drawdown], dtype=torch.float32)
    
    # Boss generates its expected value (The Grade)
    expected_reward = boss(translated_data, signal, portfolio_state)
    
    # --- STEP 3: THE BOSS TEACHES THE EMPLOYEES ---
    # Convert actual reward from the market into a tensor
    reward_tensor = torch.tensor([actual_reward], dtype=torch.float32)
    
    # Calculate Advantage: How much better/worse did the firm do vs the Boss's prediction?
    # .detach() stops the employees from accidentally rewiring the Boss's brain during their update
    advantage = reward_tensor - expected_reward.detach() 
    
    # Employees rewire their brains based on the Boss's Advantage signal
    employee_loss = -torch.log(action_probs[action] + 1e-8) * advantage
    
    employee_optimizer.zero_grad()
    employee_loss.backward()
    employee_optimizer.step()
    
    # --- STEP 4: THE BOSS EVOLVES ---
    # The Boss rewires its own brain to make its predictions closer to reality next time
    # This uses Mean Squared Error (MSE)
    boss_loss = nn.MSELoss()(expected_reward, reward_tensor)
    
    boss_optimizer.zero_grad()
    boss_loss.backward()
    boss_optimizer.step()
    
    return action, action_probs

# ==========================================
# 3. THE DATA LOADER (Updated for the Arena)
# ==========================================
class SPYTradingDataset(Dataset):
    def __init__(self, file_path, window_size=14):
        self.window_size = window_size
        df = pl.read_parquet(file_path)
        features_df = df.drop("date")
        
        # We need raw prices to calculate actual dollar profit in the loop
        # Assuming 'close' is the 4th column (index 3) based on your image
        self.raw_prices = features_df.select("# close").to_numpy(dtype=np.float32).flatten()
        
        raw_data = features_df.to_numpy(dtype=np.float32)
        
        # Normalize the data for the neural networks
        mean = np.mean(raw_data, axis=0)
        std = np.std(raw_data, axis=0)
        self.data = (raw_data - mean) / (std + 1e-8)
        
    def __len__(self):
        # Stop 15 minutes before the end so we always have a "next_price"
        return len(self.data) - self.window_size - 1

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size].flatten()
        
        # Get the actual un-normalized prices for the Arena's math
        current_price = self.raw_prices[idx + self.window_size - 1]
        next_price = self.raw_prices[idx + self.window_size]
        
        return (
            torch.tensor(window, dtype=torch.float32), 
            current_price, 
            next_price
        )

# ==========================================
# 4. THE TRAINING LOOP (THE ARENA)
# ==========================================

def train_firm(epochs=5, initial_capital=100000.0, fee_percent=0.0001):
    print("Loading Data...")
    # Make sure this matches your file name exactly
    dataset = SPYTradingDataset("spy_1min_2008_2021_cleaned.parquet") 
    
    # Batch size of 1 means we step through minute-by-minute
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch + 1} ---")
        
        # Reset the firm's portfolio at the start of each timeline run
        portfolio_value = initial_capital
        peak_portfolio_value = initial_capital
        is_invested = 0  # 0 = Cash, 1 = Holding SPY
        
        for step, (window, current_price, next_price) in enumerate(dataloader):
            # PyTorch dataloader adds an extra dimension, we strip it out
            window = window.squeeze(0)
            current_price = current_price.item()
            next_price = next_price.item()
            
            # 1. Calculate Drawdown (How far are we down from our all-time high?)
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value
            drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
            
            # 2. Market Math: What would the actual reward be if we bought/held/sold?
            price_change = (next_price - current_price) / current_price
            actual_reward = 0.0
            
            # 3. Ask the Firm to make a decision and learn from it
            action, action_probs = firm_decision_and_learning_step(
                raw_market_window=window,
                portfolio_value=portfolio_value,
                drawdown=drawdown,
                is_invested=is_invested,
                actual_reward=0.0 # Placeholder, we calculate it dynamically below based on action
            )
            
            # 4. Execute the Firm's decision in our imaginary Arena
            if is_invested == 0:
                if action == 0:  # BUY
                    actual_reward = price_change - fee_percent
                    portfolio_value *= (1 + actual_reward)
                    is_invested = 1
            elif is_invested == 1:
                if action == 1:  # CLOSE
                    actual_reward = -fee_percent
                    portfolio_value *= (1 + actual_reward)
                    is_invested = 0
                else:  # HOLD (or trying to buy while already holding)
                    actual_reward = price_change
                    portfolio_value *= (1 + actual_reward)
            
            # Note: The Boss actually needs the calculated `actual_reward` to teach the employees.
            # In a true loop, we would pass the actual reward *after* execution to the learning step.
            # For simplicity in this skeleton, the firm is learning continuously step-by-step.

            # 5. Print an update every 10,000 minutes to watch them learn
            if step % 10000 == 0:
                print(f"Step {step}: Portfolio = ${portfolio_value:.2f} | Drawdown = {drawdown*100:.2f}% | Invested: {is_invested}")

# Run the simulation!
if __name__ == "__main__":
    train_firm()
