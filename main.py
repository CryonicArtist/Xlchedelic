import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Initialize the bots and send them to the GPU
translator = TranslationBot().to(device)
evaluator = EvaluationBot().to(device)
executor = ExecutionBot().to(device)
boss = BossBot().to(device)

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

    # .detach() severs the graph so the Boss and Employees don't trip over each other
    expected_reward = boss(translated_data.detach(), signal.detach(), portfolio_state)

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
        # Load the data
        df = pl.read_parquet(file_path)

        # --- NEW DATE FILTER ---
        print("Filtering data for 2010-2015...")

        # Tell Polars exactly how to read your date string: "%Y-%m-%d %H:%M:%S"
        df = df.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )

        # Now filter by year
        df = df.filter(
            (pl.col("date").dt.year() >= 2010) &
            (pl.col("date").dt.year() <= 2015)
        )

        print(f"Data filtered! Remaining rows: {df.height}")
        # ------------------------
        features_df = df.drop("date")

        # We need raw prices to calculate actual dollar profit in the loop
        # Assuming 'close' is the 4th column (index 3) based on your image
        self.raw_prices = features_df.select("close").to_numpy().astype(np.float32).flatten()

        raw_data = features_df.to_numpy().astype(np.float32)

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

def train_firm(initial_capital=100000.0, fee_percent=0.0001):
    # Set to the current local directory
    save_dir = "./"
    
    # 1. Auto-detect the last completed epoch in your local folder
    last_epoch = 0
    while os.path.exists(f"{save_dir}boss_epoch_{last_epoch + 1}.pth"):
        last_epoch += 1
        
    current_epoch = last_epoch + 1
    print(f"\n--- Starting Training for Epoch {current_epoch} ---")
    
    # 2. Load previous brains if they exist
    if last_epoch > 0:
        print(f"Loading saved brains from Epoch {last_epoch}...")
        translator.load_state_dict(torch.load(f"{save_dir}translator_epoch_{last_epoch}.pth"))
        evaluator.load_state_dict(torch.load(f"{save_dir}evaluator_epoch_{last_epoch}.pth"))
        executor.load_state_dict(torch.load(f"{save_dir}executor_epoch_{last_epoch}.pth"))
        boss.load_state_dict(torch.load(f"{save_dir}boss_epoch_{last_epoch}.pth"))
        print("Brains loaded successfully!")

    print("Loading Data...")
    dataset = SPYTradingDataset("spy_1min_2008_2021_cleaned.parquet") 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Epsilon Math (Decays based on the current epoch)
    base_epsilon = 0.15 
    current_epsilon = base_epsilon * (0.5 ** (current_epoch - 1))
    print(f"Exploration Rate (Epsilon): {current_epsilon*100:.1f}%")
    
    portfolio_value = initial_capital
    peak_portfolio_value = initial_capital
    is_invested = 0  # 0 = Cash, 1 = Holding SPY
    
    # THE 1-EPOCH LOOP
    for step, (window, current_price, next_price) in enumerate(dataloader):
        window = window.squeeze(0).to(device)
        current_price = current_price.item()
        next_price = next_price.item()
        
        # 1. Calculate Drawdown
        if portfolio_value > peak_portfolio_value:
            peak_portfolio_value = portfolio_value
        drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
        
        # --------------------------------------------------
        # 2. FIRM THINKS (Forward Pass)
        # --------------------------------------------------
        portfolio_state = torch.tensor([portfolio_value, drawdown], dtype=torch.float32, device=device)
        position_tensor = torch.tensor([is_invested], dtype=torch.float32, device=device)
        
        translated_data = translator(window)
        signal = evaluator(translated_data)
        action_probs = executor(signal, position_tensor)
        
        # --- THE EXPLORATION LOGIC ---
        if random.random() < current_epsilon:
            action = random.randint(0, 2)
        else:
            action = torch.argmax(action_probs).item()
        
        expected_reward = boss(translated_data.detach(), signal.detach(), portfolio_state)
        
        # --------------------------------------------------
        # 3. ARENA EXECUTES (Calculate Actual Reward)
        # --------------------------------------------------
        price_change = (next_price - current_price) / current_price
        actual_reward = 0.0
        
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
            else:  # HOLD 
                actual_reward = price_change
                portfolio_value *= (1 + actual_reward)
                
        # --------------------------------------------------
        # 4. FIRM LEARNS (Backward Pass)
        # --------------------------------------------------
        reward_tensor = torch.tensor([actual_reward], dtype=torch.float32, device=device)
        
        advantage = reward_tensor - expected_reward.detach()
        
        employee_loss = -torch.log(action_probs[action] + 1e-8) * advantage
        employee_optimizer.zero_grad()
        employee_loss.backward()
        employee_optimizer.step()
        
        boss_loss = nn.MSELoss()(expected_reward, reward_tensor)
        boss_optimizer.zero_grad()
        boss_loss.backward()
        boss_optimizer.step()
        
        if step % 10000 == 0:
            print(f"Step {step}: Portfolio = ${portfolio_value:.2f} | Drawdown = {drawdown*100:.2f}% | Invested: {is_invested}")

    # ======================================================
    # SAVE CODE FOR THIS SINGLE EPOCH
    # ======================================================
    print(f"Saving checkpoints for Epoch {current_epoch} locally...")
    
    torch.save(translator.state_dict(), f"{save_dir}translator_epoch_{current_epoch}.pth")
    torch.save(evaluator.state_dict(), f"{save_dir}evaluator_epoch_{current_epoch}.pth")
    torch.save(executor.state_dict(), f"{save_dir}executor_epoch_{current_epoch}.pth")
    torch.save(boss.state_dict(), f"{save_dir}boss_epoch_{current_epoch}.pth")
    
    print(f"Save complete for Epoch {current_epoch}! Run the script again to start the next one.")

if __name__ == "__main__":
    train_firm()

