# Reinforcement Learning-Based Stock Trading Agent Using PPO

This project implements a stock trading agent using **Proximal Policy Optimization (PPO)**, a reinforcement learning (RL) method. The agent is trained to optimize trading strategies based on historical market data and technical indicators. The goal is to maximize long-term profits and outperform traditional strategies.

## Libraries & Installation

To run this project, you will need the following libraries:

- `stable-baselines3` – For implementing the PPO algorithm.
- `gym` – For creating a custom stock trading environment.
- `pandas` – For handling historical market data and feature engineering.
- `numpy` – For numerical operations.
- `matplotlib` – For visualizing the performance of the agent.

You can install the required libraries using pip:

```bash
pip install stable-baselines3 gym pandas numpy matplotlib
```

Make sure you have Python 3.6 or higher installed.

## Project Structure

The project is organized as follows:

```
.
├── environment.py           # Custom trading environment
├── agent.py                 # PPO agent implementation
├── train.py                 # Training script
├── test.py                  # Testing and evaluating the agent
├── data/                    # Folder for storing stock data
├── results/                 # Folder for storing training results and plots
├── README.md                # Project description
└── requirements.txt         # List of required libraries
```

- **`environment.py`**: Contains the custom stock trading environment using `gym`.
- **`agent.py`**: Defines the PPO agent, including the model and training logic.
- **`train.py`**: Main script for training the agent using PPO.
- **`test.py`**: Script to evaluate the trained agent and visualize performance.
- **`data/`**: A folder to store the stock data used in training.
- **`results/`**: Stores the plots and training results.

## Setting Up on Your PC

1. **Clone the repository**:
   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/stock-trading-ppo.git
   cd stock-trading-ppo
   ```

2. **Install Dependencies**:
   Install the required libraries using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   Download the historical stock data (you can use sources like Yahoo Finance or Alpha Vantage) and place it in the `data/` folder. Make sure the data is in CSV format with columns like `Date`, `Open`, `Close`, `High`, `Low`, `Volume`.

4. **Modify the environment**:
   If you want to use different stock data or customize the environment, you can edit the `environment.py` file. Adjust the data loading logic or add new features as needed.

5. **Training the Agent**:
   Run the `train.py` script to start training the agent. It will use PPO to optimize the trading strategy based on the data and indicators.

   ```bash
   python train.py
   ```

6. **Evaluate the Agent**:
   Once trained, you can test the agent’s performance using the `test.py` script:

   ```bash
   python test.py
   ```

## License

This project is open-source and available under the MIT License.

---
