import sys
from FourRooms import FourRooms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def run_training(strategy, stochastic=False, runs=3, episodes=2000):
    os.makedirs('plots', exist_ok=True)
    rewards = np.zeros((runs, episodes))
    
    for run in range(runs):
        #passing stochastic to Four Rooms
        fourRoomsObj = FourRooms('simple', stochastic=stochastic)
        q_table = np.zeros((11, 11, 2, 4))
        param = strategy['init']
        
        for ep in tqdm(range(episodes), desc=f"{strategy['name']} Run {run+1}"):
            fourRoomsObj.newEpoch()
            state, packages_left, done = fourRoomsObj.getPosition(), fourRoomsObj.getPackagesRemaining(), False
            param = max(param * strategy['decay'], strategy['min'])
            
            while not done:
                x, y = state
                x, y = max(0, min(x, 10)), max(0, min(y, 10))
                s_pack = int(packages_left > 0)
                
                # Action selection
                if strategy['name'] == 'ε-greedy':
                    action = np.random.randint(4) if np.random.random() < param else np.argmax(q_table[x, y, s_pack])
                else:  # Boltzmann
                    probs = np.exp(q_table[x, y, s_pack]/param) / np.sum(np.exp(q_table[x, y, s_pack]/param))
                    action = np.random.choice(4, p=probs)
                
                # Environment step
                cell, new_state, packages_left, done = fourRoomsObj.takeAction(action)
                reward = 10 if (cell > 0 and packages_left == 0) else -1 if (done and packages_left > 0) else -0.1
                rewards[run, ep] += reward
                
                # Q-update
                new_x, new_y = new_state
                new_x, new_y = max(0, min(new_x, 10)), max(0, min(new_y, 10))
                best_next = np.argmax(q_table[new_x, new_y, int(packages_left > 0)])
                q_table[x, y, s_pack, action] += 0.1 * (reward + 0.99 * q_table[new_x, new_y, int(packages_left > 0), best_next] - q_table[x, y, s_pack, action])
                state = new_state
        
        if run == runs-1:  # Save path for last run
            suffix = 'stochastic' if stochastic else ""
            fourRoomsObj.showPath(-1, savefig=f"plots/{strategy['name']}{suffix}_path_.png")
    
    return rewards

def plot_results(strategies, all_rewards):
    suffix = "_stochastic" if stochastic else ""
    title = "Exploration strategy Comparism"
    if stochastic:
        title += "(Stochastic Actions)"
    plt.figure(figsize=(10, 5))
    for i, (name, rewards) in enumerate(zip([s['name'] for s in strategies], all_rewards)):
        mean, std = np.mean(rewards, axis=0), np.std(rewards, axis=0)
        plt.plot(mean, label=name, color=['blue', 'orange'][i])
        plt.fill_between(range(1000), mean-std, mean+std, alpha=0.2, color=['blue', 'orange'][i])
    plt.title("Exploration Strategy Comparison")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"plots/_exploration_comparison{suffix}.png")

if __name__ == "__main__":
    stochastic = '-stochastic' in sys.argv

    strategies = [
        {'name': 'ε-greedy', 'init': 0.2, 'decay': 0.995, 'min': 0.01},
        {'name': 'Boltzmann', 'init': 1.0, 'decay': 0.995, 'min': 0.1}
    ]
    
    all_rewards = [run_training(s, stochastic=stochastic) for s in strategies]
    plot_results(strategies, all_rewards)