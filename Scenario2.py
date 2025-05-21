from FourRooms import FourRooms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def run_training(runs=5, episodes=1000):
    os.makedirs('plots', exist_ok=True)
    rewards = np.zeros((runs, episodes))
    
    for run in range(runs):
        fourRoomsObj = FourRooms('multi') 
        q_table = np.zeros((11, 11, 5, 4))
        epsilon = 0.2
        epsilon_min = 0.01
        epsilon_decay = 0.995
        
        for ep in tqdm(range(episodes), desc=f"Run {run+1}"):
            fourRoomsObj.newEpoch()
            state = fourRoomsObj.getPosition()
            packages_left = fourRoomsObj.getPackagesRemaining()
            done = False
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            while not done:
                x, y = state
                x, y = max(0, min(x, 10)), max(0, min(y, 10))
                s_pack = packages_left
                
                # ε-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[x, y, s_pack])
                
                # Take action
                cell, new_state, new_packages_left, done = fourRoomsObj.takeAction(action)
                
                # My Reward selection
                reward = -0.1  #penalty
                if new_packages_left < packages_left:
                    reward += 2  # package collected
                if done and new_packages_left == 0:
                    reward += 10  # all packages collected,finished
                elif done and new_packages_left > 0:
                    reward -= 1  # ends without collecting all packages
                
                rewards[run, ep] += reward
                
                # Q-learning update
                new_x, new_y = new_state
                new_x, new_y = max(0, min(new_x, 10)), max(0, min(new_y, 10))
                best_next = np.argmax(q_table[new_x, new_y, new_packages_left])
                
                q_table[x, y, s_pack, action] += 0.1 * (reward + 0.99 * q_table[new_x, new_y, new_packages_left, best_next] - q_table[x, y, s_pack, action])
                
                state = new_state
                packages_left = new_packages_left
        
        if run == runs - 1:
            fourRoomsObj.showPath(-1, savefig="plots/multi_path.png")
    
    return rewards

if __name__ == "__main__":
    all_rewards = run_training()
    
