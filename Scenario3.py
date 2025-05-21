from FourRooms import FourRooms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def run_training(runs=5, episodes=1000):
    os.makedirs('plots', exist_ok=True)
    rewards = np.zeros((runs, episodes))
    
    for run in range(runs):
        fourRoomsObj = FourRooms('rgb') 
        q_table = np.zeros((11, 11, 5, 4))
        epsilon = 0.2
        epsilon_min = 0.01
        epsilon_decay = 0.995
        
        for ep in tqdm(range(episodes), desc=f"Run {run+1}"):
            fourRoomsObj.newEpoch()
            state = fourRoomsObj.getPosition()
            next_package = 1
            done = False
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            while not done:
                x, y = state
                x, y = max(0, min(x, 10)), max(0, min(y, 10))
                
                # ε-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[x, y, next_package])
                
                # Take action
                cell, new_state, _, done = fourRoomsObj.takeAction(action)
                
                # My Reward selection
                reward = -0.1  #small penalty step
                if cell == next_package:
                    reward += 5
                    next_package += 1
                    if next_package > 3:
                        next_package = 0 # thus all packages collected
                        done = True 
                elif cell in [1, 2, 3] and cell != next_package:
                    # for wrong package picked, episode ends with penalty
                    reward -= 5
                    done = True  
                
                rewards[run, ep] += reward
                
                # Q-learning update
                new_x, new_y = new_state
                new_x, new_y = max(0, min(new_x, 10)), max(0, min(new_y, 10))
                best_next = np.argmax(q_table[new_x, new_y, next_package])
                
                q_table[x, y, next_package, action] += 0.1 * (reward + 0.99 * q_table[new_x, new_y, next_package, best_next] - q_table[x, y, next_package, action])
                
                state = new_state
        
        #final path shown after last episode run
        if run == runs - 1:
            fourRoomsObj.showPath(-1, savefig="plots/rgb_path.png")
    
    return rewards

if __name__ == "__main__":
    all_rewards = run_training()
    
