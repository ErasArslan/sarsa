import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

'''Größe 10x10'''
SIZE = 10
'''
Mögliche Bewegung:
4 - links, rechts, hoch, runter
8 - links, rechts, hoch, runter, linkshoch, linksrunter, rechtshoch, rechtsrunter
 '''
Possible_MOVE = 4
'''
Wind:
0 - Windstille
1 - Standardwind ein
3 - Wirkürlicher wind mit einer Wahrscheinlichkeit von 1/3
'''
WIND = 1

'''Anzahl MAX. EPISODEN'''
MAX_EPISODES = 25000
'''Zeigt Zwischenstand nach *Episoden'''
SHOW_AFTER_EPISODES = 1000
'''Konstanten zur Berechnung von SARSA'''
epsilon = 0.1
LEARNING_RATE = 0.1
'''Hilfswerte zum Plotten etc.'''
start_q_table = None
DISCOUNT = 0.95
MOVE_PENALTY = 1
ISLAND_REWARD = 25
'''Farben(RGB) etc.'''
SAYLINGBOAT_N = 1
ISLAND_N = 2
GRID_N = 3
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (255, 250, 255)}

class BlobIsland:
    def __init__(self):
        self.x = 3
        self.y = 7

class BlobAgent:
    def __init__(self):
        self.x = 3
        self.y = 0

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def action(self, choice):
        '''
        Bewegungen: links, rechts, hoch runter (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=0, y=1)
        elif choice == 1:
            self.move(x=1, y=0)
        elif choice == 2:
            self.move(x=-1, y=0)
        elif choice == 3:
            self.move(x=0, y=-1)
        elif choice == 4:
            self.move(x=1, y=1)
        elif choice == 5:
            self.move(x=-1 , y=-1)
        elif choice == 6:
            self.move(x=-1 , y=1)
        elif choice == 7:
            self.move(x=1 , y=-1)
        # print(f"{self.x},{self.y}")

    def move(self, x=False, y=False):
        if not x:
            self.x += -x
        else:
            self.x += x

        if not y:

            self.y += -y
        else:
            self.y += y

        '''
        Aufgabe a,b
        Standardwind wie in der Aufgabe vorgegeben
        '''
        if WIND == 1:
            if self.y == 3:
                self.x = self.x - 1
            elif self.y == 4:
                self.x = self.x - 1
            elif self.y == 5:
                self.x = self.x - 1
            elif self.y == 8:
                self.x = self.x - 1
            elif self.y == 6:
                self.x = self.x - 2
            elif self.y == 7:
                self.x = self.x - 2
        elif WIND == 2:
            if self.y == 2 or self.y == 3 or self.y == 4 or self.y == 7:
               s = 1
            elif self.y == 5 or self.y == 6:
               s = 2
            else:
                s = 0
            choice = np.random.randomint(0, 2)
            if choice == 0:
               y_stoch = s
            elif choice == 1:
               y_stoch = s - 1
            elif choice == 2:
               y_stoch = s + 1
            self.y = self + y_stoch

        '''
         Aufgabe a,b,c
         Grenzen werden gesteckt x: 0-10, y: 0-7
         '''
        if self.x < 0:
            self.x = 0
        elif self.x > 7 - 1:
            self.x = 7 - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

if start_q_table is None:
    q_table = {}
    for i in range(-SIZE + 1, SIZE):
        for ii in range(-SIZE + 1, SIZE):
            for iii in range(-SIZE + 1, SIZE):
                for iiii in range(-SIZE + 1, SIZE):
                    #Bewegung mit 4 oder 8
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(Possible_MOVE)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(MAX_EPISODES):
    agent = BlobAgent()
    island = BlobIsland()
    if episode % SHOW_AFTER_EPISODES == 0:
        print(f"Bei #{episode} ist Epsilon {epsilon}")
        print(f"{SHOW_AFTER_EPISODES} ep Durchschnitt: {np.mean(episode_rewards[-SHOW_AFTER_EPISODES:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    COST = 0
    for i in range(400):
        #time.sleep(0.05)
        obs = (agent - island, agent - island)
        # print(f"x{agent.x} y{agent.y}")
        #print(obs)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
            print(np.argmax(q_table[obs]))
        else:
            action = np.random.randint(0, Possible_MOVE)
        agent.action(action)
        COST += 1
        if agent.x == island.x and agent.y == island.y:
            reward = ISLAND_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (agent - island, agent - island)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        if reward == ISLAND_REWARD:
            new_q = ISLAND_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            '''
            Erkennbarer grid
            '''
            env[0][0] = d[GRID_N]
            env[0][2] = d[GRID_N]
            env[0][4] = d[GRID_N]
            env[0][6] = d[GRID_N]
            env[0][8] = d[GRID_N]

            env[1][1] = d[GRID_N]
            env[1][3] = d[GRID_N]
            env[1][5] = d[GRID_N]
            env[1][7] = d[GRID_N]
            env[1][9] = d[GRID_N]

            env[2][0] = d[GRID_N]
            env[2][2] = d[GRID_N]
            env[2][4] = d[GRID_N]
            env[2][6] = d[GRID_N]
            env[2][8] = d[GRID_N]

            env[3][1] = d[GRID_N]
            env[3][3] = d[GRID_N]
            env[3][5] = d[GRID_N]
            env[3][7] = d[GRID_N]
            env[3][9] = d[GRID_N]

            env[4][0] = d[GRID_N]
            env[4][2] = d[GRID_N]
            env[4][4] = d[GRID_N]
            env[4][6] = d[GRID_N]
            env[4][8] = d[GRID_N]

            env[5][1] = d[GRID_N]
            env[5][3] = d[GRID_N]
            env[5][5] = d[GRID_N]
            env[5][7] = d[GRID_N]
            env[5][9] = d[GRID_N]

            env[6][0] = d[GRID_N]
            env[6][2] = d[GRID_N]
            env[6][4] = d[GRID_N]
            env[6][6] = d[GRID_N]
            env[6][8] = d[GRID_N]
            ''''''

            env[island.x][island.y] = d[ISLAND_N]
            env[agent.x][agent.y] = d[SAYLINGBOAT_N]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300))
            cv2.imshow("Projekt 11: Segeln lernen", np.array(img))
            if reward == ISLAND_REWARD:
                print(f'Kosten: {COST}')
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == ISLAND_REWARD:
            break

    # print(episode_reward)
    episode_rewards.append(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_AFTER_EPISODES,)) / SHOW_AFTER_EPISODES, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
if WIND == 0:
    plt.title(f"Wind:AUS, Bewegungsrichtungen: {Possible_MOVE}")
elif WIND == 1:
    plt.title(f"Wind:Standard, Bewegungsrichtungen: {Possible_MOVE}")
elif WIND == 2:
    plt.title(f"Wind:Willkürlich, Bewegungsrichtungen: {Possible_MOVE}")
plt.ylabel(f"Belohnung/{SHOW_AFTER_EPISODES}")
plt.xlabel("Episoden")
plt.show()

with open(f"./table/plot-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)