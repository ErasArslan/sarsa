import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
sizex = 7
sizey = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ISLAND_REACHED = 25

COST = 0
epsilon = 0.1
EPS_DECAY = 0.9998

SHOW_EVERY = 3000

start_q_table = None # or filename
LEARNING_RATE = 0.1
DISCOUNT = 0.95

Boot_Colour = 1
ISLAND_COLOUR = 2

d = {1: (255, 175, 0),
     2: (0, 255, 0)
     }

class BlobIsland:
    def __init__(self, COST=None):
        self.x = 4
        self.y = 8

class BlobPlayer:
    def __init__(self):
        print("start")
        time.sleep(2)
        self.x = 0
        self.y = 4
        print(self.x)
        print(self.y)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y-other.y)

    def action(self, choice):
        #time.sleep(0.1)
        if choice == 0:
            self.move(x=0 , y=1)
        elif choice == 1:
            self.move(x=1 , y=0)
        elif choice == 2:
            self.move(x=-1 , y=0)
        elif choice == 3:
            self.move(x=0 , y=-1)
        #Aufgabe C
        # elif choice == 4:
        #     self.move(x=1, y=1)
        # elif choice == 5:
        #     self.move(x=-1 , y=-1)
        # elif choice == 6:
        #     self.move(x=-1 , y=1)
        # elif choice == 7:
        #     self.move(x=1 , y=-1)

    def move(self, x=False, y=False):
        print("ybefore")
        print(self.y)
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

        #standardwind a-b
        if self.x == 3:
            self.y = self.y + 1
        elif self.x == 4:
            self.y = self.y + 1
        elif self.x == 5:
            self.y = self.y + 1
        elif self.x == 8:
            self.y = self.y + 1
        elif self.x == 6:
            self.y = self.y + 2
        elif self.x == 7:
            self.y = self.y + 2

        #randomwind c
        #if self.x == 3 or self.x == 4 or self.x == 5 or self.x == 8:
        #    s = 1
        #elif self.x == 6 or self.x == 7:
        #    s = 2
        #choice = np.random.randomint(0,2)
        #if choice == 0:
        #    y_stoch = s
        #elif choice == 1:
        #    y_stoch = s - 1
        #elif choice == 2:
        #    y_stoch = s + 1
        #if self.x == 3 or self.x == 4 or self.x == 5 or self.x == 8 or self.x == 6 or self.x == 7:
        #    self.y = self + y_stoch

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range (HM_EPISODES):
    player = BlobPlayer()
    island = BlobIsland()
    enemy = BlobPlayer()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    COST = 0
    for i in range(20000):
        obs = (player - island, (0, 0))
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        COST += 1
        print(f'Cost: {COST}')
        player.action(action)

        if player.x == island.x and player.y == island.y:
            reward = ISLAND_REACHED
        else:
            reward = -MOVE_PENALTY
        new_obs = (player - island, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == ISLAND_REACHED:
            new_q = ISLAND_REACHED
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward+DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[island.x][island.y] = d[ISLAND_COLOUR]
            env[player.x][player.y] = d[Boot_Colour]

            img = Image.fromarray(env, "RGB")
            print("player at position")
            print(player)
            img = img.resize((400, 400))
            cv2.imshow("", np.array(img))
            if reward == ISLAND_REACHED:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        episode_reward += reward
        if reward == ISLAND_REACHED:
            break


    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/ SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"rward{SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle","wb") as f:
    pickle.dump(q_table, f)