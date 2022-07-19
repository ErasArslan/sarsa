'''
GIT-Repository: https://github.com/ErasArslan/sarsa.git
'''

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

'''
Mögliche Bewegung:
4 - links, rechts, hoch, runter
8 - links, rechts, hoch, runter, linkshoch, linksrunter, rechtshoch, rechtsrunter
9 - links, rechts, hoch, runter, linkshoch, linksrunter, rechtshoch, rechtsrunter, vom Wind treiben lassen
 '''
POSSIBLE_MOVE = 8
'''
Wind:
0 - Windstille
1 - Standardwind ein
2 - Willkürlicher Wind mit einer Wahrscheinlichkeit von 1/3 für alle gleich null Werte
3 - Willkürlicher Wind mit einer Wahrscheinlichkeit von 1/3 für alle ungleich null Werte
4 - Willkürlicher Wind zu jeder Episode verändert von 1/3 für alle ungleich null Werte
5 - Willkürlicher Wind zu jedem Schritt verändert von 1/3 für alle ungleich null Werte
'''
WIND = 1
'''Konstanten zur Berechnung von SARSA'''
epsilon = 0.1
learnrate = 0.1
gamma = 0.96
'''Anzahl MAX. EPISODEN'''
MAX_EPISODES = 1000
'''Zeigt Zwischenstand nach *Episoden'''
SHOW_AFTER_EPISODES = 50
ROUNDS = 300

'''Hilfswerte zum Plotten etc.'''
start_q_table = None
style.use("ggplot")
#sigma = 0
#SIGMA_ARRAY = [77, 65, 68, 69, 45, 66, 89, 45, 69, 82, 65, 83]
MOVE_P = 1
ISLAND_REWARD = 25
'''Größe 10x10'''
SIZE = 10
'''Farben(RGB) etc.'''
SAYLINGBOAT_N = 1
ISLAND_N = 2
GRID_N = 3
WINDS_N = 4
WINDN_N = 5
d = {1: (19, 69, 139),  #Boatcolor Brown
     2: (0, 255, 0),    #Islandcolor Lime
     3: (255, 175, 0),  #Gridcolor White
     4: (139, 0, 139),  #SouthWind Yellow
     5: (0, 0, 255)}    #NorthWind Red

LEGENDLABEL = "DEFAULT"
WIND1 = np.random.randint(0, 3)
WIND2 = np.random.randint(0, 3)
WIND3 = np.random.randint(0, 3)
WIND4 = np.random.randint(0, 3)
WIND5 = np.random.randint(0, 3)
WIND6 = np.random.randint(0, 3)
WIND7 = np.random.randint(0, 3)
WIND8 = np.random.randint(0, 3)
WIND9 = np.random.randint(0, 3)
WIND10 = np.random.randint(0, 3)
EPS_DECAY = 0.9998

class IslandClass:
    def __init__(self):
        self.x = 3
        self.y = 7

class SailingBoatClass:
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
        elif choice == 8:
            self.move(x=0 , y=0)
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
            elif self.y == 0:
                #print (f"WIND1 = {WIND1}")
                if WIND1 == 0:
                    self.x = self.x
                elif WIND1 == 1:
                    self.x = self.x - 1
                else:
                    self.x = self.x + 1
            elif self.y == 1:
                #print(f"WIND2 = {WIND2}")
                if WIND2 == 0:
                    self.x = self.x
                elif WIND2 == 1:
                    self.x = self.x + 1
                elif WIND2 == 2:
                    self.x = self.x - 1
            elif self.y == 2:
                #print(f"WIND3 = {WIND3}")
                if WIND3 == 0:
                    self.x = self.x
                elif WIND3 == 1:
                    self.x = self.x + 1
                elif WIND3 == 2:
                    self.x = self.x - 1
            elif self.y == 9:
                #print(f"WIND10 = {WIND10}")
                if WIND10 == 0:
                    self.x = self.x
                elif WIND10 == 1:
                    self.x = self.x + 1
                elif WIND10 == 2:
                    self.x = self.x - 1
        elif WIND == 3:
            if self.y == 3:
                if WIND4 == 0:
                    self.x = self.x - 1
                elif WIND4 == 1:
                    self.x = self.x + 1
                elif WIND4 == 2:
                    self.x = self.x
            elif self.y == 4:
                if WIND5 == 0:
                    self.x = self.x - 1
                elif WIND5 == 1:
                    self.x = self.x + 1
                elif WIND5 == 2:
                    self.x = self.x
            elif self.y == 5:
                if WIND6 == 0:
                    self.x = self.x - 1
                elif WIND6 == 1:
                    self.x = self.x + 1
                elif WIND6 == 2:
                    self.x = self.x
            elif self.y == 8:
                if WIND9 == 0:
                    self.x = self.x - 1
                elif WIND9 == 1:
                    self.x = self.x + 1
                elif WIND9 == 2:
                    self.x = self.x
            elif self.y == 6:
                if WIND7 == 0:
                    self.x = self.x - 2
                elif WIND7 == 1:
                    self.x = self.x - 3
                elif WIND7 == 2:
                    self.x = self.x - 1
            elif self.y == 7:
                if WIND8 == 0:
                    self.x = self.x - 2
                elif WIND8 == 1:
                    self.x = self.x - 3
                elif WIND8 == 2:
                    self.x = self.x - 1
        elif WIND == 4 or WIND == 5:
            if self.y == 3:
                if WIND4 == 0:
                    self.x = self.x - 1
                elif WIND4 == 1:
                    self.x = self.x + 1
                elif WIND4 == 2:
                    self.x = self.x
            elif self.y == 4:
                if WIND5 == 0:
                    self.x = self.x - 1
                elif WIND5 == 1:
                    self.x = self.x + 1
                elif WIND5 == 2:
                    self.x = self.x
            elif self.y == 5:
                if WIND6 == 0:
                    self.x = self.x - 1
                elif WIND6 == 1:
                    self.x = self.x + 1
                elif WIND6 == 2:
                    self.x = self.x
            elif self.y == 8:
                if WIND9 == 0:
                    self.x = self.x - 1
                elif WIND9 == 1:
                    self.x = self.x + 1
                elif WIND9 == 2:
                    self.x = self.x
            elif self.y == 6:
                if WIND7 == 0:
                    self.x = self.x - 2
                elif WIND7 == 1:
                    self.x = self.x - 3
                elif WIND7 == 2:
                    self.x = self.x - 1
            elif self.y == 7:
                if WIND8 == 0:
                    self.x = self.x - 2
                elif WIND8 == 1:
                    self.x = self.x - 3
                elif WIND8 == 2:
                    self.x = self.x - 1
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
    Q_table = {}
    for i in range(-SIZE + 1, SIZE):
        for ii in range(-SIZE + 1, SIZE):
            for iii in range(-SIZE + 1, SIZE):
                for iiii in range(-SIZE + 1, SIZE):
                    #Bewegung mit 4 oder 8
                    Q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(POSSIBLE_MOVE)]
else:
    with open(start_q_table, "rb") as f:
        Q_table = pickle.load(f)

episode_rewards = []

for episode in range(MAX_EPISODES):
    if WIND == 4:
        WIND4 = np.random.randint(0, 3)
        WIND5 = np.random.randint(0, 3)
        WIND6 = np.random.randint(0, 3)
        WIND7 = np.random.randint(0, 3)
        WIND8 = np.random.randint(0, 3)
        WIND9 = np.random.randint(0, 3)
    sailingboat = SailingBoatClass()
    island = IslandClass()
    if episode % SHOW_AFTER_EPISODES == 0:
        print(f"Bei #{episode} ist Epsilon {epsilon}")
        print(f"{SHOW_AFTER_EPISODES} ep Durchschnitt: {np.mean(episode_rewards[-SHOW_AFTER_EPISODES:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    STEPS = 0
    for i in range(ROUNDS):
        if WIND == 5:
            WIND4 = np.random.randint(0, 3)
            WIND5 = np.random.randint(0, 3)
            WIND6 = np.random.randint(0, 3)
            WIND7 = np.random.randint(0, 3)
            WIND8 = np.random.randint(0, 3)
            WIND9 = np.random.randint(0, 3)
        #time.sleep(0.005)
        #print (i)

        observation = (sailingboat - island, sailingboat - island)
        # print(f"x{agent.x} y{agent.y}")
        #print(obs)
        if np.random.random() > epsilon:
            action = np.argmax(Q_table[observation])
            #print(np.argmax(q_table[obs]))
        else:
            action = np.random.randint(0, POSSIBLE_MOVE)
        sailingboat.action(action)

        if sailingboat.x == island.x and sailingboat.y == island.y:
            reward = ISLAND_REWARD
        else:
            reward = -MOVE_P

        new_observation = (sailingboat - island, sailingboat - island)
        Q_next = np.max(Q_table[new_observation])
        Q_current = Q_table[observation][action]
        if reward == ISLAND_REWARD:
            Q_new = ISLAND_REWARD
        else:
            '''
            Sarsa 
            Q(s_t, a_t) = (1-alpha) * Q(s_t, a_t) + alpha * (c(s_t, a_t) + gamma * Q(s_t+1, a_t+1)
            '''
            Q_new = (1 - learnrate) * Q_current + learnrate * (reward + gamma * Q_next)
            STEPS += 1
            learnrate = learnrate - 1 / STEPS
            #print(learnrate)
            #print(new_q)
            #new_q = current_q + learnrate * (reward + gamma * max_future_q - current_q)
        Q_table[observation][action] = Q_new

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            '''
            Erkennbarer grid
            Jedes zweite Element im Bereich des Sees wird Weiß geplottet
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
            env[7][0] = d[ISLAND_N]
            env[7][1] = d[ISLAND_N]
            env[7][2] = d[ISLAND_N]
            env[7][3] = d[ISLAND_N]
            env[7][4] = d[ISLAND_N]
            env[7][5] = d[ISLAND_N]
            env[7][6] = d[ISLAND_N]
            env[7][7] = d[ISLAND_N]
            env[7][8] = d[ISLAND_N]
            env[7][9] = d[ISLAND_N]
            env[8][0] = d[ISLAND_N]
            env[8][1] = d[ISLAND_N]
            env[8][2] = d[ISLAND_N]
            env[8][3] = d[ISLAND_N]
            env[8][4] = d[ISLAND_N]
            env[8][5] = d[ISLAND_N]
            env[8][6] = d[ISLAND_N]
            env[8][7] = d[ISLAND_N]
            env[8][8] = d[ISLAND_N]
            env[8][9] = d[ISLAND_N]
            env[9][0] = d[ISLAND_N]
            env[9][1] = d[ISLAND_N]
            env[9][2] = d[ISLAND_N]
            env[9][3] = d[ISLAND_N]
            env[9][4] = d[ISLAND_N]
            env[9][5] = d[ISLAND_N]
            env[9][6] = d[ISLAND_N]
            env[9][7] = d[ISLAND_N]
            env[9][8] = d[ISLAND_N]
            env[9][9] = d[ISLAND_N]
            '''
            Farben je nach wind Plotten
            Unterer Bereich für Windanzeige genutzt.
            Felder 8-10 genutzt um Windstärke anzuzeigen
            Wind der gegen Norden geht ist ROT
            Wind der gegen Süden geht ist Viollet
            WIND:
            [0] Alle Winde ausgeschaltet
            [1] Wind aus der Aufgabenstellung ist eingeschaltet
            [2] Stochastischer Wind für alle werte die null sind
            [3] Stochstischer Wind für den Bereich mit vorher definiertem Wind
            '''
            if WIND == 1:
                env[7][3] = d[WINDN_N]
                env[7][4] = d[WINDN_N]
                env[7][5] = d[WINDN_N]
                env[7][6] = d[WINDN_N]
                env[8][6] = d[WINDN_N]
                env[7][7] = d[WINDN_N]
                env[8][7] = d[WINDN_N]
                env[7][8] = d[WINDN_N]
            if WIND == 2:
                env[7][3] = d[WINDN_N]
                env[7][4] = d[WINDN_N]
                env[7][5] = d[WINDN_N]
                env[7][6] = d[WINDN_N]
                env[8][6] = d[WINDN_N]
                env[7][7] = d[WINDN_N]
                env[8][7] = d[WINDN_N]
                env[7][8] = d[WINDN_N]
                ##Stochstischer WInd
                if WIND1 == 1:
                    env[7][0] = d[WINDN_N]
                elif WIND1 == 2:
                    env[7][0] = d[WINDS_N]
                if WIND2 == 1:
                    env[7][1] = d[WINDN_N]
                elif WIND2 == 2:
                    env[7][1] = d[WINDS_N]
                if WIND3 == 1:
                    env[7][2] = d[WINDN_N]
                elif WIND3 == 2:
                    env[7][2] = d[WINDS_N]
                if WIND10 == 1:
                    env[7][9] = d[WINDN_N]
                elif WIND10 == 2:
                    env[7][9] = d[WINDS_N]
            if WIND == 3 or WIND == 4 or WIND == 5:
                if WIND4 == 0:
                    env[7][3] = d[ISLAND_N]
                elif WIND4 == 1:
                    env[7][3] = d[WINDN_N]
                    env[8][3] = d[WINDN_N]
                elif WIND4 == 2:
                    env[7][3] = d[WINDN_N]
                if WIND5 == 0:
                    env[7][4] = d[ISLAND_N]
                elif WIND5 == 1:
                    env[7][4] = d[WINDN_N]
                    env[8][4] = d[WINDN_N]
                elif WIND5 == 2:
                    env[7][4] = d[WINDN_N]
                if WIND6 == 0:
                    env[7][5] = d[ISLAND_N]
                elif WIND6 == 1:
                    env[7][5] = d[WINDN_N]
                    env[8][5] = d[WINDN_N]
                elif WIND6 == 2:
                    env[7][5] = d[WINDN_N]
                if WIND9 == 0:
                    env[7][8] = d[ISLAND_N]
                elif WIND9 == 1:
                    env[7][8] = d[WINDN_N]
                    env[8][8] = d[WINDN_N]
                elif WIND9 == 2:
                    env[7][8] = d[WINDN_N]
                if WIND7 == 0:
                    env[7][6] = d[WINDN_N]
                    env[8][6] = d[WINDN_N]
                elif WIND7 == 1:
                    env[7][6] = d[WINDN_N]
                    env[8][6] = d[WINDN_N]
                    env[9][6] = d[WINDN_N]
                elif WIND7 == 2:
                    env[7][6] = d[WINDN_N]
                    env[8][6] = d[ISLAND_N]
                    env[9][6] = d[ISLAND_N]
                if WIND8 == 0:
                    env[7][7] = d[WINDN_N]
                    env[8][7] = d[WINDN_N]
                elif WIND8 == 1:
                    env[7][7] = d[WINDN_N]
                    env[8][7] = d[WINDN_N]
                    env[9][7] = d[WINDN_N]
                elif WIND8 == 2:
                    env[7][7] = d[WINDN_N]
                    env[8][7] = d[ISLAND_N]
                    env[9][7] = d[ISLAND_N]
            env[island.x][island.y] = d[ISLAND_N]
            env[sailingboat.x][sailingboat.y] = d[SAYLINGBOAT_N]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((200, 200))
            picture = cv2.imread('SarsaLegende.png')
            if WIND == 0:
                LEGENDLABEL = f"Wind:Aus Bewegung: {POSSIBLE_MOVE}"
            elif WIND == 1:
                LEGENDLABEL = f"Wind:Norden Bewegung: {POSSIBLE_MOVE}"
            elif WIND == 2:
                LEGENDLABEL= f"Wind(für null):Stochastisch Bewegung: {POSSIBLE_MOVE}"
            elif WIND == 3:
                LEGENDLABEL= f"Wind(für ungleich null):Stochastisch Bewegung: {POSSIBLE_MOVE}"
            elif WIND == 4:
                LEGENDLABEL= f"Wind(jede Episode):Stochastisch Bewegung: {POSSIBLE_MOVE}"
            elif WIND == 5:
                LEGENDLABEL= f"Wind(jeder Zustand):Stochastisch Bewegung: {POSSIBLE_MOVE}"
            cv2.imshow(LEGENDLABEL, picture)
            cv2.imshow("Projekt 11: Segeln lernen", np.array(img))
            if reward == ISLAND_REWARD:
                print(f'Kosten: {STEPS}')
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
    epsilon *= EPS_DECAY
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_AFTER_EPISODES,)) / SHOW_AFTER_EPISODES, mode='valid')
# if (sigma == 2):
#     s = "".join([chr(c) for c in SIGMA_ARRAY])
#     print(s)
plt.plot([i for i in range(len(moving_avg))], moving_avg)
if WIND == 0:
    plt.title(f"Wind:AUS, Bewegungsrichtungen: {POSSIBLE_MOVE}")
elif WIND == 1:
    plt.title(f"Wind:Standard, Bewegungsrichtungen: {POSSIBLE_MOVE}")
elif WIND == 2:
    plt.title(f"Wind:Willkürlich(gleich:null), Bewegungsrichtungen: {POSSIBLE_MOVE}")
elif WIND == 3:
    plt.title(f"Wind:Willkürlich(ungleich:null), Bewegungsrichtungen: {POSSIBLE_MOVE}")
elif WIND == 4:
    plt.title(f"Wind:Willkürlich(jede Episode), Bewegungsrichtungen: {POSSIBLE_MOVE}")
elif WIND == 5:
    plt.title(f"Wind:Willkürlich(jeder Schritt), Bewegungsrichtungen: {POSSIBLE_MOVE}")
plt.ylabel(f"Belohnung/{SHOW_AFTER_EPISODES}")
plt.xlabel("Episoden")
plt.show()

with open(f"./table/plot-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(Q_table, f)
