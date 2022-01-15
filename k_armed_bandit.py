import random
import matplotlib.pyplot as plt

random.seed(10)

class Bandit:

    def __init__(self, k_arms):
        self.arms=[]
        for i in range(k_arms):
            self.arms.append(random.randrange(1,100))
        print(self.arms)

    def pull(self, arm):
        return random.randrange(0,self.arms[arm])


class Action:

    def __init__(self, ID):
        self.ID = ID
        self.N = 0
        self.total = 0

    def add(self,n):
        self.N += 1
        self.total += n

    def expected_value(self):
        if self.N == 0:
            return 0
        else:
            return self.total/self.N

    def unexplored(self):
        return self.N==0

    def get_ID(self):
        return self.ID

    def get_total(self):
        return self.total

    def __lt__(self, other):
        return self.expected_value() < other.expected_value()


class Solver:

    def __init__(self, k_arms, e, T):
        self.bandit = Bandit(k_arms)
        self.N = k_arms
        self.e = e
        self.T = T
        self.actions = [Action(i) for i in range(self.N)]

    def can_explore(self):
        return len([a for a in self.actions if a.unexplored()]) > 0

    def explore(self):
        arm = random.choice([a for a in self.actions if a.unexplored()])
        arm.add(self.bandit.pull(arm.get_ID()))

    def exploit(self):
        arm = sorted([a for a in self.actions])[-1]
        arm.add(self.bandit.pull(arm.get_ID()))

    def total_reward(self):
        return sum([a.get_total() for a in self.actions])
        
    def reset_actions(self):
        self.actions = [Action(i) for i in range(self.N)]

    def e_greedy(self):
        plot_data = []
        for t in range(self.T):
            if random.random() < self.e:
                if self.can_explore():
                    self.explore()
                else:
                    self.exploit()
            else:
                self.exploit()
            plot_data.append([t+1, self.total_reward()/(t+1)])

        print(f"E-Greedy Reward: {self.total_reward()}")
        self.reset_actions()
        return plot_data

    def e_first(self):
        plot_data = []
        for t in range(self.T):
            if t < self.e * self.T:
                if self.can_explore():
                    self.explore()
                else:
                    self.exploit()
            else:
                self.exploit()
            plot_data.append([t+1, self.total_reward()/(t+1)])

        print(f"E-First Reward: {self.total_reward()}")
        self.reset_actions()
        return plot_data
        
        
if __name__ == "__main__":
    b10 = Bandit(10)
    
    solver = Solver(10, 0.01, 2000)
    eg=solver.e_greedy()
    ef=solver.e_first()

    eg_x = [i[0] for i in eg]
    eg_y = [i[1] for i in eg]

    ef_x = [i[0] for i in ef]
    ef_y = [i[1] for i in ef]
    
    plt.plot(eg_x, eg_y, label = "e-greedy")
    plt.plot(ef_x, ef_y, label = "e-first")
    plt.legend()
    plt.show()
                
                    
        
    
