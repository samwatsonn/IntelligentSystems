from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

        
def plot_points(ax, X, y):
    col_a = []
    col_b = []
    for i in range(len(X)):
        if y[i] == 1:
            col_a.append(X[i])
        else:
            col_b.append(X[i])
    ax.scatter([x for (x,y) in col_a], [y for (x,y) in col_a], c='r')
    ax.scatter([x for (x,y) in col_b], [y for (x,y) in col_b], c='b')

def plot_decision_boundary(ax, X, w0, w1, w2):
    X_cpy = X.copy()
    margin = 0.5
    x_min = np.min(X_cpy[:, 0])-margin
    x_max = np.max(X_cpy[:, 0])+margin
    y_min = np.min(X_cpy[:, 1])-margin
    y_max = np.max(X_cpy[:, 1])+margin
    x = np.linspace(x_min,x_max)
    m = (-1 * w2) / w1
    c = (-1 * w0) / w1
    ax.plot(x,m*x+c)
    ax.set_xlim((x_min,x_max))
    ax.set_ylim((y_min,y_max))

def plot():
    fig,ax = plt.subplots()
    plot_points(ax, X, y)
    plot_decision_boundary(ax, X, p2.weights[0], p2.weights[1], p2.weights[2])
    plt.show()
    

def add(l1,l2):
    return [x + y for x,y in zip(l1,l2)]

def mult(x, l):
    return [x * i for i in l]

class Perceptron:

    learning_rate = 0.3

    def __init__(self, samples, outputs):
        self.samples = [[1] + sample for sample in samples]
        self.outputs = outputs
        self.weights = [1] * len(self.samples[0])

    def calc_z(self, n):
        sum = 0
        for i in range(len(self.weights)):
            sum += self.samples[n][i] * self.weights[i]
        return sum

    def threshold(self, n): 
        if n>0:
            return 1
        else:
            return -1

    def learn(self):
        missed = 0
        for n in range(len(self.samples)):
            if self.outputs[n] * self.threshold(self.calc_z(n)) <= 0:
                self.weights = add(self.weights, mult(self.outputs[n] * (self.learning_rate), self.samples[n])) 
                missed+=1
        print(f"Weights: {self.weights}")
        print(f"Number missed: {missed}")

X, y = datasets.make_classification(
    n_features=2,
    n_classes=2,
    n_samples=100,
    n_redundant=0,
    n_clusters_per_class=1
)

y_norm = [2 * x - 1 for x in y]
        
p2 = Perceptron(X.tolist(), y_norm)

if __name__ == "__main__":    

    fig,ax = plt.subplots()
    plot_points(ax, X, y)
    plot_decision_boundary(ax, X, p2.weights[0], p2.weights[1], p2.weights[2])
    plt.show()
    
    for i in range(5):
        print(f"Iteration {i}")
        p2.learn()
        plot()

