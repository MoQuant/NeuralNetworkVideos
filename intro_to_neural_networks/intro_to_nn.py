import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

class ai:

    def __init__(self, inputs, outputs, epochs=100):
        self.m = inputs
        self.n = outputs
        self.epochs = epochs

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        b = -1

        self.the_errors = []
        
        for epoch in range(self.epochs):
            # Forwad Propigation
            for i in self.axis:
                if i == self.axis[0]:
                    self.L[i] = x.T.dot(self.W[i]) + b
                else:
                    self.L[i] = self.L[i+1].T @ self.W[i] + b
                self.SL[i] = self.sigmoid(self.L[i])

            # Backward Propigation
            gradient = {}
            for i in self.raxis:
                if i == self.raxis[0]:
                    error = (y - self.SL[i])**2
                    self.the_errors.append(error.tolist())
                    delta = error*self.sigmoid(self.L[i], dv=True)
                else:
                    error = self.W[i-1] @ delta
                    delta = error*self.sigmoid(self.L[i], dv=True)
                    
                gradient[i] = delta
                
            for u in self.axis:
                self.W[u] -= gradient[u]

        self.the_errors = np.array(self.the_errors)

    def sigmoid(self, x, dv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if dv:
            return f*(1 - f)
        return f

    def build_parameters(self):
        m, n = self.m, self.n

        self.axis = list(range(m, n, -1))
        self.raxis = self.axis[::-1]

        self.W = {}
        self.L = {}
        self.SL = {}

        for i in self.axis:
            self.W[i] = np.random.random((i, i-1))
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)


x = [1, 1, 1, 0, 0, 0]
y = [0.35, 0.45, 0.25]

model = ai(6, 3, epochs=250)
model.build_parameters()

model(x, y)

y1, y2, y3 = [], [], []

for x1, x2, x3 in model.the_errors:
    y1.append(x1)
    y2.append(x2)
    y3.append(x3)

    ax.cla()
    ax.set_title('Neural Network Error Minimization')
    ax.plot(y1, color='red', label='Variable 1')
    ax.plot(y2, color='orange', label='Variable 2')
    ax.plot(y3, color='blue', label='Variable 3')
    ax.legend()
    plt.pause(0.01)

plt.show()

