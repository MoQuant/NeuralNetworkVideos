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


class rnnai:

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
                    if epoch == 0:
                        self.L[i] = x.T.dot(self.W[i]) + b
                    else:
                        self.L[i] = x.T.dot(self.W[i]) + b + self.rW[i] @ self.rSL[i] + b
                else:
                    self.L[i] = self.L[i+1].T @ self.W[i] + b
                self.SL[i] = self.sigmoid(self.L[i])

            for i in self.raxis:
                self.rL[i] = self.SL[i].T @ self.rW[i]
                self.rSL[i] = self.sigmoid(self.rL[i])

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

            gradient2 = {}
            for i in self.axis:
                error = delta.T @ self.rW[i]
                delta = error*self.sigmoid(self.rL[i], dv=True)
                gradient2[i] = delta 
                
            for u in self.axis:
                self.W[u] -= gradient[u]

            for u in self.axis:
                self.rW[u] -= gradient2[u]

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

        self.rW = {}
        self.rL = {}
        self.rSL = {}

        for i in self.axis:
            self.W[i] = np.random.random((i, i-1))
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)

            self.rW[i] = np.random.random((i-1, i-2))
            self.rL[i] = np.zeros(i-1)
            self.rSL[i] = np.zeros(i-1)

x = [0.11, 0.33, 0.41, 0.05, 0.03, 0.6]
y = [0.15, 0.15, 0.15]

epochs = 300

modelA = rnnai(len(x), len(y), epochs=epochs)
modelA.build_parameters()
modelA(x, y)

modelB = ai(len(x), len(y), epochs=epochs)
modelB.build_parameters()
modelB(x, y)

rnn = []
fnn = []


for I, J in zip(modelA.the_errors, modelB.the_errors):
    rnn.append(np.mean(I))
    fnn.append(np.mean(J))
    ax.cla()
    ax.set_title('RNN vs. FNN')
    ax.plot(rnn, color='red', label='RNN')
    ax.plot(fnn, color='orange', label='FNN')
    ax.legend()
    plt.pause(0.1)

plt.show()




