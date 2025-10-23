import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

class nnet:

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

class rnnet:

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

            for i in self.raxis:
                self.Lr[i] = self.SL[i] @ self.Wr[i]  + b
                self.SLr[i] = self.sigmoid(self.Lr[i])
            

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
                error = delta @ self.Wr[i]
                delta = error*self.sigmoid(self.Lr[i], dv=True)
                gradient2[i] = delta 

            for u in self.axis:
                self.W[u] -= gradient[u]
            
            for u in self.axis:
                self.Wr[u] -= gradient2[u]

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

        self.Wr = {}
        self.Lr = {}
        self.SLr = {}

        for i in self.axis:
            self.W[i] = np.random.random((i, i-1))
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)
            self.Wr[i] = np.random.random((i-1, i-2))
            self.Lr[i] = np.zeros(i-1)
            self.SLr[i] = np.zeros(i-1)

x = [1, 1, 1, 0, 0, 0]
y = [0.35, 0.45, 0.25]

modelA = rnnet(6, 3, epochs=250)
modelA.build_parameters()
modelA(x, y)

modelB = nnet(6, 3, epochs=250)
modelB.build_parameters()
modelB(x, y)

rnn, fnn = [], []
for RNN, FNN in zip(modelA.the_errors, modelB.the_errors):
    rnn.append(np.mean(RNN))
    fnn.append(np.mean(FNN))
    ax.cla()
    ax.set_title('RNN vs. FNN')
    ax.plot(rnn, color='blue', label='RNN')
    ax.plot(fnn, color='red', label='FNN')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    plt.pause(0.1)

plt.show()



