import numpy as np
import matplotlib.pyplot as plt

class ai:

    def __init__(self, inputs, outputs, epochs=100):
        self.m = inputs
        self.n = outputs
        self.epochs = epochs

    def __call__(self, x, y, ax):
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
                
            for k, u in enumerate(self.axis):
                self.W[u] -= 1.5*gradient[u]
                ax[k].cla()
                mo, no = self.W[u].shape
                mx, my = np.meshgrid(list(range(mo)), list(range(no)))
                ax[k].contourf(mx, my, self.W[u].T, cmap='jet_r')
            plt.pause(0.001)


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

model = ai(len(x), len(y))
model.build_parameters()

fig = plt.figure(figsize=(10, 5))
ax = [fig.add_subplot(1, 3, i+1) for i in range(3)]

x = np.random.randn(200, 6)
y = np.random.random((200, 3))

for ix, iy in zip(x, y):
    model(ix, iy, ax)

plt.show()
